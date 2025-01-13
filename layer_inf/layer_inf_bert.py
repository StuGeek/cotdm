import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
import warnings
from typing import Tuple
from transformers.models.bert.modeling_bert import BertAttention

logger = logging.get_logger(__name__)


class BertPreProcessOutput:
    def __init__(
        self,
        input_ids,
        position_ids,
        token_type_ids,
        inputs_embeds,
        past_key_values_length,
        extended_attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        past_key_values,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.inputs_embeds = inputs_embeds
        self.past_key_values_length = past_key_values_length
        self.extended_attention_mask = extended_attention_mask
        self.head_mask = head_mask
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_extended_attention_mask = encoder_extended_attention_mask
        self.past_key_values = past_key_values
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

    # def to(self, device):
    #     if self.input_ids != None:
    #         self.input_ids = self.input_ids.to(device)
    #     if self.position_ids != None:
    #         self.position_ids = self.position_ids.to(device)
    #     if self.token_type_ids != None:
    #         self.token_type_ids = self.token_type_ids.to(device)
    #     if self.inputs_embeds != None:
    #         self.inputs_embeds = self.inputs_embeds.to(device)
    #     if self.extended_attention_mask != None:
    #         self.extended_attention_mask = self.extended_attention_mask.to(device)
    #     if self.encoder_hidden_states != None:
    #         self.encoder_hidden_states = self.encoder_hidden_states.to(device)
    #     if self.encoder_extended_attention_mask != None:
    #         self.encoder_extended_attention_mask = self.encoder_extended_attention_mask.to(device)
    #     if self.past_key_values != None:
    #         self.past_key_values = self.past_key_values.to(device)


class BertLayerData:
    def __init__(
        self,
        all_hidden_states,
        all_self_attentions,
        all_cross_attentions,
        next_decoder_cache,
        hidden_states=None,
    ):
        self.all_hidden_states = all_hidden_states
        self.all_self_attentions = all_self_attentions
        self.all_cross_attentions = all_cross_attentions
        self.next_decoder_cache = next_decoder_cache
        self.hidden_states = hidden_states


class BertInferenceIntermediateData:
    def __init__(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=None,
        extended_attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_extended_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        all_hidden_states=None,
        all_self_attentions=None,
        all_cross_attentions=None,
        next_decoder_cache=None,
        hidden_states=None,
        layer_head_mask=None,
        past_key_value=None,
        self_attention_outputs=None,
        present_key_value=None,
        outputs=None,
    ):
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.inputs_embeds = inputs_embeds
        self.past_key_values_length = past_key_values_length
        self.extended_attention_mask = extended_attention_mask
        self.head_mask = head_mask
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_extended_attention_mask = encoder_extended_attention_mask
        self.past_key_values = past_key_values
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.all_hidden_states = all_hidden_states
        self.all_self_attentions = all_self_attentions
        self.all_cross_attentions = all_cross_attentions
        self.next_decoder_cache = next_decoder_cache
        self.hidden_states = hidden_states
        self.layer_head_mask = layer_head_mask
        self.past_key_value = past_key_value
        self.self_attention_outputs = self_attention_outputs
        self.present_key_value = present_key_value
        self.outputs = outputs


def invert_attention_mask(encoder_attention_mask):
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=torch.float32
    )  # fp16 compatibility
    encoder_extended_attention_mask = (
        1.0 - encoder_extended_attention_mask
    ) * torch.finfo(torch.float32).min

    return encoder_extended_attention_mask


def get_extended_attention_mask(attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float32

    if not (attention_mask.dim() == 2 and False):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if False:
            extended_attention_mask = (
                ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):
    """
    Prepare the head mask if needed.

    Args:
        head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (`int`):
            The number of hidden layers in the model.
        is_attention_chunked: (`bool`, *optional*, defaults to `False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
        `[None]` for each layer.
    """
    if head_mask is not None:
        pass
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask


def bert_inference_preprocess(
    # embeddings: object,
    encoded_input: object,
    config: object,
) -> object:
    input_ids = encoded_input.get("input_ids")
    attention_mask = encoded_input.get("attention_mask")
    token_type_ids = encoded_input.get("token_type_ids")
    position_ids = encoded_input.get("position_ids")
    head_mask = encoded_input.get("head_mask")
    inputs_embeds = encoded_input.get("inputs_embeds")
    encoder_hidden_states = encoded_input.get("encoder_hidden_states")
    encoder_attention_mask = encoded_input.get("encoder_attention_mask")
    past_key_values = encoded_input.get("past_key_values")
    use_cache = encoded_input.get("use_cache")
    output_attentions = encoded_input.get("output_attentions")
    output_hidden_states = encoded_input.get("output_hidden_states")
    return_dict = encoded_input.get("return_dict")

    output_attentions = (
        output_attentions if output_attentions is not None else config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else config.use_return_dict

    if config.is_decoder:
        use_cache = use_cache if use_cache is not None else config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # past_key_values_length
    past_key_values_length = (
        past_key_values[0][0].shape[2] if past_key_values is not None else 0
    )

    if attention_mask is None:
        attention_mask = torch.ones(
            ((batch_size, seq_length + past_key_values_length)), device=device
        )

    stage_embeddings_position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    stage_embeddings_token_type_ids = torch.zeros(stage_embeddings_position_ids.size(), dtype=torch.long)
    if token_type_ids is None:
        # if hasattr(embeddings, "token_type_ids"):
        buffered_token_type_ids = stage_embeddings_token_type_ids.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
            batch_size, seq_length
        )
        token_type_ids = buffered_token_type_ids_expanded
        # else:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # bert_model = BertModel(config)
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(
        attention_mask, input_shape, dtype=torch.float32
    )

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = get_head_mask(head_mask, config.num_hidden_layers)

    return BertInferenceIntermediateData(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
        extended_attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_extended_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def bert_embeddings_preprocess(stage_embeddings_position_ids, stage_embeddings_token_type_ids, intermediate_data, config):
    input_ids=intermediate_data.input_ids
    position_ids=intermediate_data.position_ids
    token_type_ids=intermediate_data.token_type_ids
    inputs_embeds=intermediate_data.inputs_embeds
    past_key_values_length=intermediate_data.past_key_values_length

    # stage_embeddings_position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    # stage_embeddings_position_ids = stage_embeddings_position_ids.to("cuda:0")
    # stage_embeddings_token_type_ids = torch.zeros(stage_embeddings_position_ids.size(), dtype=torch.long)
    # stage_embeddings_token_type_ids = stage_embeddings_token_type_ids.to("cuda:0")

    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]

    # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
    # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
    # issue #5664
    if token_type_ids is None:
        # if hasattr(stage_embeddings, "token_type_ids"):
        buffered_token_type_ids = token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        intermediate_data.token_type_ids = buffered_token_type_ids_expanded
        # else:
        #     intermediate_data.token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

    if position_ids is None:
        intermediate_data.position_ids = stage_embeddings_position_ids[:, past_key_values_length : seq_length + past_key_values_length]


def bert_word_embeddings_forward(stage_word_embeddings, intermediate_data):
    if intermediate_data.inputs_embeds is None:
        intermediate_data.inputs_embeds = stage_word_embeddings(intermediate_data.input_ids)


def bert_token_type_embeddings_forward(stage_token_type_embeddings, intermediate_data):
    token_type_embeddings = stage_token_type_embeddings(intermediate_data.token_type_ids)
    intermediate_data.hidden_states = intermediate_data.inputs_embeds + token_type_embeddings


def bert_position_embeddings_forward(stage_position_embeddings, intermediate_data, config):
    position_embedding_type = getattr(config, "position_embedding_type", "absolute")
    if position_embedding_type == "absolute":
        position_embeddings = stage_position_embeddings(intermediate_data.position_ids)
        intermediate_data.hidden_states += position_embeddings


def bert_embeddings_LayerNorm_or_dropout_forward(stage_embeddings_LayerNorm_or_dropout, intermediate_data):
    intermediate_data.hidden_states = stage_embeddings_LayerNorm_or_dropout(intermediate_data.hidden_states)


def bert_embeddings_foward(stage_embeddings, intermediate_data):
    if intermediate_data.inputs_embeds is None:
        intermediate_data.inputs_embeds = stage_embeddings.word_embeddings(intermediate_data.input_ids)
    token_type_embeddings = stage_embeddings.token_type_embeddings(intermediate_data.token_type_ids)

    intermediate_data.hidden_states = intermediate_data.inputs_embeds + token_type_embeddings
    if stage_embeddings.position_embedding_type == "absolute":
        position_embeddings = stage_embeddings.position_embeddings(intermediate_data.position_ids)
        intermediate_data.hidden_states += position_embeddings
    intermediate_data.hidden_states = stage_embeddings.LayerNorm(intermediate_data.hidden_states)
    intermediate_data.hidden_states = stage_embeddings.dropout(intermediate_data.hidden_states)


def bert_embeddings_foward2(embeddings, intermediate_data):
    return embeddings(
        input_ids=intermediate_data.input_ids,
        position_ids=intermediate_data.position_ids,
        token_type_ids=intermediate_data.token_type_ids,
        inputs_embeds=intermediate_data.inputs_embeds,
        past_key_values_length=intermediate_data.past_key_values_length,
    )


def bert_encoder_preprocess(intermediate_data, config):
    all_hidden_states = () if intermediate_data.output_hidden_states else None
    all_self_attentions = () if intermediate_data.output_attentions else None
    all_cross_attentions = (
        ()
        if intermediate_data.output_attentions and config.add_cross_attention
        else None
    )

    # if bert_encoder.gradient_checkpointing and bert_encoder.training:
    #     if use_cache:
    #         logger.warning_once(
    #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #         )
    #         use_cache = False

    next_decoder_cache = () if intermediate_data.use_cache else None

    intermediate_data.all_hidden_states = all_hidden_states
    intermediate_data.all_self_attentions = all_self_attentions
    intermediate_data.all_cross_attentions = all_cross_attentions
    intermediate_data.next_decoder_cache = next_decoder_cache


def bert_encoder_layer_attention_forward(stage_attention, intermediate_data, idx):
    if intermediate_data.output_hidden_states:
        intermediate_data.all_hidden_states = intermediate_data.all_hidden_states + (
            intermediate_data.hidden_states,
        )

    intermediate_data.layer_head_mask = (
        intermediate_data.head_mask[idx]
        if intermediate_data.head_mask is not None
        else None
    )
    intermediate_data.past_key_value = (
        intermediate_data.past_key_values[idx]
        if intermediate_data.past_key_values is not None
        else None
    )

    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = (
        intermediate_data.past_key_value[:2]
        if intermediate_data.past_key_value is not None
        else None
    )
    self_attention_outputs = stage_attention(
        intermediate_data.hidden_states,
        intermediate_data.extended_attention_mask,
        intermediate_data.layer_head_mask,
        output_attentions=intermediate_data.output_attentions,
        past_key_value=self_attn_past_key_value,
    )

    intermediate_data.self_attention_outputs = self_attention_outputs


def apply_chunking_to_forward(
    # forward_fn: Callable[..., torch.Tensor],
    stage_intermediate,
    stage_output,
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    # num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    # if num_args_in_forward_chunk_fn != len(input_tensors):
    #     raise ValueError(
    #         f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
    #         "tensors are given"
    #     )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(
            input_tensor.chunk(num_chunks, dim=chunk_dim)
            for input_tensor in input_tensors
        )
        # apply forward fn to every tuple
        output_chunks = tuple(
            # forward_fn(*input_tensors_chunk)
            stage_output(stage_intermediate(*input_tensors_chunk), *input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks)
        )
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return input_tensors


def bert_encoder_layer_crossattention(intermediate_data, config):
    if config.add_cross_attention:
        if not config.is_decoder:
            raise ValueError(
                f"{config._name_or_path} should be used as a decoder model if cross attention is added"
            )
        stage_crossattention = BertAttention(config, position_embedding_type="absolute")

    cross_attn_present_key_value = None
    if config.is_decoder and intermediate_data.encoder_hidden_states is not None:
        if not config.add_cross_attention:
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {config._name_or_path} has to be instantiated with cross-attention layers"
                " by setting `config.add_cross_attention=True`"
            )

        # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        cross_attn_past_key_value = (
            intermediate_data.past_key_value[-2:]
            if intermediate_data.past_key_value is not None
            else None
        )
        cross_attention_outputs = stage_crossattention(
            attention_output,
            intermediate_data.extended_attention_mask,
            intermediate_data.layer_head_mask,
            intermediate_data.encoder_hidden_states,
            intermediate_data.encoder_extended_attention_mask,
            cross_attn_past_key_value,
            intermediate_data.output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = (
            outputs + cross_attention_outputs[1:-1]
        )  # add cross attentions if we output attention weights

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        cross_attn_present_key_value = cross_attention_outputs[-1]
        present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_data.present_key_value = present_key_value


def bert_encoder_layer_intermediate_output_forward(
    stage_intermediate, stage_output, intermediate_data, attention_output, config
):
    # if decoder, the last output is tuple of self-attn cache
    if config.is_decoder:
        intermediate_data.outputs = intermediate_data.self_attention_outputs[1:-1]
        intermediate_data.present_key_value = intermediate_data.self_attention_outputs[
            -1
        ]
    else:
        intermediate_data.outputs = intermediate_data.self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

    intermediate_data.outputs = bert_encoder_layer_crossattention(
        intermediate_data, config
    )
    intermediate_data.outputs = (
        () if intermediate_data.outputs is None else intermediate_data.outputs
    )

    seq_len_dim = 1
    input_tensors = apply_chunking_to_forward(
        # feed_forward_chunk,
        stage_intermediate,
        stage_output,
        config.chunk_size_feed_forward,
        seq_len_dim,
        attention_output,
    )

    intermediate_output = stage_intermediate(*input_tensors)
    layer_output = stage_output(intermediate_output, *input_tensors)
    intermediate_data.outputs = (layer_output,) + intermediate_data.outputs

    # if decoder, return the attn key/values as the last output
    if config.is_decoder:
        intermediate_data.outputs = intermediate_data.outputs + (
            intermediate_data.present_key_value,
        )


def bert_encoder_layer_intermediate_forward(
    stage_intermediate, intermediate_data, attention_output, config
):
    # if decoder, the last output is tuple of self-attn cache
    if config.is_decoder:
        intermediate_data.outputs = intermediate_data.self_attention_outputs[1:-1]
        intermediate_data.present_key_value = intermediate_data.self_attention_outputs[
            -1
        ]
    else:
        intermediate_data.outputs = intermediate_data.self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

    intermediate_data.outputs = bert_encoder_layer_crossattention(
        intermediate_data, config
    )
    intermediate_data.outputs = (
        () if intermediate_data.outputs is None else intermediate_data.outputs
    )

    bert_encoder_layer_intermediate_output = stage_intermediate(*attention_output)
    return bert_encoder_layer_intermediate_output


def bert_encoder_layer_output_forward(
    stage_output, intermediate_data, bert_encoder_layer_intermediate_output, attention_output, config
):
    # print(bert_encoder_layer_intermediate_output)
    # print(attention_output)
    layer_output = stage_output(
        bert_encoder_layer_intermediate_output, attention_output
    )
    intermediate_data.outputs = (layer_output,) + intermediate_data.outputs

    # if decoder, return the attn key/values as the last output
    if config.is_decoder:
        intermediate_data.outputs = intermediate_data.outputs + (
            intermediate_data.present_key_value,
        )

    intermediate_data.hidden_states = intermediate_data.outputs[0]
    if intermediate_data.use_cache:
        intermediate_data.next_decoder_cache += (intermediate_data.outputs[-1],)
    if intermediate_data.output_attentions:
        intermediate_data.all_self_attentions = (
            intermediate_data.all_self_attentions + (intermediate_data.outputs[1],)
        )
        if config.add_cross_attention:
            intermediate_data.all_cross_attentions = (
                intermediate_data.all_cross_attentions + (intermediate_data.outputs[2],)
            )


def bert_encoder_layer_forward(
    stage_attention,
    stage_intermediate,
    stage_output,
    intermediate_data,
    config,
) -> Tuple[torch.Tensor]:
    bert_encoder_layer_attention_forward(stage_attention, intermediate_data, 0)

    attention_output = intermediate_data.self_attention_outputs[0]

    if config.chunk_size_feed_forward > 0:
        bert_encoder_layer_intermediate_output_forward(
            stage_intermediate, stage_output, intermediate_data, attention_output, config
        )
    else:
        intermediate_output = bert_encoder_layer_intermediate_forward(
            stage_intermediate, intermediate_data, attention_output, config
        )
        bert_encoder_layer_output_forward(
            stage_output, intermediate_data, intermediate_output, attention_output, config
        )


# def feed_forward_chunk(stage_intermediate, stage_output, attention_output):
#     intermediate_output = stage_intermediate(attention_output)
#     layer_output = stage_output(intermediate_output, attention_output)
#     return layer_output


def bert_layer_forward_i(layer_module, idx, intermediate_data, config):
    if intermediate_data.output_hidden_states:
        intermediate_data.all_hidden_states = intermediate_data.all_hidden_states + (
            intermediate_data.hidden_states,
        )

    layer_head_mask = (
        intermediate_data.head_mask[idx]
        if intermediate_data.head_mask is not None
        else None
    )
    past_key_value = (
        intermediate_data.past_key_values[idx]
        if intermediate_data.past_key_values is not None
        else None
    )

    layer_outputs = layer_module(
        intermediate_data.hidden_states,
        intermediate_data.extended_attention_mask,
        layer_head_mask,
        intermediate_data.encoder_hidden_states,
        intermediate_data.encoder_extended_attention_mask,
        past_key_value,
        intermediate_data.output_attentions,
    )

    intermediate_data.hidden_states = layer_outputs[0]
    if intermediate_data.use_cache:
        intermediate_data.next_decoder_cache += (layer_outputs[-1],)
    if intermediate_data.output_attentions:
        intermediate_data.all_self_attentions = intermediate_data.all_self_attentions + (
            layer_outputs[1],
        )
        if config.add_cross_attention:
            intermediate_data.all_cross_attentions = (
                intermediate_data.all_cross_attentions + (layer_outputs[2],)
            )


def bert_layer_forward_i2(layer_module, idx, bert_layer_data, intermediate_data, config):
    if intermediate_data.output_hidden_states:
        bert_layer_data.all_hidden_states = bert_layer_data.all_hidden_states + (
            bert_layer_data.hidden_states,
        )

    layer_head_mask = (
        intermediate_data.head_mask[idx]
        if intermediate_data.head_mask is not None
        else None
    )
    past_key_value = (
        intermediate_data.past_key_values[idx]
        if intermediate_data.past_key_values is not None
        else None
    )

    layer_outputs = layer_module(
        bert_layer_data.hidden_states,
        intermediate_data.extended_attention_mask,
        layer_head_mask,
        intermediate_data.encoder_hidden_states,
        intermediate_data.encoder_extended_attention_mask,
        past_key_value,
        intermediate_data.output_attentions,
    )

    bert_layer_data.hidden_states = layer_outputs[0]
    if intermediate_data.use_cache:
        bert_layer_data.next_decoder_cache += (layer_outputs[-1],)
    if intermediate_data.output_attentions:
        bert_layer_data.all_self_attentions = bert_layer_data.all_self_attentions + (
            layer_outputs[1],
        )
        if config.add_cross_attention:
            bert_layer_data.all_cross_attentions = (
                bert_layer_data.all_cross_attentions + (layer_outputs[2],)
            )

    return BertLayerData(
        bert_layer_data.all_hidden_states,
        bert_layer_data.all_self_attentions,
        bert_layer_data.all_cross_attentions,
        bert_layer_data.next_decoder_cache,
        bert_layer_data.hidden_states,
    )


def get_encoder_outputs(intermediate_data):
    if intermediate_data.output_hidden_states:
        intermediate_data.all_hidden_states = intermediate_data.all_hidden_states + (
            intermediate_data.hidden_states,
        )

    if not intermediate_data.return_dict:
        return tuple(
            v
            for v in [
                intermediate_data.hidden_states,
                intermediate_data.next_decoder_cache,
                intermediate_data.all_hidden_states,
                intermediate_data.all_self_attentions,
                intermediate_data.all_cross_attentions,
            ]
            if v is not None
        )
    else:
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=intermediate_data.hidden_states,
            past_key_values=intermediate_data.next_decoder_cache,
            hidden_states=intermediate_data.all_hidden_states,
            attentions=intermediate_data.all_self_attentions,
            cross_attentions=intermediate_data.all_cross_attentions,
        )


def bert_pooler_forward(pooler, encoder_outputs, return_dict):
    sequence_output = encoder_outputs[0]
    pooled_output = pooler(sequence_output) if pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )


def bert_only_mlm_head_forward(cls, outputs):
    sequence_output = outputs[0]
    prediction_scores = cls(sequence_output)

    return prediction_scores


def bert_hidden_states_forward(stage, hidder_states):
    return stage(hidder_states)


def bert_predictions_head_transform_forward(transform, outputs):
    sequence_output = outputs[0]
    hidden_states = transform(sequence_output)

    return hidden_states


def bert_decoder_forward(decoder, hidden_states):
    prediction_scores = decoder(hidden_states)

    return prediction_scores


def get_predicted_token(prediction_scores, encoded_input, tokenizer):
    mask_token_index = torch.where(
        encoded_input["input_ids"] == tokenizer.mask_token_id
    )[1]
    assert len(mask_token_index) == 1, "Sentence should only have one mask token."
    mask_token_index = mask_token_index.item()
    mask_token_logits = prediction_scores[0, mask_token_index]
    top_token_id = torch.argmax(mask_token_logits).item()
    predicted_token = tokenizer.decode([top_token_id])

    return predicted_token
