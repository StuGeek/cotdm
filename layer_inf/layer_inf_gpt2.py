import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import warnings

GPT2_TEMPERATURE = 1.0
GPT2_TOP_K = 50

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
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(torch.float32).min

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
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
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
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
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
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
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

class GPT2PreProcessOutput:
    def __init__(self, input_ids, past_key_values, attention_mask, token_type_ids, 
                 position_ids, head_mask, inputs_embeds, encoder_hidden_states, 
                 encoder_attention_mask, use_cache, output_attentions, 
                 output_hidden_states, return_dict, input_shape):
        self.input_ids = input_ids
        self.past_key_values = past_key_values
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.head_mask = head_mask
        self.inputs_embeds = inputs_embeds
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_attention_mask = encoder_attention_mask
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.input_shape = input_shape

    def to(self, device, non_blocking=False):
        if self.input_ids != None:
            self.input_ids = self.input_ids.to(device, non_blocking=non_blocking)
        # if self.past_key_values != None:
        #     self.past_key_values = self.past_key_values.to(device)
        if self.attention_mask != None:
            self.attention_mask = self.attention_mask.to(device, non_blocking=non_blocking)
        if self.token_type_ids != None:
            self.token_type_ids = self.token_type_ids.to(device, non_blocking=non_blocking)
        if self.position_ids != None:
            self.position_ids = self.position_ids.to(device, non_blocking=non_blocking)
        # if self.head_mask != None:
        #     self.head_mask = self.head_mask.to(device)
        if self.inputs_embeds != None:
            self.inputs_embeds = self.inputs_embeds.to(device, non_blocking=non_blocking)
        if self.encoder_hidden_states != None:
            self.encoder_hidden_states = self.encoder_hidden_states.to(device, non_blocking=non_blocking)
        if self.encoder_attention_mask != None:
            self.encoder_attention_mask = self.encoder_attention_mask.to(device, non_blocking=non_blocking)

        return self

class GPT2BlockData:
    def __init__(self, output_shape, presents, all_hidden_states, all_self_attentions, all_cross_attentions):
        self.output_shape = output_shape
        self.presents = presents
        self.all_hidden_states = all_hidden_states
        self.all_self_attentions =  all_self_attentions
        self.all_cross_attentions = all_cross_attentions

    def to(self, device, non_blocking=False):
        # if self.output_shape != None:
        #     self.output_shape = self.output_shape.to(device)
        # if self.presents != None:
        #     self.presents = self.presents.to(device)
        if self.all_hidden_states != None:
            self.all_hidden_states = self.all_hidden_states.to(device, non_blocking=non_blocking)
        if self.all_self_attentions != None:
            self.all_self_attentions =  self.all_self_attentions.to(device, non_blocking=non_blocking)
        if self.all_cross_attentions != None:
            self.all_cross_attentions = self.all_cross_attentions.to(device, non_blocking=non_blocking)

        return self

def gpt2_preprocess(encoded_input, config, total_h_layer_num):
    input_ids = encoded_input.get("input_ids")
    past_key_values = encoded_input.get("past_key_values")
    attention_mask = encoded_input.get("attention_mask")
    token_type_ids = encoded_input.get("token_type_ids")
    position_ids = encoded_input.get("position_ids")
    head_mask = encoded_input.get("head_mask")
    inputs_embeds = encoded_input.get("inputs_embeds")
    encoder_hidden_states = encoded_input.get("encoder_hidden_states")
    encoder_attention_mask = encoded_input.get("encoder_attention_mask")
    use_cache = encoded_input.get("use_cache")
    output_attentions = encoded_input.get("output_attentions")
    output_hidden_states = encoded_input.get("output_hidden_states")
    return_dict = encoded_input.get("return_dict")

    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else config.use_cache
    return_dict = return_dict if return_dict is not None else config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * total_h_layer_num)
    else:
        past_length = past_key_values[0][0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # GPT2Attention mask.
    if attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_attention_mask = invert_attention_mask(encoder_attention_mask)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = get_head_mask(head_mask, config.n_layer)

    return GPT2PreProcessOutput(input_ids, past_key_values, attention_mask, token_type_ids, 
                position_ids, head_mask, inputs_embeds, encoder_hidden_states, 
                encoder_attention_mask, use_cache, output_attentions, 
                output_hidden_states, return_dict, input_shape)

def wte_forward(wte, preprocess_output):
    hidden_states = None

    if preprocess_output.inputs_embeds is None:
        preprocess_output.inputs_embeds = wte(preprocess_output.input_ids)

    if preprocess_output.token_type_ids is not None:
        token_type_embeds = wte(preprocess_output.token_type_ids)
        hidden_states = token_type_embeds

    return hidden_states

def wpe_forward(wpe, preprocess_output, hidden_states):
    position_embeds = wpe(preprocess_output.position_ids)
    if hidden_states != None:
        hidden_states = hidden_states + preprocess_output.inputs_embeds + position_embeds
    else:
        hidden_states = preprocess_output.inputs_embeds + position_embeds
    return hidden_states

def drop_forward(drop, hidden_states):
    return drop(hidden_states)

def block_preprocess(preprocess_output, hidden_states, config):
    output_shape = preprocess_output.input_shape + (hidden_states.size(-1),)

    # if self.gradient_checkpointing and self.training:
    #     if use_cache:
    #         logger.warning_once(
    #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #         )
    #         use_cache = False

    presents = () if preprocess_output.use_cache else None
    all_self_attentions = () if preprocess_output.output_attentions else None
    all_cross_attentions = () if preprocess_output.output_attentions and config.add_cross_attention else None
    all_hidden_states = () if preprocess_output.output_hidden_states else None

    return GPT2BlockData(output_shape, presents, all_hidden_states, all_self_attentions, all_cross_attentions)

def gpt2_block_forward_i(block, idx, preprocess_output, block_preprocess_output, hidden_states, config):
    # Model parallel
        # if self.model_parallel:
        #     torch.cuda.set_device(hidden_states.device)
        #     # Ensle(ure layer_past is on same device as hidden_states (might not be correct)
        #     if layer_past is not None:
        #         layer_past = tuppast_state.to(hidden_states.device) for past_state in layer_past)
        #     # Ensure that attention_mask is always on the same device as hidden_states
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(hidden_states.device)
        #     if isinstance(head_mask, torch.Tensor):
        #         head_mask = head_mask.to(hidden_states.device)
        if preprocess_output.output_hidden_states:
            block_preprocess_output.all_hidden_states = block_preprocess_output.all_hidden_states + (hidden_states,)

        # if self.gradient_checkpointing and self.training:
        #
        #     def create_custom_forward(module):
        #         def custom_forward(*inputs):
        #             # None for past_key_value
        #             return module(*inputs, use_cache, output_attentions)
        #
        #         return custom_forward
        #
        #     outputs = torch.utils.checkpoint.checkpoint(
        #         create_custom_forward(block),
        #         hidden_states,
        #         None,
        #         attention_mask,
        #         head_mask[i],
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #     )
        # else:
        
        outputs = block(
            hidden_states,
            layer_past=preprocess_output.past_key_values[idx],
            attention_mask=preprocess_output.attention_mask,
            head_mask=preprocess_output.head_mask[idx],
            encoder_hidden_states=preprocess_output.encoder_hidden_states,
            encoder_attention_mask=preprocess_output.encoder_attention_mask,
            use_cache=preprocess_output.use_cache,
            output_attentions=preprocess_output.output_attentions,
        )

        hidden_states = outputs[0]
        if preprocess_output.use_cache is True:
            block_preprocess_output.presents = block_preprocess_output.presents + (outputs[1],)

        if preprocess_output.output_attentions:
            block_preprocess_output.all_self_attentions = block_preprocess_output.all_self_attentions + (outputs[2 if preprocess_output.use_cache else 1],)
            if config.add_cross_attention:
                block_preprocess_output.all_cross_attentions = block_preprocess_output.all_cross_attentions + (outputs[3 if preprocess_output.use_cache else 2],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        # if self.model_parallel:
        #     for k, v in self.device_map.items():
        #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
        #             hidden_states = hidden_states.to("cuda:" + str(k + 1))

        return preprocess_output, block_preprocess_output, hidden_states

def ln_f_forward(ln_f, preprocess_output, block_preprocess_output, hidden_states):
    hidden_states = ln_f(hidden_states)

    hidden_states = hidden_states.view(block_preprocess_output.output_shape)
    # Add last hidden state
    if preprocess_output.output_hidden_states:
        block_preprocess_output.all_hidden_states = block_preprocess_output.all_hidden_states + (hidden_states,)

    if not preprocess_output.return_dict:
        return tuple(
            v
            for v in [hidden_states, block_preprocess_output.presents, 
                      block_preprocess_output.all_hidden_states, 
                      block_preprocess_output.all_self_attentions, 
                      block_preprocess_output.all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=block_preprocess_output.presents,
        hidden_states=block_preprocess_output.all_hidden_states,
        attentions=block_preprocess_output.all_self_attentions,
        cross_attentions=block_preprocess_output.all_cross_attentions,
    )


def lm_head_forward(lm_head, transformer_outputs):
    hidden_states = transformer_outputs[0]
    lm_logits = lm_head(hidden_states)

    return lm_logits


def get_next_token(lm_logits):
    temperature = GPT2_TEMPERATURE
    top_k = GPT2_TOP_K
    
    next_token_logits = lm_logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
    top_k_values, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
    top_k_indices = top_k_indices.squeeze().tolist()
    next_token_index = torch.multinomial(top_k_values, num_samples=1).item()
    next_token = top_k_indices[next_token_index]

    return next_token
