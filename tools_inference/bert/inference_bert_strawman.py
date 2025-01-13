from tools_inference.bert.inference_bert_cotdm import inference_bert_cotdm


def inference_bert_strawman(encoded_input, model_idx):
    return inference_bert_cotdm(encoded_input, model_idx)
