from tools_inference.bert.inference_bert_cotdm import inference_bert_cotdm


def inference_bert_alpaserve(encoded_input, model_idx):
    return inference_bert_cotdm(encoded_input, model_idx)
