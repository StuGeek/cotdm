from tools_inference.gpt2.inference_gpt2_cotdm import inference_gpt2_cotdm, get_generated_text_cotdm


def inference_gpt2_alpaserve(input, model_idx):
    return inference_gpt2_cotdm(input, model_idx)


def get_generated_text_alpaserve(encoded_input, model_idx):
    return get_generated_text_cotdm(encoded_input, model_idx, inference_gpt2_alpaserve)