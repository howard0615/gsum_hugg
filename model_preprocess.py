from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("fnlp/bart-large-chinese")
model_static_dict = model.state_dict()

z_encoder = {}
for k, v in model_static_dict.items():
    if k.startswith("model.encoder."):
        z_encoder[k.replace("encoder", "z_encoder")]=v

z_encoder_attn = {}
for k, v in model_static_dict.items():
    if k.startswith("model.decoder.layers.") and "encoder_attn" in k:
        print(k)
        z_encoder_attn[k.replace("encoder_attn", "z_encoder_attn")]=v

import torch

torch.save(model_static_dict, "./chinese_gsum_bart/pytorch_model.bin")