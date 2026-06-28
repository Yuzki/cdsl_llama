from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig


BASE_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
PROJECT_DIR = Path(__file__).resolve().parent


def create_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


def create_tokenizer(model_name: str = BASE_MODEL_NAME, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        add_eos_token=True,
        trust_remote_code=True,
        **kwargs,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer
