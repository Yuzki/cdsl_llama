import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse


def chat():
    model_name = "meta-llama/Llama-2-13b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bitベースモデルの有効化
        bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
        bnb_4bit_compute_dtype=torch.float16,  # 4bitベースモデルのdtype (float16 or bfloat16)
        bnb_4bit_use_double_quant=False,  # 4bitベースモデルのネストされた量子化の有効化 (二重量子化)
    )

    # モデルの準備
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model_dict = {1: "./llama-2-13b-skt-eng", 2: "./llama-2-13b-skt-eng-context", 3: "./llama-2-13b-skt-ger-context", 4: "./llama-2-13b-skt-all-context"}
    model_num = input(
        "モデル選択\n[1] Skt-Eng (no context) [2] Skt-Eng (+context) [3] Skt-Eng&Ger (+context) [4] Skt-Eng&Ger&Fre&Lat&Skt (+context) (default is the most recently created model)\n"
    )
    if model_num:
        print(f"Loading {model_dict[int(model_num)]} model.")
        peftmodel_name = model_dict[int(model_num)]
    else:
        print(f"Loading recently created model.")
        peftmodel_name = "./results"
    model = PeftModel.from_pretrained(base_model, peftmodel_name)

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, add_eos_token=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # 推論の実行
    try:
        while True:
            # word = input("Input a Sanskrit word: ")
            # prompt = f"[INST]What is the meaning of {word}?[/INST]"

            base_prompt = input("Prompt: ")
            prompt = f"### Instruction:\n{base_prompt}\n\n### Response:\n"

            for i in range(args.response):
                inputs = tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )
                outputs = model.generate(
                    **inputs.to(model.device),
                    max_new_tokens=256,
                    # do_sample=True,
                    temperature=0.7,
                    # return_dict_in_generate=True,
                )
                output = tokenizer.decode(
                    # outputs.sequences[0, inputs.input_ids.shape[1] :]
                    outputs[0], skip_special_tokens=True
                )
                print(f"Answer ({i}): ", output)

    except KeyboardInterrupt:
        print("Quit.")


def chat_base_model():
    model_name = "meta-llama/Llama-2-13b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bitベースモデルの有効化
        bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
        bnb_4bit_compute_dtype=torch.float16,  # 4bitベースモデルのdtype (float16 or bfloat16)
        bnb_4bit_use_double_quant=False,  # 4bitベースモデルのネストされた量子化の有効化 (二重量子化)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # モデル名
        quantization_config=bnb_config,  # 量子化パラメータ
        device_map="auto",
        use_auth_token=True,
    )
    model.config.use_cache = True  # キャッシュ (学習時はFalse)
    model.config.pretraining_tp = 2  # 事前学習で使用したテンソル並列ランク(7B:1、13B:2)

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,  # モデル名
        use_fast=False,  # Fastトークナイザーの有効化
        add_eos_token=True,  # データへのEOSの追加を指示
        trust_remote_code=True,
        use_auth_token=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # fp16でのオーバーフロー問題対策

    # 推論の実行
    try:
        while True:
            # word = input("Input a Sanskrit word: ")
            # prompt = f"[INST]What is the meaning of {word}?[/INST]"

            base_prompt = input("Prompt: ")
            prompt = f"#Instruction:\n{base_prompt}\n\n# Response:\n"

            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            outputs = model.generate(
                **inputs.to(model.device),
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
            )
            output = tokenizer.decode(outputs.sequences[0, inputs.input_ids.shape[1] :])
            print("Answer: ", output)

    except KeyboardInterrupt:
        print("\nQuit.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chat with llama.')
    parser.add_argument('-l', '--llama', action='store_true', help='Use base model')
    parser.add_argument('-r', '--response', type=int, default=1,  help='Number of responses (default 1)')
    args = parser.parse_args()
    if args.llama:
        chat_base_model()
    else:
        chat()
