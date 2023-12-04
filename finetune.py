# パッケージのインポート
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments, logging, pipeline)
from trl import SFTTrainer

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def create_dataset():
    """create dataset and save json"""

    json_files = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".json") and not f == "cdsl-dataset.json"
    ]

    # for sanskrit
    skt_eng_dict_list = [
        "wil",
        "yat",
        "gst",
        "ben",
        "mw72",
        "lan",
        "lrv",
        "ap90",
        "cae",
        "md",
        "mw",
        "shs",
        # 'ap',
        # 'pd'
    ]

    eng_skt_dict_list = ["mwe", "bor", "ae"]
    skt_fre_dict_list = ["bur", "stc"]
    skt_ger_dict_list = ["pwg", "gra", "pw", "ccs", "sch"]
    skt_lat_dict_list = ["bop"]
    skt_skt_dict_list = ["armh", "vcp", "skd"]
    grk_eng_dict_list = ["lsj"]
    lat_eng_dict_list = ["ls"]

    dictionary_dict = {
        "Sanskrit-English": skt_eng_dict_list,
        "English-Sanskrit": eng_skt_dict_list,
        "Sanskrit-French": skt_fre_dict_list,
        "Sanskrit-German": skt_ger_dict_list,
        "Sanskrit-Latin": skt_lat_dict_list,
        "Sanskrit-Sanskrit": skt_skt_dict_list,
        "Greek-English": grk_eng_dict_list,
        "Latin-English": lat_eng_dict_list,
    }

    dataset_data = []
    for json_file in json_files:
        for key, value in dictionary_dict.items():
            if json_file.split(".")[0] in value:
                dict_type = key

        context_format = (
            f"The {dict_type} dictionary contains the following description:"
        )

        with open(os.path.join(data_dir, json_file), "r", encoding="utf-8") as f:
            data_list = json.load(f)

            for data in data_list:
                dataset_data.append(
                    {
                        "instruction": f"What is the meaning of Sanskrit word '{data['headword']}'?",
                        "input": f"{context_format} {data['description']}",
                        "output": data["description"],
                    }
                )

    with open(os.path.join(data_dir, "cdsl-dataset.json"), "w") as f:
        json.dump(dataset_data, f)


def create_instruction_dataset():
    """create dataset for instruction"""

    # load dataset
    dataset = load_dataset(
        "json", data_files=os.path.join(data_dir, "cdsl-dataset.json"), split="train"
    )
    dataset = dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)["test"]

    # プロンプトテンプレートの準備
    def generate_prompt(data_point):
        if data_point["input"]:
            result = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}
<|endoftext|>
"""
        else:
            result = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}
<|endoftext|>
"""
        return result

    # テキスト列の追加
    def add_text(example):
        example["text"] = generate_prompt(example)

        return example

    dataset = dataset.map(add_text)

    return dataset


def train(dataset, bnb_config, model_name):
    """finetuning by Sanskrit-English dictionaries"""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # モデル名
        quantization_config=bnb_config,  # 量子化パラメータ
        device_map="auto",
    )
    model.config.use_cache = False  # キャッシュ (学習時はFalse)
    model.config.pretraining_tp = 2  # 事前学習で使用したテンソル並列ランク(7B:1、13B:2)

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,  # モデル名
        use_fast=False,  # Fastトークナイザーの有効化
        add_eos_token=True,  # データへのEOSの追加を指示
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # fp16でのオーバーフロー問題対策

    # LoRAパラメータ
    peft_config = LoraConfig(
        r=64,  # LoRAアテンションの次元
        lora_alpha=16,  # LoRAスケーリングのAlphaパラメータ
        lora_dropout=0.1,  # LoRA レイヤーのドロップアウト確率
        bias="none",  # LoRAのバイアス種別 ("none","all", "lora_only")
        task_type="CAUSAL_LM",  # タスク種別
        target_modules=[
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "k_proj",
            "v_proj",
        ],
    )

    # 学習パラメータ
    training_arguments = TrainingArguments(
        output_dir="./results",  # 出力ディレクトリ
        num_train_epochs=1,  # エポック数
        per_device_train_batch_size=4,  # 学習用のGPUあたりのバッチサイズ
        gradient_accumulation_steps=1,  # 勾配を蓄積するための更新ステップの数
        optim="paged_adamw_32bit",  # オプティマイザ
        save_steps=0,  # 何ステップ毎にチェックポイントを保存するか
        logging_steps=25,  # 何ステップ毎にログを記録するか
        learning_rate=2e-4,  # 初期学習率 (AdamW オプティマイザー)
        weight_decay=0.001,  # bias/LayerNormウェイトを除く全レイヤーに適用するウェイト減衰
        fp16=True,  # fp16学習の有効化 (T4:True,A100:False)
        bf16=False,  # bf16学習の有効化 (T4:False,A100:True)
        max_grad_norm=0.3,  # 最大法線勾配 (勾配クリッピング)
        max_steps=-1,  # 学習ステップ数 (num_train_epochsをオーバーライド)
        warmup_ratio=0.03,  # 線形ウォームアップのステップ比率 (0から学習率まで)
        group_by_length=True,  # シーケンスを同じ長さのバッチにグループ化 (メモリ節約して学習速度が大幅アップ)
        lr_scheduler_type="cosine",  # 学習率スケジュール
        report_to="wandb",  # レポート
    )

    # SFTパラメータ
    trainer = SFTTrainer(
        model=model,  # モデル
        tokenizer=tokenizer,  # トークナイザー
        train_dataset=dataset,  # データセット
        dataset_text_field="text",  # データセットのtext列
        peft_config=peft_config,  # PEFTパラメータ
        args=training_arguments,  # 学習パラメータ
        max_seq_length=None,  # 使用する最大シーケンス長
        packing=False,  # 同じ入力シーケンスに複数サンプルをパッキング(効率を高める)
    )

    # モデルの学習
    trainer.train()
    trainer.model.save_pretrained("./results")


def main():
    create_dataset()

    dataset = create_instruction_dataset()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bitベースモデルの有効化
        bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
        bnb_4bit_compute_dtype=torch.float16,  # 4bitベースモデルのdtype (float16 or bfloat16)
        bnb_4bit_use_double_quant=False,  # 4bitベースモデルのネストされた量子化の有効化 (二重量子化)
    )

    model_name = "meta-llama/Llama-2-13b-chat-hf"

    train(dataset, bnb_config, model_name)


if __name__ == "__main__":
    main()
