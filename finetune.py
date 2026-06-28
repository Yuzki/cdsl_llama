import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

from model_utils import (
    BASE_MODEL_NAME,
    PROJECT_DIR,
    create_bnb_config,
    create_tokenizer,
)


DATA_DIR = PROJECT_DIR / "data"
DATASET_FILE = DATA_DIR / "cdsl-dataset.json"
OUTPUT_DIR = PROJECT_DIR / "results"

DICTIONARY_TYPES = {
    "Sanskrit-English": (
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
    ),
    "English-Sanskrit": ("mwe", "bor", "ae"),
    "Sanskrit-French": ("bur", "stc"),
    "Sanskrit-German": ("pwg", "gra", "pw", "ccs", "sch"),
    "Sanskrit-Latin": ("bop",),
    "Sanskrit-Sanskrit": ("armh", "vcp", "skd"),
    "Greek-English": ("lsj",),
    "Latin-English": ("ls",),
}


def dictionary_type_for(json_file: Path) -> str:
    dictionary_name = json_file.stem

    for dictionary_type, dictionary_names in DICTIONARY_TYPES.items():
        if dictionary_name in dictionary_names:
            return dictionary_type

    raise ValueError(f"Unknown dictionary source: {json_file.name}")


def create_dataset(data_dir: Path = DATA_DIR, output_file: Path = DATASET_FILE) -> None:
    """Create instruction records from source JSON dictionaries."""
    json_files = sorted(
        path for path in data_dir.glob("*.json") if path.name != output_file.name
    )
    dataset_data = []

    for json_file in json_files:
        dict_type = dictionary_type_for(json_file)
        context_format = (
            f"The {dict_type} dictionary contains the following description:"
        )

        with json_file.open(encoding="utf-8") as f:
            data_list = json.load(f)

        for data in data_list:
            dataset_data.append(
                {
                    "instruction": (
                        f"What is the meaning of Sanskrit word '{data['headword']}'?"
                    ),
                    "input": f"{context_format} {data['description']}",
                    "output": data["description"],
                }
            )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(dataset_data, f, ensure_ascii=False)


def generate_prompt(data_point) -> str:
    if data_point["input"]:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}
<|endoftext|>
"""

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}
<|endoftext|>
"""


def add_text(example):
    example["text"] = generate_prompt(example)
    return example


def create_instruction_dataset(dataset_file: Path = DATASET_FILE):
    """Load generated records and add the training prompt column."""
    dataset = load_dataset(
        "json",
        data_files=str(dataset_file),
        split="train",
    )
    dataset = dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)["test"]
    return dataset.map(add_text)


def train(dataset, model_name: str = BASE_MODEL_NAME) -> None:
    """Fine-tune the base model with dictionary instruction records."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=create_bnb_config(),
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 2

    tokenizer = create_tokenizer(model_name)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
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

    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        args=training_arguments,
        max_seq_length=None,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(str(OUTPUT_DIR))


def main() -> None:
    create_dataset()
    dataset = create_instruction_dataset()
    train(dataset)


if __name__ == "__main__":
    main()
