import argparse
import glob
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from model_utils import (
    BASE_MODEL_NAME,
    PROJECT_DIR,
    create_bnb_config,
    create_tokenizer,
)


DEFAULT_ADAPTER_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "model"


def load_base_model(model_name: str = BASE_MODEL_NAME):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=create_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.pretraining_tp = 2
    return model


def discover_adapter_models(model_dir: Path = MODEL_DIR) -> dict[int, Path]:
    return {
        n + 1: Path(model)
        for n, model in enumerate(glob.glob(str(model_dir / "llama-2-*")))
    }


def select_adapter_path(model_dict: dict[int, Path]) -> Path:
    print("モデル選択")
    for key, value in model_dict.items():
        print(f"[{key}] {value.name}")

    model_num = input("Input number: ")

    if not model_num:
        print("Loading recently created model.")
        return DEFAULT_ADAPTER_DIR

    selected_model = model_dict[int(model_num)]
    print(f"Loading {selected_model} model.")
    return selected_model


def run_adapter_chat(model, tokenizer, response_count: int) -> None:
    try:
        while True:
            base_prompt = input("Prompt: ")
            prompt = f"### Instruction:\n{base_prompt}\n\n### Response:\n"

            for i in range(response_count):
                inputs = tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )
                outputs = model.generate(
                    **inputs.to(model.device),
                    max_new_tokens=256,
                    temperature=0.7,
                )
                output = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                )
                print(f"Answer ({i}): ", output)

    except KeyboardInterrupt:
        print("Quit.")


def chat(response_count: int) -> None:
    base_model = load_base_model()
    adapter_path = select_adapter_path(discover_adapter_models())
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = create_tokenizer()
    run_adapter_chat(model, tokenizer, response_count)


def chat_base_model() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=create_bnb_config(),
        device_map="auto",
        use_auth_token=True,
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 2

    tokenizer = create_tokenizer(use_auth_token=True)

    try:
        while True:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with llama.")
    parser.add_argument("-l", "--llama", action="store_true", help="Use base model")
    parser.add_argument(
        "-r", "--response", type=int, default=1, help="Number of responses (default 1)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llama:
        chat_base_model()
    else:
        chat(args.response)


if __name__ == "__main__":
    main()
