import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

SEED = 42
MODEL_ID = "microsoft/BioGPT-Large"

EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_LENGTH = 768

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints" / "biogpt_large_pubmedqa_lora"
FINAL_ADAPTER_DIR = ARTIFACT_DIR / "adapters" / "biogpt_large_pubmedqa_lora"
RESULT_DIR = PROJECT_DIR / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


class TokenizedDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.examples = []

        for record in records:
            response_ids = tokenizer(
                record["response"] + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]

            prompt_ids = tokenizer(
                record["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LENGTH - len(response_ids),
            )["input_ids"]

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def create_collator(tokenizer):
    def collate_batch(batch):
        max_length = max(len(item["input_ids"]) for item in batch)
        batch_size = len(batch)

        input_ids = torch.full(
            (batch_size, max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long,
        )

        for index, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"])
            attention_mask[index, :length] = torch.tensor(item["attention_mask"])
            labels[index, :length] = torch.tensor(item["labels"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_batch


@torch.inference_mode()
def evaluate(model, validation_loader):
    model.eval()
    losses = []

    for batch in validation_loader:
        batch = {key: value.to("cuda") for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(outputs.loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(DATA_DIR / "biogpt_pubmedqa_train.jsonl")
    validation_records = load_jsonl(
        DATA_DIR / "biogpt_pubmedqa_validation.jsonl"
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training rows:", len(train_records))
    print("Validation rows:", len(validation_records))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    train_dataset = TokenizedDataset(train_records, tokenizer)
    validation_dataset = TokenizedDataset(validation_records, tokenizer)
    collator = create_collator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading BioGPT-Large in 4-bit mode...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )
    total_training_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_training_steps * 0.05))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    training_history = []
    global_step = 0

    print("\nStarting BioGPT-Large LoRA fine-tuning...")

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for step, batch in enumerate(progress_bar, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            running_loss += loss.item()

            should_update = (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        average_train_loss = running_loss / len(train_loader)
        validation_loss = evaluate(model, validation_loader)

        epoch_summary = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(average_train_loss, 4),
            "validation_loss": round(validation_loss, 4),
        }

        training_history.append(epoch_summary)

        checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        print(
            f"\nEpoch {epoch} complete | "
            f"train loss: {average_train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f}"
        )
        print("Checkpoint saved to:", checkpoint_path)

    model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    summary_path = RESULT_DIR / "biogpt_large_pubmedqa_training_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": MODEL_ID,
                "epochs": EPOCHS,
                "train_rows": len(train_dataset),
                "validation_rows": len(validation_dataset),
                "history": training_history,
                "final_adapter_path": str(FINAL_ADAPTER_DIR),
            },
            file,
            indent=2,
        )

    print("\nBioGPT-Large LoRA fine-tuning complete.")
    print("Final adapter saved to:", FINAL_ADAPTER_DIR)
    print("Training summary saved to:", summary_path)


if __name__ == "__main__":
    main()