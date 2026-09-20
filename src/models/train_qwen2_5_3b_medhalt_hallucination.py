import gc
import json
import math
import random
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MAX_LENGTH = 768
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 1
EVALUATION_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
RANDOM_SEED = 42

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
RESULT_DIR = PROJECT_DIR / "results"

TRAIN_PATH = DATA_DIR / "controlled_medhalt_hallucination_train.jsonl"
VALIDATION_PATH = (
    DATA_DIR / "controlled_medhalt_hallucination_validation.jsonl"
)

CHECKPOINT_DIR = (
    ARTIFACT_DIR
    / "checkpoints"
    / "qwen2_5_3b_medhalt_hallucination"
)

BEST_ADAPTER_DIR = (
    ARTIFACT_DIR
    / "adapters"
    / "qwen2_5_3b_medhalt_hallucination_best_validation"
)

FINAL_ADAPTER_DIR = (
    ARTIFACT_DIR
    / "adapters"
    / "qwen2_5_3b_medhalt_hallucination_final"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "qwen2_5_3b_medhalt_hallucination_training_summary.json"
)


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_quantization_config(compute_dtype):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


class LabelCompletionDataset(Dataset):
    """
    The prompt is visible to the model, but loss is calculated only on the
    target label: supported or hallucinated.
    """

    def __init__(self, records, tokenizer):
        self.examples = []

        for record in records:
            label_ids = tokenizer(
                f" {record['label']}",
                add_special_tokens=False,
            )["input_ids"]

            # One token is reserved for EOS after the target label.
            prompt_max_length = MAX_LENGTH - len(label_ids) - 1

            prompt_ids = tokenizer(
                record["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=prompt_max_length,
            )["input_ids"]

            input_ids = (
                prompt_ids
                + label_ids
                + [tokenizer.eos_token_id]
            )

            labels = (
                [-100] * len(prompt_ids)
                + label_ids
                + [tokenizer.eos_token_id]
            )

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CompletionDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_batch_length = max(
            len(example["input_ids"])
            for example in batch
        )

        input_ids = []
        attention_masks = []
        labels = []

        for example in batch:
            sequence_length = len(example["input_ids"])
            padding_length = max_batch_length - sequence_length

            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            attention_masks.append(
                [1] * sequence_length
                + [0] * padding_length
            )

            labels.append(
                example["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


@torch.inference_mode()
def evaluate_loss(model, validation_loader):
    model.eval()

    total_loss = 0.0

    for batch in tqdm(
        validation_loader,
        desc="Calculating validation loss",
        leave=False,
    ):
        batch = {
            name: tensor.to("cuda")
            for name, tensor in batch.items()
        }

        outputs = model(**batch)
        total_loss += outputs.loss.item()

    model.train()

    return total_loss / len(validation_loader)


def get_trainable_parameter_count(model):
    trainable_parameters = 0
    all_parameters = 0

    for parameter in model.parameters():
        all_parameters += parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    trainable_percent = (
        100 * trainable_parameters / all_parameters
    )

    return trainable_parameters, all_parameters, trainable_percent


def save_adapter(model, tokenizer, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    set_random_seed(RANDOM_SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(TRAIN_PATH)
    validation_records = load_jsonl(VALIDATION_PATH)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training rows:", len(train_records))
    print("Validation rows:", len(validation_records))
    print("\nLoading Qwen2.5-3B from the base model in 4-bit mode...")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_quantization_config(compute_dtype),
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    base_model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    trainable_parameters, all_parameters, trainable_percent = (
        get_trainable_parameter_count(model)
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
        "|| All parameters:",
        f"{all_parameters:,}",
        "|| Trainable percent:",
        f"{trainable_percent:.4f}",
    )

    train_dataset = LabelCompletionDataset(
        train_records,
        tokenizer,
    )

    validation_dataset = LabelCompletionDataset(
        validation_records,
        tokenizer,
    )

    collator = CompletionDataCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )

    total_update_steps = updates_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    print("\nStarting separate Med-HALT Qwen2.5-3B LoRA fine-tuning...")

    best_validation_loss = float("inf")
    best_epoch = None
    epoch_summaries = []

    for epoch_number in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        total_training_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_number}/{NUM_EPOCHS}",
        )

        for step_number, batch in enumerate(
            progress_bar,
            start=1,
        ):
            batch = {
                name: tensor.to("cuda")
                for name, tensor in batch.items()
            }

            outputs = model(**batch)
            raw_loss = outputs.loss

            scaled_loss = (
                raw_loss / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            total_training_loss += raw_loss.item()

            is_update_step = (
                step_number % GRADIENT_ACCUMULATION_STEPS == 0
                or step_number == len(train_loader)
            )

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix(
                loss=f"{raw_loss.item():.4f}"
            )

        average_training_loss = (
            total_training_loss / len(train_loader)
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
        )

        epoch_checkpoint_directory = (
            CHECKPOINT_DIR / f"epoch_{epoch_number}"
        )

        save_adapter(
            model,
            tokenizer,
            epoch_checkpoint_directory,
        )

        print(
            f"\nEpoch {epoch_number} complete | "
            f"train loss: {average_training_loss:.4f} | "
            f"validation loss: {validation_loss:.4f}"
        )

        print("Checkpoint saved to:", epoch_checkpoint_directory)

        epoch_summaries.append(
            {
                "epoch": epoch_number,
                "train_loss": round(average_training_loss, 6),
                "validation_loss": round(validation_loss, 6),
                "checkpoint": str(
                    epoch_checkpoint_directory.relative_to(PROJECT_DIR)
                ),
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch_number

            save_adapter(
                model,
                tokenizer,
                BEST_ADAPTER_DIR,
            )

            print(
                "New best validation adapter saved to:",
                BEST_ADAPTER_DIR,
            )

    save_adapter(
        model,
        tokenizer,
        FINAL_ADAPTER_DIR,
    )

    training_summary = {
        "model": MODEL_ID,
        "task": (
            "Controlled Med-HALT binary hallucination detection: "
            "supported versus hallucinated answer."
        ),
        "training_data": str(
            TRAIN_PATH.relative_to(PROJECT_DIR)
        ),
        "validation_data": str(
            VALIDATION_PATH.relative_to(PROJECT_DIR)
        ),
        "frozen_test_policy": (
            "The controlled Med-HALT test split was not used for "
            "training, checkpoint selection, or calibration."
        ),
        "training_configuration": {
            "max_length": MAX_LENGTH,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "quantization": "4-bit NF4",
            "compute_dtype": str(compute_dtype),
        },
        "trainable_parameters": trainable_parameters,
        "all_parameters": all_parameters,
        "trainable_parameter_percent": round(
            trainable_percent,
            6,
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": round(
            best_validation_loss,
            6,
        ),
        "best_adapter": str(
            BEST_ADAPTER_DIR.relative_to(PROJECT_DIR)
        ),
        "final_adapter": str(
            FINAL_ADAPTER_DIR.relative_to(PROJECT_DIR)
        ),
        "epochs": epoch_summaries,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2)

    print("\nQwen2.5-3B Med-HALT LoRA fine-tuning complete.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", f"{best_validation_loss:.4f}")
    print("Best adapter saved to:", BEST_ADAPTER_DIR)
    print("Final adapter saved to:", FINAL_ADAPTER_DIR)
    print("Training summary saved to:", SUMMARY_PATH)

    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()