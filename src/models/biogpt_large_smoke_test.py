import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "microsoft/BioGPT-Large"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected by PyTorch.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading:", MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    prompt = (
        "Task: Answer with yes, no, or maybe.\n"
        "Research question: Is aspirin commonly used to reduce fever?\n"
        "Evidence: Aspirin is an analgesic and antipyretic medicine used for pain and fever.\n"
        "Decision:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("\nBioGPT output:", response)
    print(f"Model memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    print("\nBioGPT-Large local compatibility check complete.")


if __name__ == "__main__":
    main()