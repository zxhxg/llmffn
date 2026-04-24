import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "/home/wlh/llmffn/models/Meta-Llama-3.1-8B"
_MODEL = None
_TOKENIZER = None


def load_fp16_model(model_id=MODEL_ID):
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    return _MODEL, _TOKENIZER


def _get_runtime_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    raise RuntimeError("No non-meta parameter device was found.")


def run_fp16(
    prompt,
    model_id=MODEL_ID,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
):
    model, tokenizer = load_fp16_model(model_id=model_id)
    inputs = tokenizer(prompt, return_tensors="pt")
    runtime_device = _get_runtime_device(model)
    inputs = {key: value.to(runtime_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    text = run_fp16(
        "Hey how are you doing today?",
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    print(text)
