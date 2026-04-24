import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel


model_id = "/home/wlh/llmffn/models/Meta-Llama-3.1-8B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

_original_to = PreTrainedModel.to


def _patched_to(self, *args, **kwargs):
    if getattr(self, "quantization_method", None) is not None:
        return self
    return _original_to(self, *args, **kwargs)


PreTrainedModel.to = _patched_to

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={"": 0},
    )
finally:
    PreTrainedModel.to = _original_to

model.eval()

for module in model.modules():
    if hasattr(module, "inv_freq") and torch.is_tensor(module.inv_freq):
        module.inv_freq = module.inv_freq.to("cuda:0")

prompt = "Hey how are you doing today?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to("cuda:0") for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
