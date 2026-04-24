from pathlib import Path

from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
model_id = str(REPO_ROOT / "models" / "Meta-Llama-3.1-8B")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map={"": "cpu"},
)

# print(model)
# print(model.model)
print(model.model.layers[0])

print(model.model.layers[0].mlp)
print(model.model.layers[0].mlp.gate_proj)
