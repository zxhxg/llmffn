from transformers import AutoModelForCausalLM


model_id = "/home/wlh/llmffn/models/Meta-Llama-3.1-8B"

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