from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/users/zyang157/data/shared/llama/models_hf/7B"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

prompt = "Hello, I am conscious and"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

generated_ids = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])