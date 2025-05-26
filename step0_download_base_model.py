from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("tinyllama-base")
tokenizer.save_pretrained("tinyllama-base")
print("✅ Base model downloaded to ./tinyllama-base")
