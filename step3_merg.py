from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "tinyllama-finetuned")

# Merge and unload
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("tinyllama-merged")
AutoTokenizer.from_pretrained("tinyllama-finetuned").save_pretrained("tinyllama-merged")
print("✅ Merged model saved to ./tinyllama-merged")
