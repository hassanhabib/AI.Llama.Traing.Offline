from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load merged model and tokenizer
model = AutoModelForCausalLM.from_pretrained("tinyllama-merged", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("tinyllama-merged")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

prompt = "### Instruction:\nWho is Hassan Habib?\n\n### Input:\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))