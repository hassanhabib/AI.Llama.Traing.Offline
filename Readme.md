# 🦙 Fine-Tune TinyLlama Locally (LoRA + Offline Inference)

This project shows how to fine-tune [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) locally on your own machine using LoRA, with 100% offline capability — no cloud, no GPUs required (but supported), no hosted models.

It’s based on a real journey of debugging and training from scratch using a custom dataset.

---

## ✅ What You’ll Do

- Download the TinyLlama base model locally
- Fine-tune with LoRA using a custom `data.jsonl`
- Merge LoRA weights into the base model
- (Optionally) Convert to `.gguf` and run offline with `llama.cpp`

---

## 🧩 0.Requirements

Install dependencies:

```bash
 pip install transformers datasets peft accelerate bitsandbytes
```

> 🚫 Do **NOT** install `bitsandbytes` if you're on Windows or using an AMD GPU

---

## 📁 Folder Structure

```
standard-llama-finetune/
├── data.jsonl                      ← training dataset (editable)
├── step0_download_base_model.py    ← download base model from Hugging Face
├── step1_fine_tuning.py            ← download base model from Hugging Face
├── step2_merg.py                   ← fine-tune TinyLlama with LoRA
├── step3_test.py                   ← merge LoRA adapter into base model
```

---

## 🔽 0. Download the Base Model

```bash
python step0_download_base_model.py
```

This will save the model to `./tinyllama-base/`

---

## 🔽 1.0. Convert PDF to Raw Text

```bash
python step1_0_pdf_to_text.py
```

This will save the .txt file at the root.

---

## 🔽 1.1. Generate JSONL Files

```bash
python step1_1_generate_jsonl.py
```

This will save the .jsonl file at the root.

---

## 🧠 2. Fine-Tune with LoRA

```bash
python step1_fine_tuning.py
```

- Trains on `data.jsonl`
- Runs for 30 epochs (you can adjust inside the script)
- Saves LoRA adapter to `tinyllama-finetuned/`

---

## 🔗 3. Merge LoRA into Base Model

```bash
python step2_merg.py
```

- Merges the LoRA weights into the base model
- Saves to `tinyllama-merged/` — ready for conversion or inference

---

## 🧪 4. Run Sanity Check (Optional)

```bash
python step3_test.py
```

Expected output:

```
Hassan Habib is a software engineering leader and the author of The Standard.
```

---

## 🦙 5. Convert to `.gguf` for llama.cpp (make sure you install CMake, clone and build llama.cpp)

```bash
cd llama.cpp/convert-hf-to-gguf
python3 convert.py ../../../tinyllama-merged --outfile standard-llama.gguf --outtype f16
```

Then run with:

```bash
cd ../../build/bin
./llama -m ../../../llama.cpp/convert-hf-to-gguf/standard-llama.gguf
```

Paste this prompt:

```
### Instruction:
Who is Hassan Habib?

### Input:

### Response:
```

---

## 📽️ Video Step-by-Step
https://youtu.be/FQr7VrK5RRQ

## 👨‍🏫 Author

Built and tested by [Hassan Habib](https://github.com/hassanhabib), fine-tuned with ❤️ and terminal grit.

---

Want to turn this into a video or GitHub tutorial? It’s built to teach.
