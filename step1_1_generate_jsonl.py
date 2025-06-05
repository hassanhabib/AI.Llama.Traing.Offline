# generate_questions_offline.py (CLI version)
import os
import json
import re
import nltk
import subprocess
import tempfile

# === SETUP ===
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# === CONFIGURATION ===
TXT_FILE = "TheStandard.txt"
JSONL_FILE = "TheStandard.jsonl"
RAW_LOG_FILE = "raw_output.jsonl"
MODEL_CLI = "llama.cpp/build/bin/llama-cli"
MODEL_PATH = "llama.cpp/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

CHUNK_MIN_WORDS = 30
CHUNK_MAX_WORDS = 100
MAX_PARAGRAPHS = 1000
MAX_QUESTIONS_PER_PARAGRAPH = 6

# === CLEAN TEXT ===
def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

# === PARAGRAPH SPLITTING ===
def split_into_paragraphs(text):
    paragraphs = []
    for block in text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        sentences = sent_tokenize(block)
        chunk = ""
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            word_count += len(words)
            chunk += " " + sentence
            if CHUNK_MIN_WORDS <= word_count <= CHUNK_MAX_WORDS:
                paragraphs.append(chunk.strip())
                chunk = ""
                word_count = 0
        if chunk:
            paragraphs.append(chunk.strip())
    return paragraphs

# === CALL CLI MODEL ===
def call_llama_cli(prompt):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_prompt:
        tmp_prompt.write(prompt)
        tmp_prompt_path = tmp_prompt.name

    result = subprocess.run(
        [MODEL_CLI, "--model", MODEL_PATH, "--prompt", prompt, "--n-predict", "256"],
        capture_output=True,
        text=True
    )

    os.remove(tmp_prompt_path)

    output = result.stdout.strip()
    return output

# === GENERATE QUESTIONS ===
def generate_questions(paragraph, max_q=MAX_QUESTIONS_PER_PARAGRAPH):
    prompt = f"""You are helping train an AI chatbot based on a book called *The Standard* by Hassan Habib.

Given the paragraph below, write up to 3 different natural-language questions that could be answered by it.
Write each question on a new line. Do not include explanations or extra commentary.

Paragraph:
{paragraph}
"""
    raw_text = call_llama_cli(prompt)

    with open(RAW_LOG_FILE, "a", encoding="utf-8") as log:
        log.write(json.dumps({
            "instruction": prompt.strip(),
            "input": "",
            "output": raw_text
        }, ensure_ascii=False) + "\n")

    questions = []
    for q in raw_text.split("\n"):
        q = q.strip()
        if q.endswith("?") and len(q.split()) >= 3 and not q.lower().startswith(("of", "and", "the")):
            q = re.sub(r'^\d+(\.\d+)*[\).]?\s*', '', q)
            questions.append(q)

    return list(set(questions))

# === MAIN ===
def main():
    if not os.path.exists(TXT_FILE):
        print(f"❌ File not found: {TXT_FILE}")
        return

    with open(TXT_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    paragraphs = split_into_paragraphs(clean_text(raw_text))
    print(f"📖 Found {len(paragraphs)} paragraphs to process.")

    written_count = 0
    skipped_count = 0

    with open(JSONL_FILE, 'w', encoding='utf-8') as out:
        for i, paragraph in enumerate(paragraphs[:MAX_PARAGRAPHS]):
            try:
                questions = generate_questions(paragraph)

                if not questions:
                    print(f"[{i+1}] ⚠️ No questions generated.")
                    skipped_count += 1
                    continue

                for j, question in enumerate(questions):
                    entry = {
                        "instruction": question,
                        "input": "",
                        "output": paragraph.strip()
                    }
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    written_count += 1
                    print(f"[{i+1}.{j+1}] ✅ {question}")

            except Exception as e:
                print(f"[{i+1}] ❌ Error: {e}")
                skipped_count += 1
                continue

    print(f"\n📦 Done: {written_count} questions saved to '{JSONL_FILE}'")
    print(f"⚠️ Skipped: {skipped_count} paragraphs")

# === ENTRY ===
if __name__ == "__main__":
    main()
