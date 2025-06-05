import fitz  # PyMuPDF
import os
import re


def clean_text(text):
    # Replace hyphenated words at line breaks (e.g., knowl-\nedge → knowledge)
    text = re.sub(r'-\n', '', text)

    # Join lines inside a paragraph (i.e., not between paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Normalize multiple newlines to exactly 2
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()


def convert_pdf_to_clean_txt(pdf_path, txt_path):
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    print(f"📄 Reading '{pdf_path}' with {len(doc)} pages...")

    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += text + "\n\n"  # separate pages
        print(f"✅ Page {page_num} extracted")

    clean = clean_text(full_text)

    with open(txt_path, "w", encoding="utf-8") as out:
        out.write(clean)

    print(f"\n✅ Cleaned text saved to '{txt_path}'")


if __name__ == "__main__":
    pdf_input = "TheStandard.pdf"
    txt_output = "TheStandard.txt"

    convert_pdf_to_clean_txt(pdf_input, txt_output)
