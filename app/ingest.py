# app/ingest.py

import pdfplumber
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text.strip()

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "।", "!", "?"]
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    pdf_path = os.path.join("data", "hsc_bangla.pdf")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Total characters extracted: {len(raw_text)}")

    chunks = chunk_text(raw_text)
    print(f"Total chunks: {len(chunks)}")

    # Optional: Save chunks to file for debugging
    with open("outputs/chunks_preview.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")
