import os
import glob
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
<<<<<<< HEAD
=======
#Testing MJ
#Testing AP
>>>>>>> 16a0d064c4ed4e50c719ad4ddf197f9bffc53c70
# Embeddings and LLM
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Vector index
import faiss

# -------------------------------
# Utilities
# -------------------------------
def read_text_files(folder: str) -> List[Tuple[str, str]]:
    """Return list of (filename, text). Reads all .txt files recursively."""
    files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    docs = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read().strip()
            if txt:
                docs.append((f, txt))
    return docs


def simple_chunk(text: str, chunk_size: int = 80, overlap: int = 40) -> List[str]:
    """Simple word-based chunking with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks


def build_corpus(docs: List[Tuple[str, str]], chunk_size=80, overlap=40) -> List[Tuple[str, str, int]]:
    """Returns list of tuples: (source_filename, chunk_text, chunk_id_per_file)."""
    corpus = []
    for fname, txt in docs:
        chunks = simple_chunk(txt, chunk_size=chunk_size, overlap=overlap)
        for idx, c in enumerate(chunks):
            corpus.append((fname, c, idx))
    return corpus


# -------------------------------
# Embeddings + FAISS index
# -------------------------------
class VectorIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.corpus_meta = []  # holds (filename, chunk_text, idx)

    def fit(self, corpus: List[Tuple[str, str, int]]):
        self.corpus_meta = corpus
        texts = [c[1] for c in corpus]
        print(f"Encoding {len(texts)} chunks...")
        embs = self.embedder.encode(texts, batch_size=64, show_progress_bar=True,
                                    convert_to_numpy=True, normalize_embeddings=True)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine similarity (since normalized)
        self.index.add(embs.astype(np.float32))

    def query(self, question: str, top_k: int = 4) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([question], convert_to_numpy=True,
                                     normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for rank in range(idxs.shape[1]):
            i = idxs[0, rank]
            s = float(scores[0, rank])
            results.append((int(i), s))
        return results


# -------------------------------
# Tiny local LLM (T5-small) to generate concise answers
# -------------------------------
class LocalGenerator:
    def __init__(self, model_name="t5-small", device=None, max_new_tokens=32):  # EDIT: reduced max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,        # EDIT: greedy decoding for shorter answers
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# Prompt builder optimized for QA
# -------------------------------
def build_prompt(question: str, context_chunks: List[str]) -> str:
    # EDIT: changed from "summarize:" to "question:" for concise QA
    context = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(context_chunks)])
    prompt = (
        f"question: {question}\n"
        f"Answer using only the context below. "
        f"If not found, say 'I don't know'.\n\n"
        f"{context}\n\n"
        "Answer:"
    )
    return prompt


# -------------------------------
# RAG pipeline
# -------------------------------
def run_rag(data_folder="data", question="How many players are there in a cricket team on the field?", top_k=4):
    # 1) Load + chunk documents
    docs = read_text_files(data_folder)
    if not docs:
        raise RuntimeError(f"No .txt files found in folder: {data_folder}")

    corpus = build_corpus(docs, chunk_size=80, overlap=40)

    # 2) Build vector index
    vindex = VectorIndex()
    vindex.fit(corpus)

    # 3) Retrieve
    hits = vindex.query(question, top_k=top_k)
    retrieved = []
    for i, score in hits:
        fname, chunk, idx = vindex.corpus_meta[i]
        retrieved.append(chunk)

    # 4) Generate concise answer using local T5
    gen = LocalGenerator()
    prompt = build_prompt(question, retrieved)
    answer = gen.generate(prompt)

    # 5) Show result
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print("-" * 80)
    print("TOP CONTEXT CHUNKS:")
    for rank, (i, s) in enumerate(hits, start=1):
        fname, chunk, idx = vindex.corpus_meta[i]
        print(f"\n[{rank}] {fname} (score={s:.3f})")
        print(chunk[:80] + ("..." if len(chunk) > 80 else ""))
    print("-" * 80)
    print("ANSWER:")
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    run_rag(
        data_folder=r"C:\MyDrive\Learn\AI\ai-rag\data",
        question="Which country plays cricket?",
        top_k=4
    )


#keeping ready for AP