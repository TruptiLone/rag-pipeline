"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    # TODO: read a UTF-8 text file at `path` and return its contents as one string.
    with open(path, "r", encoding="utf-8", newline="") as file:
        return file.read()

# Step 2 - load_text_directory
from pathlib import Path

def load_text_directory(directory):
    # TODO: read every .txt file in `directory` and return their contents as a list of strings
    # Read only .txt files, ordered lexicographically by filename.
    return [
        load_text_file(file)
        for file in sorted(Path(directory).iterdir(), key=lambda file: file.name)
        if file.is_file() and file.suffix == ".txt"
    ]

# Step 3 - extract_text_from_html
from html.parser import HTMLParser


def extract_text_from_html(html):
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.hidden_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.hidden_depth += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self.hidden_depth:
                self.hidden_depth -= 1

        def handle_data(self, data):
            if not self.hidden_depth:
                self.parts.append(data)

    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)

# Step 4 - normalize_text
import unicodedata

def normalize_text(text):
    # TODO: NFKC-normalize the text and collapse runs of whitespace into single spaces.
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())

# Step 5 - make_document
def make_document(text, source, title):
    # TODO: wrap text with source and title metadata into a document dict.
    return {
        "text": text,
        "source": source,
        "title": title,
    }

# Step 6 - chunk_fixed_size
def chunk_fixed_size(text, chunk_size):
    # TODO: split text into consecutive non-overlapping chunks of length chunk_size
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), chunk_size)
    ]

# Step 7 - chunk_by_tokens
def chunk_by_tokens(text, tokenizer, max_tokens):
    # TODO: split text into chunks of at most max_tokens token ids using the tokenizer
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    return [
        tokenizer.decode(token_ids[start:start + max_tokens])
        for start in range(0, len(token_ids), max_tokens)
    ]

# Step 8 - chunk_by_sentences
import re

def chunk_by_sentences(text, max_chars):
    # TODO: split text on .!? boundaries and greedily pack whole sentences under max_chars.
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    sentences = re.findall(r"[^.!?]+[.!?]*|[.!?]+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        combined = f"{current} {sentence}" if current else sentence

        if current and len(combined) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = combined

    if current:
        chunks.append(current)

    return chunks

# Step 9 - chunk_with_overlap
def chunk_with_overlap(text, chunk_size, overlap):
    # TODO: return sliding-window chunks of length chunk_size sharing `overlap` chars
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
    ]

# Step 10 - attach_chunk_metadata
def attach_chunk_metadata(chunks, source):
    # TODO: wrap each chunk string with source, position, and chunk_id metadata.
    return [
        {
            "text": chunk,
            "source": source,
            "position": position,
            "chunk_id": f"{source}::{position}",
        }
        for position, chunk in enumerate(chunks)
    ]

# Step 11 - load_embedding_model
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    # TODO: return a sentence-transformers model instance for the given model_name.
    return SentenceTransformer(model_name)

# Step 12 - embed_text
import numpy as np

def embed_text(model, text):
    # TODO: Return a 1D float32 numpy embedding vector for the given text string.
    return np.asarray(model.encode(text), dtype=np.float32).reshape(-1)

# Step 13 - embed_chunks
def embed_chunks(model, chunks, batch_size=32):
    """Batch-embed a list of chunk strings or chunk dicts into a 2D float32 matrix."""
    # TODO: normalize chunk inputs to strings, encode in batches, return (n, d) float32 array
    texts = [
        chunk["text"] if isinstance(chunk, dict) else chunk
        for chunk in chunks
    ]

    if not texts:
        return np.empty(
            (0, model.get_sentence_embedding_dimension()),
            dtype=np.float32,
        )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)

# Step 14 - l2_normalize
import numpy as np

def l2_normalize(matrix):
    # TODO: rescale each row of `matrix` to unit L2 norm, leaving all-zero rows unchanged.
    normalized = np.array(matrix, copy=True)
    if not np.issubdtype(normalized.dtype, np.floating):
        normalized = normalized.astype(float)

    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    np.divide(normalized, norms, out=normalized, where=norms != 0)
    return normalized

# Step 15 - save_corpus
import json
from pathlib import Path

import numpy as np
def save_corpus(embeddings, chunks, directory):
    # TODO: persist embeddings (.npy) and chunks (.json) into directory, then reload and return both.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    embeddings_path = directory / "embeddings.npy"
    chunks_path = directory / "chunks.json"

    np.save(embeddings_path, embeddings, allow_pickle=False)
    with chunks_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False)

    with chunks_path.open("r", encoding="utf-8") as file:
        loaded_chunks = json.load(file)

    return {
        "embeddings": np.load(embeddings_path, allow_pickle=False),
        "chunks": loaded_chunks,
    }

# Step 16 - cosine_similarity_search
import numpy as np

def cosine_similarity_search(query_vector, chunk_matrix):
    """Cosine similarity between query_vector (d,) and each row of chunk_matrix (n,d)."""
    # TODO: compute cosine similarity between the query vector and every chunk row
    query = np.asarray(query_vector, dtype=float)
    chunks = np.asarray(chunk_matrix, dtype=float)

    dot_products = chunks @ query
    norms = np.linalg.norm(chunks, axis=1) * np.linalg.norm(query)

    # Assign zero similarity when either vector has zero norm.
    return np.divide(
        dot_products,
        norms,
        out=np.zeros_like(dot_products),
        where=norms != 0,
    )

# Step 17 - top_k_indices
import numpy as np

def top_k_indices(scores, k):
    """Return indices of the k highest scores in descending order."""
    # TODO: rank the score array and return the top-k positions as a numpy array
    return np.argsort(-np.asarray(scores), kind="stable")[:k]

# Step 18 - top_k_chunks
import numpy as np

def top_k_chunks(scores, chunks, k):
    # TODO: return list of (chunk, score) tuples for the top-k scores, sorted descending
    indices = top_k_indices(scores, min(k, len(chunks)))
    return [(chunks[i], float(scores[i])) for i in indices]

# Step 19 - retrieve
def retrieve(query, model, chunk_matrix, chunks, k):
    # TODO: embed the query, score it against chunk_matrix, return top-k (chunk, score) pairs.
    query_vector = embed_text(model, query)
    scores = cosine_similarity_search(query_vector, chunk_matrix)
    return top_k_chunks(scores, chunks, k)

# Step 20 - build_faiss_index
import faiss
import numpy as np

def build_faiss_index(chunk_matrix):
    # TODO: build a FAISS inner-product index and add all rows of chunk_matrix to it
    vectors = np.ascontiguousarray(chunk_matrix, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

# Step 21 - faiss_search
import numpy as np

def faiss_search(index, query_vector, k):
    """Return top-k (scores, indices) as 1D arrays for a single query vector."""
    # TODO: query the FAISS index with the single query vector and return flat top-k arrays
    query = np.ascontiguousarray(
    query_vector, dtype=np.float32
    ).reshape(1, -1)

    if index.ntotal == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    scores, indices = index.search(query, index.ntotal)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    indices = np.asarray(indices, dtype=np.int64).ravel()

    # Highest score first; smallest index first for equal scores.
    order = np.lexsort((indices, -scores))[:k]
    return scores[order], indices[order]

# Step 22 - compare_faiss_to_numpy
import numpy as np

def compare_faiss_to_numpy(query_vector, chunk_matrix, index, k):
    # TODO: return True iff FAISS and numpy cosine search agree on the top-k indices
    k = min(k, len(chunk_matrix))
    if k == 0:
        return True

    scores = cosine_similarity_search(query_vector, chunk_matrix)
    numpy_indices = top_k_indices(scores, k)

    normalized_chunks = l2_normalize(chunk_matrix)
    normalized_query = l2_normalize(
        np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    )[0]

    index = build_faiss_index(normalized_chunks)
    _, faiss_indices = faiss_search(index, normalized_query, k)

    return set(numpy_indices.tolist()) == set(faiss_indices.tolist())

# Step 23 - save_faiss_index
import faiss

def save_faiss_index(index, path):
    """Write `index` to `path` and return the index loaded back from disk."""
    # TODO: persist the index to `path` and reload it; return the reloaded index
    faiss.write_index(index, str(path))
    return faiss.read_index(str(path))

# Step 24 - build_prompt_template
def build_prompt_template():
    # TODO: return a RAG prompt template string with {context} and {question} placeholders.
    return (
        "Answer the question using only the provided context. "
        "If the context does not contain the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

# Step 25 - format_context
def format_context(retrieved):
    # TODO: render each (chunk, score) as '[i] {text} (source={source})' and join with newlines
    return "\n".join(
        f"[{i}] {chunk['text']} (source={chunk['source']})"
        for i, (chunk, score) in enumerate(retrieved, start=1)
    )

# Step 26 - truncate_context
def truncate_context(context, max_chars):
    # TODO: trim context so len(result) <= max_chars, preferring a whitespace boundary
    if max_chars <= 0:
        return ""
    if len(context) <= max_chars:
        return context

    # Keep a complete word when the limit lands just before whitespace.
    if context[max_chars].isspace():
        return context[:max_chars].rstrip()

    # Otherwise, cut at the last whitespace within the budget.
    for i in range(max_chars - 1, -1, -1):
        if context[i].isspace():
            return context[:i].rstrip()

    return context[:max_chars]

# Step 27 - add_system_instruction
def add_system_instruction(prompt):
    """Prepend a fixed system instruction to the prompt."""
    # TODO: return a string that starts with a system instruction telling the model to use only the context
    instruction = (
        "You are a helpful assistant. "
        "Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I do not know'."
    )
    return instruction + "\n\n" + prompt

# Step 28 - load_generator
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_generator(model_name='sshleifer/tiny-gpt2'):
    # TODO: load a small local causal LM and its tokenizer, ensuring tokenizer.pad_token is set.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer

