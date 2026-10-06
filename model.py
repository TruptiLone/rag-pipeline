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

# Step 29 - generate_answer
import torch


def generate_answer(model, tokenizer, prompt, max_new_tokens=100):
    torch.manual_seed(42)

    if hasattr(model, "eval"):
        model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    if hasattr(model, "device"):
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=True,
    )

# Step 30 - rag_answer
def rag_answer(query, chunks, embeddings, embed_model, generator, tokenizer, k=3):
    # TODO: embed query, retrieve top-k chunks, build prompt, generate answer, return dict.
    retrieved = retrieve(query, embed_model, embeddings, chunks, k)

    prompt = build_prompt_template().format(
        context=format_context(retrieved),
        question=query,
    )
    prompt = add_system_instruction(prompt)

    answer = generate_answer(
        generator, tokenizer, prompt, max_new_tokens=100
    )

    return {
        "answer": answer,
        "sources": [chunk for chunk, score in retrieved],
        "query": query,
    }

# Step 31 - track_source_chunk_ids
def track_source_chunk_ids(source_chunks):
    # TODO: return the list of chunk ids from the retrieved source chunks, preserving order
    return [chunk["id"] for chunk in source_chunks if "id" in chunk]

# Step 32 - append_source_references
def append_source_references(answer_text, source_chunks):
    # TODO: append a 'Sources: [id1, id2, ...]' line to answer_text using the source chunk ids
    ids = track_source_chunk_ids(source_chunks)
    references = ", ".join(str(chunk_id) for chunk_id in ids)
    return f"{answer_text}\nSources: [{references}]"

# Step 33 - query_rewrite
def query_rewrite(raw_query):
    # TODO: clean and normalize a raw user query into a better search query
    query = normalize_text(raw_query).lower()

    # Repeatedly remove prefixes to handle combinations like
    # "please could you tell me".
    filler = r"^(?:please|could you|can you|tell me|i want to know)\b[\s,]*"

    while True:
        cleaned = re.sub(filler, "", query)
        if cleaned == query:
            break
        query = cleaned

    return query.rstrip("?.! ").strip()

# Step 34 - hyde_retrieve
def hyde_retrieve(query, hypothetical_answer, chunks, embeddings, embed_model, k=5):
    # TODO: embed the hypothetical answer and return the top-k chunks by cosine similarity.
    vector = embed_text(embed_model, hypothetical_answer)
    scores = cosine_similarity_search(vector, embeddings)
    ranked = top_k_chunks(scores, chunks, k)

    return [chunk for chunk, score in ranked]

# Step 35 - reciprocal_rank_fusion
def reciprocal_rank_fusion(ranked_lists, k=60):
    # TODO: merge ranked lists of ids into one (id, score) list sorted by fused score.
    scores = {}

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

# Step 36 - bm25_search
import math
from collections import Counter

def bm25_search(query, chunks, k=5, k1=1.5, b=0.75):
    # TODO: score chunks against the query with BM25 and return top-k (index, score) pairs
    documents = [
        (chunk["text"] if isinstance(chunk, dict) else chunk).lower().split()
        for chunk in chunks
    ]
    if not documents or k <= 0:
        return []

    query_terms = set(query.lower().split())
    term_counts = [Counter(tokens) for tokens in documents]
    document_freq = Counter(
        term for counts in term_counts for term in counts
    )

    n = len(documents)
    avg_length = sum(map(len, documents)) / n
    if avg_length == 0:
        return []

    results = []
    for index, (tokens, counts) in enumerate(zip(documents, term_counts)):
        overlap = query_terms.intersection(counts)
        if not overlap:
            continue

        length_factor = k1 * (1 - b + b * len(tokens) / avg_length)
        score = 0.0

        for term in overlap:
            df = document_freq[term]
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf = counts[term]
            score += idf * tf * (k1 + 1) / (tf + length_factor)

        results.append((index, float(score)))
    return sorted(results, key=lambda item: item[1], reverse=True)[:k]

# Step 37 - hybrid_search
import numpy as np

def hybrid_search(query, chunks, embeddings, embed_model, alpha=0.5, k=5):
    # TODO: blend normalized dense cosine scores with BM25 scores and return the top-k (idx, score) pairs.
    if not chunks or k <= 0:
        return []

    query_vector = embed_text(embed_model, query)
    dense_scores = cosine_similarity_search(query_vector, embeddings)

    lexical_scores = np.zeros(len(chunks), dtype=float)
    for index, score in bm25_search(query, chunks, k=len(chunks)):
        lexical_scores[index] = score

    def min_max_scale(scores):
        scores = np.asarray(scores, dtype=float)
        span = scores.max() - scores.min()
        if span == 0:
            return np.zeros_like(scores)
        return (scores - scores.min()) / span

    combined = (
        alpha * min_max_scale(dense_scores)
        + (1 - alpha) * min_max_scale(lexical_scores)
    )

    indices = np.argsort(-combined, kind="stable")[:k]
    return [(int(i), float(combined[i])) for i in indices]

# Step 38 - rerank_cross_encoder
def rerank_cross_encoder(query, candidate_chunks, cross_encoder):
    # TODO: score (query, chunk) pairs with cross_encoder and return chunks sorted by descending score
    if not candidate_chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidate_chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [chunk for chunk, score in ranked]

# Step 39 - maximal_marginal_relevance
def maximal_marginal_relevance(query_embedding, candidate_embeddings, k=5, lambda_param=0.5):
    # TODO: greedily pick indices balancing query relevance and diversity from already-selected items.
    candidates = np.asarray(candidate_embeddings, dtype=float)
    query = np.asarray(query_embedding, dtype=float)

    if k <= 0 or len(candidates) == 0:
        return []

    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")

    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    relevance = candidates @ query

    # Pick the most relevant candidate first.
    first = int(np.argmax(relevance))
    selected = [first]
    available = np.ones(len(candidates), dtype=bool)
    available[first] = False
    max_similarity = candidates @ candidates[first]

    while len(selected) < min(k, len(candidates)):
        scores = (
            lambda_param * relevance
            - (1 - lambda_param) * max_similarity
        )
        scores[~available] = -np.inf

        # Ties go to the smaller index.
        index = int(np.argmax(scores))
        selected.append(index)
        available[index] = False

        max_similarity = np.maximum(
            max_similarity,
            candidates @ candidates[index],
        )

    return selected

# Step 40 - filter_by_metadata
def filter_by_metadata(chunks, filter_dict):
    # TODO: return only chunks whose metadata contains every key/value pair in filter_dict
    return [
        chunk
        for chunk in chunks
        if all(
            key in chunk.get("metadata", {})
            and chunk["metadata"][key] == value
            for key, value in filter_dict.items()
        )
    ]

# Step 41 - build_eval_set
def build_eval_set():
    return [
        {
            "question": "What is RAG?",
            "answer": "Retrieval-Augmented Generation combines a retriever with a generator.",
            "relevant_ids": ["c1", "c2"],
        },
        {
            "question": "What does FAISS do?",
            "answer": "FAISS performs fast nearest-neighbor search over dense vectors.",
            "relevant_ids": ["c3"],
        },
        {
            "question": "Why normalize embeddings?",
            "answer": "So that inner products equal cosine similarities.",
            "relevant_ids": ["c4", "c5"],
        },
        {
            "question": "What is BM25?",
            "answer": "A lexical ranking function based on term frequency and document length.",
            "relevant_ids": ["c6"],
        },
    ]

# Step 42 - hit_rate_at_k
def hit_rate_at_k(retrieved_ids_per_query, relevant_ids_per_query, k):
    # TODO: return the fraction of queries with at least one relevant id in the top-k retrieved
    if not retrieved_ids_per_query or k <= 0:
        return 0.0

    hits = sum(
        bool(set(retrieved[:k]) & set(relevant))
        for retrieved, relevant in zip(retrieved_ids_per_query, relevant_ids_per_query)
    )
    return float(hits / len(retrieved_ids_per_query))

# Step 43 - recall_at_k
def recall_at_k(retrieved_ids_per_query, relevant_ids_per_query, k):
    # TODO: average over queries the fraction of relevant ids found in the top-k retrieved ids
    if not retrieved_ids_per_query or k <= 0:
        return 0.0

    total_recall = 0.0

    for retrieved, relevant in zip(
        retrieved_ids_per_query, relevant_ids_per_query
    ):
        gold = set(relevant)
        if gold:
            total_recall += len(set(retrieved[:k]) & gold) / len(gold)

    return float(total_recall / len(retrieved_ids_per_query))

# Step 44 - mean_reciprocal_rank
def mean_reciprocal_rank(retrieved_ids_per_query, relevant_ids_per_query):
    # TODO: average the reciprocal rank of the first relevant id across queries
    if not retrieved_ids_per_query:
        return 0.0

    total = 0.0

    for retrieved, relevant in zip(
        retrieved_ids_per_query, relevant_ids_per_query
    ):
        gold = set(relevant)

        for position, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in gold:
                total += 1.0 / position
                break

    return float(total / len(retrieved_ids_per_query))

# Step 45 - faithfulness_score
def faithfulness_score(answer, context_chunks):
    # TODO: return the fraction of answer tokens that appear in the context text
    answer_tokens = normalize_text(answer).lower().split()
    if not answer_tokens:
        return 0.0

    context = " ".join(
        chunk["text"] if isinstance(chunk, dict) else chunk
        for chunk in context_chunks
    )
    context_tokens = set(normalize_text(context).lower().split())

    supported = sum(token in context_tokens for token in answer_tokens)
    return float(supported / len(answer_tokens))

# Step 46 - relevance_score
def relevance_score(answer, question):
    # TODO: return token-overlap (Jaccard) similarity between answer and question in [0, 1]
    answer_tokens = set(
        re.findall(r"\w+", normalize_text(answer).lower())
    )
    question_tokens = set(
        re.findall(r"\w+", normalize_text(question).lower())
    )

    union = answer_tokens | question_tokens
    if not union:
        return 0.0

    return float(len(answer_tokens & question_tokens) / len(union))

# Step 47 - handle_no_context
def handle_no_context(scored_chunks, threshold=0.2):
    """Return {'abstain': bool, 'message': str} based on top score vs threshold."""
    # TODO: abstain when no chunk's score strictly exceeds the threshold
    has_context = any(
        (item["score"] if isinstance(item, dict) else item[1]) > threshold
        for item in scored_chunks
    )

    return {
        "abstain": not has_context,
        "message": "" if has_context else "I do not know",
    }

# Step 48 - deduplicate_chunks
import numpy as np

def deduplicate_chunks(chunks, embeddings, similarity_threshold=0.95):
    embeddings = np.asarray(embeddings)

    if not chunks:
        return [], np.empty((0, 0), dtype=embeddings.dtype)

    kept_indices = []

    for i in range(len(chunks)):
        if kept_indices:
            similarities = embeddings[kept_indices] @ embeddings[i]
            if np.any(similarities > similarity_threshold):
                continue

        kept_indices.append(i)

    return (
        [chunks[i] for i in kept_indices],
        embeddings[kept_indices],
    )

