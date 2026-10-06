"""
RAG Pipeline scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of a from-scratch Retrieval-Augmented Generation pipeline."""
import subprocess
import sys
import importlib


def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


# Make sure optional dependencies used by solution.py are available before import.
_ensure("faiss-cpu", "faiss")
_ensure("rank-bm25", "rank_bm25")
_ensure("sentence-transformers", "sentence_transformers")
_ensure("transformers")
_ensure("torch")
_ensure("numpy")
_ensure("beautifulsoup4", "bs4")
_ensure("nltk")

import numpy as np
import torch

from solution import (
    load_text_file,
    load_text_directory,
    extract_text_from_html,
    normalize_text,
    make_document,
    chunk_fixed_size,
    chunk_by_tokens,
    chunk_by_sentences,
    chunk_with_overlap,
    attach_chunk_metadata,
    load_embedding_model,
    embed_text,
    embed_chunks,
    l2_normalize,
    save_corpus,
    cosine_similarity_search,
    top_k_indices,
    top_k_chunks,
    retrieve,
    build_faiss_index,
    faiss_search,
    compare_faiss_to_numpy,
    save_faiss_index,
    build_prompt_template,
    format_context,
    truncate_context,
    add_system_instruction,
    load_generator,
    generate_answer,
    rag_answer,
    track_source_chunk_ids,
    append_source_references,
    query_rewrite,
    hyde_retrieve,
    reciprocal_rank_fusion,
    bm25_search,
    hybrid_search,
    rerank_cross_encoder,
    maximal_marginal_relevance,
    filter_by_metadata,
    build_eval_set,
    hit_rate_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    faithfulness_score,
    relevance_score,
    handle_no_context,
    deduplicate_chunks,
    cache_query_embedding,
    update_chat_memory,
    rewrite_followup,
)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # 1) Pre-populated corpus (fixed here so the pipeline is reproducible).
    raw_docs = [
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was completed in 1889.",
        "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 60 million square miles.",
        "Python is a high-level programming language known for its readable syntax and large standard library.",
    ]
    docs = [normalize_text(d) for d in raw_docs]
    print(f"[corpus] {len(docs)} docs, first preview: {docs[0][:60]!r}")

    # 2) Chunk each document and attach metadata.
    all_chunks = []
    for i, doc in enumerate(docs):
        pieces = chunk_with_overlap(doc, chunk_size=120, overlap=20)
        all_chunks.extend(attach_chunk_metadata(pieces, source=f"doc_{i}"))
    print(f"[chunks] produced {len(all_chunks)} chunks")

    # 3) Load embedding model and embed all chunks (then L2-normalize).
    embed_model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_chunks(embed_model, all_chunks, batch_size=8)
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    print(f"[embeddings] shape={embeddings.shape}, dtype={embeddings.dtype}")

    # 4) Dense retrieval via numpy cosine search.
    question = build_eval_set()[0]["question"]
    clean_q = query_rewrite(question)
    print(f"[query] raw={question!r} clean={clean_q!r}")

    retrieved = retrieve(clean_q, embed_model, embeddings, all_chunks, k=3)
    for r in retrieved:
        chunk, score = r if isinstance(r, tuple) else (r, None)
        text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
        print(f"  retrieved score={score} text={text[:60]!r}")

    # 5) FAISS sanity check (same top-k as numpy path).
    index = build_faiss_index(embeddings)
    q_vec = l2_normalize(np.asarray([embed_text(embed_model, clean_q)], dtype=np.float32))[0]
    faiss_ids, faiss_scores = faiss_search(index, q_vec, k=3)
    print(f"[faiss] top ids={list(faiss_ids)} scores={[round(float(s),3) for s in faiss_scores]}")

    # 6) BM25 + hybrid for comparison.
    bm25_hits = bm25_search(clean_q, all_chunks, k=3)
    print(f"[bm25] {len(bm25_hits)} hits")
    hybrid_hits = hybrid_search(clean_q, all_chunks, embeddings, embed_model, alpha=0.5, k=3)
    print(f"[hybrid] {len(hybrid_hits)} hits")

    # 7) Prompt assembly + local generation.
    context_chunks = [r[0] if isinstance(r, tuple) else r for r in retrieved]
    gen_model, gen_tok = load_generator("sshleifer/tiny-gpt2")
    answer = rag_answer(clean_q, all_chunks, embeddings, embed_model, gen_model, gen_tok, k=3)
    answer_text = answer["answer"] if isinstance(answer, dict) else str(answer)
    print(f"[answer] {answer_text[:120]!r}")

    cited = append_source_references(answer_text, context_chunks)
    print(f"[cited] {cited[:160]!r}")

    # 8) Evaluation metrics on the toy eval set.
    eval_set = build_eval_set()
    retrieved_ids, relevant_ids = [], []
    for item in eval_set:
        hits = retrieve(query_rewrite(item["question"]), embed_model, embeddings, all_chunks, k=5)
        ids = []
        for h in hits:
            chunk = h[0] if isinstance(h, tuple) else h
            ids.append(chunk.get("chunk_id") if isinstance(chunk, dict) else chunk)
        retrieved_ids.append(ids)
        relevant_ids.append(item["relevant_ids"])
    print(f"[eval] hit@3={hit_rate_at_k(retrieved_ids, relevant_ids, 3):.2f} "
          f"recall@3={recall_at_k(retrieved_ids, relevant_ids, 3):.2f} "
          f"mrr={mean_reciprocal_rank(retrieved_ids, relevant_ids):.2f}")

    faith = faithfulness_score(answer_text, context_chunks)
    rel = relevance_score(answer_text, clean_q)
    print(f"[quality] faithfulness={faith:.2f} relevance={rel:.2f}")

    # 9) Conversational memory: a follow-up turn.
    history = update_chat_memory([], clean_q, answer_text)
    followup = "And what about its main benefits?"
    standalone = rewrite_followup(followup, history)
    print(f"[followup] standalone={standalone!r}")


if __name__ == "__main__":
    main()
