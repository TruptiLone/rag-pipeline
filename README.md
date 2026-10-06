# RAG Pipeline

Construct a complete RAG system step by step: raw document ingestion and chunking, embeddings, dense and hybrid retrieval, prompt assembly, local generation, evaluation, and conversational memory. By the end you will have a pipeline that answers grounded questions with citations and can be measured on retrieval and answer-quality metrics.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_text_file
- [x] **2.** load_text_directory
- [x] **3.** extract_text_from_html
- [x] **4.** normalize_text
- [x] **5.** make_document
- [x] **6.** chunk_fixed_size
- [x] **7.** chunk_by_tokens
- [x] **8.** chunk_by_sentences
- [x] **9.** chunk_with_overlap
- [x] **10.** attach_chunk_metadata
- [x] **11.** load_embedding_model
- [x] **12.** embed_text
- [x] **13.** embed_chunks
- [x] **14.** l2_normalize
- [x] **15.** save_corpus
- [x] **16.** cosine_similarity_search
- [x] **17.** top_k_indices
- [x] **18.** top_k_chunks
- [x] **19.** retrieve
- [x] **20.** build_faiss_index
- [x] **21.** faiss_search
- [x] **22.** compare_faiss_to_numpy
- [x] **23.** save_faiss_index
- [x] **24.** build_prompt_template
- [x] **25.** format_context
- [x] **26.** truncate_context
- [x] **27.** add_system_instruction
- [x] **28.** load_generator
- [x] **29.** generate_answer
- [x] **30.** rag_answer
- [x] **31.** track_source_chunk_ids
- [x] **32.** append_source_references
- [x] **33.** query_rewrite
- [x] **34.** hyde_retrieve
- [x] **35.** reciprocal_rank_fusion
- [x] **36.** bm25_search
- [x] **37.** hybrid_search
- [x] **38.** rerank_cross_encoder
- [x] **39.** maximal_marginal_relevance
- [x] **40.** filter_by_metadata
- [x] **41.** build_eval_set
- [x] **42.** hit_rate_at_k
- [x] **43.** recall_at_k
- [x] **44.** mean_reciprocal_rank
- [x] **45.** faithfulness_score
- [x] **46.** relevance_score
- [x] **47.** handle_no_context
- [x] **48.** deduplicate_chunks
- [x] **49.** cache_query_embedding
- [x] **50.** update_chat_memory
- [x] **51.** rewrite_followup

---

Built on Deep-ML.
