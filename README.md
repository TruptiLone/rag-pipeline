# Technical Knowledge Assistant — RAG Pipeline

This project searches a collection of technical documents to help answer questions with supporting passages. The included corpus contains **12 project-authored notes about retrieval-augmented generation (RAG)**: chunking, embeddings, cosine similarity, FAISS, BM25, hybrid retrieval, reranking, evaluation, grounding, persistence, and conversation memory. Each document is a UTF-8 text file; each searchable record is a chunk with text, a source, a position, and a stable `chunk_id`.

A user supplies a question; the system retrieves passages and can pass them to a pretrained language model to produce an answer. The model is not trained from scratch. The current interface is a command-line application; a web frontend and backend service are planned extensions.

## RAG explained: design decisions, evaluation, and interview preparation

### A short project summary

This project explores how to answer questions from a known collection of documents instead of relying only on a language model's internal knowledge. It separates finding evidence from writing an answer, preserves the sources supplied to the model, and tests retrieval independently of generation. The current deliverable is a local command-line prototype with a small technical corpus, measured BM25 retrieval, and integration tests—not a deployed enterprise assistant.

A useful interview explanation is: “The difficult part of RAG is making sure the right evidence reaches the model and then checking whether the answer actually follows that evidence. I organized the pipeline so those two failure modes can be investigated separately.”

### What each stage means and why it exists

| Stage | Definition | Why it is needed | This project's implementation |
|---|---|---|---|
| Ingestion | Read source material and capture its identity | A system cannot cite or update information reliably without knowing where it came from | Local UTF-8 notes and a document manifest; an HTML extractor exists, but there is no web crawler |
| Normalization | Make text formatting consistent | Unicode variants and accidental whitespace can interfere with matching and create duplicate representations | NFKC normalization and whitespace collapsing; raw files remain available |
| Chunking | Divide documents into retrievable passages | Whole documents can be too large or unfocused for a question and the model's input limit | Character, token, sentence, and overlap helpers; the demo uses 800 characters with 100-character overlap |
| Embedding and indexing | Encode passages as vectors and organize them for lookup | Semantic retrieval can find related meanings even when exact words differ | Sentence-transformer embeddings, L2 normalization, NumPy search, and an optional FAISS index |
| Retrieval | Select passages likely to help answer a question | The generator needs relevant evidence rather than the entire corpus | BM25, dense cosine retrieval, and hybrid comparison; generation currently uses dense retrieval |
| Reranking | Re-score a shortlist or reduce redundant passages | First-stage retrieval can return loosely relevant or repetitive content | Cross-encoder and MMR helpers exist but are not automatically used by `rag_answer()` |
| Context preparation | Assemble evidence, source labels, instructions, and the question within a token budget | Prevents oversized prompts and keeps the evidence traceable | Ranked context formatting, a score-based abstention check, and tokenizer-aware budgeting |
| Generation | Ask a pretrained language model to produce a continuation using the context | Converts retrieved evidence into an understandable response | Optional local instruction model; returns answer text and the supplied source records |
| Evaluation | Measure retrieval and answer behavior on labeled questions | A fluent answer can still be wrong, and passing unit tests does not prove useful retrieval | Retrieval metrics are measured; answer-quality validation remains a separate milestone |

There are two different flows: **indexing** prepares the corpus when documents change; **query serving** retrieves and generates when a user asks a question. The CLI demonstrates both in one process. A service should persist the prepared index and load models once rather than rebuilding them for every request.

### Evaluation metrics versus configuration parameters

**Metrics are outcomes we measure. Parameters are settings we change.** For example, recall@3 is a metric; choosing `k=3` is a retrieval parameter. Higher similarity is not the same as a higher probability that an answer is correct.

| Metric | How it is measured | What it tells us | Current status |
|---|---|---|---|
| Hit rate@k | Queries with at least one gold-relevant top-k result ÷ evaluated queries | Whether retrieval finds any useful evidence | Implemented and measured for BM25 |
| Recall@k | For each query: distinct relevant IDs found in top k ÷ all relevant IDs; average across queries | How much required evidence was recovered | Implemented and measured; empty gold sets contribute zero in the helper |
| Precision@k | Relevant retrieved results ÷ retrieved results up to k | How much irrelevant material reaches the context | Proposed; not yet implemented |
| MRR@k | Average `1 / rank` of the first relevant result, or zero if absent from top k | Whether useful evidence appears near the top | Implemented; the report evaluates lists truncated to three |
| Groundedness | Review whether each substantive claim is supported by the supplied passages | Whether generation invents or contradicts evidence | Not yet measured; word overlap is only a heuristic |
| Answer correctness and completeness | Compare against reviewed reference answers and required facts | Whether the response answers the question accurately and sufficiently | Reference answers exist; semantic assessment is not implemented |
| Citation correctness | Check whether each cited passage supports its associated claim | Whether references provide evidence rather than decoration | Source records are returned; claim-level validation is not implemented |
| Abstention behavior | Measure refusal on unanswerable questions and unnecessary refusal on answerable ones separately | Whether the assistant knows when it lacks evidence | Gating tested with test doubles; end-to-end rates unmeasured |
| Latency and cost | Measure retrieval, time to first token, total response time, token use, and compute cost | Whether the system is usable under realistic load | Mean BM25 query time recorded; p50/p95 and generation costs unmeasured |

**Worked example:** gold evidence is `{A, B}` and the top three results are `[X, A, C]`. Hit@3 is `1`, recall@3 is `1/2`, precision@3 is `1/3`, and reciprocal rank is `1/2`. Retrieving one useful passage succeeds on hit rate but still misses half the evidence.

In this repository, `faithfulness_score()` counts answer tokens appearing in context, and `relevance_score()` calculates Jaccard token overlap with the question. Neither proves semantic correctness. For example, “FAISS does not search vectors” shares many words with a passage saying it does, while reversing its meaning. Production evaluation should distinguish retrieved evidence quality, answer relevance, groundedness, and completeness; human review should check the reliability of automated judging. [Microsoft evaluation guidance](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-llm-evaluation-phase).

| Configuration | Tradeoff to investigate |
|---|---|
| Chunk size and overlap | Small chunks can lose context; large chunks can dilute relevance; more overlap adds duplication |
| Retrieval `k` | More passages may recover evidence but increase noise, prompt length, and cost |
| Embedding model | Representation quality, language/domain coverage, speed, memory, and the need to rebuild the index |
| BM25 `k1` and `b` | Term-frequency saturation and document-length normalization |
| Hybrid `alpha` | Relative contribution of normalized dense and lexical scores; here `0.5` is a default, not an optimized result |
| Reranker candidate count / MMR lambda | Relevance and diversity gains versus additional computation |
| Abstention threshold | Fewer unsupported answers can mean more unnecessary refusals; calibrate on development questions |
| Input/output token budgets | Evidence coverage and answer length versus latency and model limits |

Change settings using development questions, record the configuration, and evaluate the selected approach on untouched test questions. Track performance by question type rather than relying only on one average. The current 12-question answerable test set is too small and synthetic to establish real-world reliability.

### Technical challenges and decisions to discuss

| Challenge | Engineering response | Evidence or limitation in this project |
|---|---|---|
| Broken provenance across stages | Standardize identifiers and check labels against the corpus | Fixed `id` versus `chunk_id`; labels now resolve to actual passages |
| Misleading evaluation | Align questions with source content and separate development/test examples | Replaced unrelated demo questions; synthetic scope is explicitly disclosed |
| Backend disagreement | Validate scores and results after index persistence; account for equal-score ties | Tests check the supplied FAISS index and detect an incorrectly ordered index |
| Missing context versus model failure | Inspect retrieval first, then the prompt, then generated claims | Separate retrieval reporting and mocked generation-path tests |
| Prompt budget overflow | Count tokens and reserve space for output before generation | The answer path admits whole passages only when they fit |
| Stale documents and embeddings | Version the corpus, configuration, and model; invalidate affected artifacts | Corpus checksum and model/chunking config exist; automatic refresh is still future work |
| Exact terms versus paraphrases | Compare lexical, dense, and hybrid retrieval under the same labels | BM25 measured; neural comparisons available to run, not yet reported |
| Cost and latency | Reuse models, cache embeddings, shortlist before expensive reranking, and profile each stage | Cache helper exists; no production load benchmark yet |

In a multi-user system, document permissions must be enforced before evidence enters the prompt. Retrieved text is untrusted data and can contain prompt-injection instructions. A “use only the context” prompt is not a security boundary. Source validation, retrieval-time authorization, safe rendering, and adversarial tests are additional requirements, not features this local prototype already provides. [OWASP RAG security guidance](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html).

### Where RAG is used and how it is evolving

**Research checked October 6, 2026.** RAG is an established application architecture, while methods for choosing and managing evidence continue to evolve. Common use cases include technical documentation assistants, internal knowledge search, product-support answers, and research tools that need attributable source passages. Enterprise guidance now addresses permission-aware retrieval and operational security rather than treating retrieval as just a vector lookup. [OWASP RAG security guidance](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html).

Two relevant directions are:

- **Context-aware indexing and hybrid reranking:** preserve document context around a chunk, combine semantic and lexical retrieval, and rerank candidates. Anthropic's contextual retrieval work illustrates this approach; its benchmark results should not be assumed to transfer to this corpus. [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).
- **Agentic retrieval:** plan multiple searches for a complex question and combine their evidence. Current Azure documentation describes multi-query planning and reranking, with some capabilities still marked preview. Extra planning also introduces latency and evaluation complexity. This project uses a fixed retrieval flow, not an autonomous agent. [Azure agentic retrieval](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview).

A larger context window does not remove the need to decide what information is current, relevant, and authorized. RAG and fine-tuning also solve different problems: retrieval supplies external evidence at query time; fine-tuning changes model behavior through training. They can be combined. For this small corpus, a whole-corpus prompt is a useful future baseline to test whether retrieval adds value at all.

### Interview discussion: use evidence, explain tradeoffs

Use these as preparation prompts, and describe only work you can explain and reproduce. Behavioral answers should focus on decisions, debugging, and lessons—not just list library names.

**“Tell me about the project.”**

> “I built a local document-question-answering pipeline with traceable evidence. I separated ingestion, retrieval, and generation so I could test them independently. The current evaluation covers a small technical corpus, and I am treating those results as integration evidence rather than claiming production accuracy.”

**“Describe a technical problem you encountered.” — STAR outline**

- **Situation:** the assembled demo had questions unrelated to its documents and citation IDs inconsistent with generated chunk IDs.
- **Task:** make the experiment test the actual pipeline rather than produce misleading scores.
- **Action:** inspect the data flow, standardize chunk identifiers, replace the corpus/evaluation mismatch, and add checks for missing evidence IDs and index round trips.
- **Result:** 13 tests pass. BM25 achieves hit@3 and recall@3 of 1.000 and MRR@3 of 0.958 on 12 synthetic answerable test questions. These results do not establish generated-answer quality.
- **Learning:** independently correct functions can still fail as a system when their metadata contracts disagree.

**“Why not immediately use the most advanced retrieval method?”**

> “I established a cheap, inspectable BM25 baseline first. Dense retrieval and reranking add dependencies and computation. I would adopt them when the same evaluation demonstrates a meaningful benefit, especially on paraphrases or ambiguous queries.”

**“How would you diagnose a wrong answer?”**

> “First check whether the answer exists in the corpus. Then check whether chunking preserved it, retrieval found it, and context preparation included it. If the evidence was present, inspect whether generation followed it. This locates the failure before changing the model or prompt.”

**“What would you improve before deployment?”**

> “I would expand and independently review the evaluation set, measure unanswerable-question behavior and citation correctness, validate neural generation, and then add access control, document refresh, request tracing, and realistic latency tests. The current prototype does not claim those capabilities.”


## Example input and output

- **Question:** “Why L2-normalize embeddings before inner-product search?”
- **Supporting document:** [Cosine similarity and normalization](data/documents/cosine.txt), chunk `cosine::0`.
- **Desired answer:** “For nonzero normalized vectors, inner products equal cosine similarities.”
- **Output structure:** `{"answer": "...", "sources": [{"chunk_id": "cosine::0", "text": "...", "source": "cosine"}], "query": "..."}`.

This is an illustrative desired answer, not a recorded generation result. Retrieved sources show the evidence supplied to the model; they do not automatically prove that every generated claim is supported.

## What is the source of the data?

**No website is scraped in the included demo.** The corpus is a small synthetic collection authored for this project, stored in [data/documents](data/documents). The [manifest](data/manifest.json) identifies each document; URLs are null because these notes have no upstream webpage. See [data/README.md](data/README.md) for provenance and evaluation limitations.

The HTML text extractor is available as a utility, but it does not fetch webpages. A future web ingestion feature should store source URLs, retrieval dates, and document versions. The application currently loads local text files.

## Workflow

```text
Local documents → normalize → chunk → attach source metadata
                                      │
                ┌─────────────────────┴────────────────────┐
                │ BM25 lexical retrieval                   │
                │ Embeddings → cosine / FAISS retrieval    │
                └─────────────────────┬────────────────────┘
                                      │
                Compare BM25, dense, and hybrid retrieval
                                      │
                Dense answer path → evidence-score check
                                      │
                Fit evidence into token budget → prompt
                                      │
                Instruction model → answer + source records
```

The default run evaluates BM25 without downloading any model. `--dense` adds semantic and hybrid comparisons, saves the corpus and index, and checks the reloaded FAISS index against NumPy. `--generate` additionally runs one question through the dense RAG answer path. Hybrid retrieval and reranking helpers are available, but the generator currently uses dense retrieval.

## Recorded results

The committed [BM25 evaluation report](reports/retrieval_test_bm25.json) records:

| Metric | Result |
|---|---:|
| Answerable test questions | 12 |
| Searchable chunks | 12 |
| Hit rate @ 3 | 1.000 |
| Recall @ 3 | 1.000 |
| MRR @ 3 | 0.958 |

These are **small synthetic sanity-check results**, not a production benchmark. Each answerable question has one relevant chunk, so hit rate and recall coincide. The report includes each query's retrieved IDs, the corpus checksum, runtime versions, and timing. No retrieval parameters were tuned on this test set.

The full evaluation file contains 12 development questions, 12 answerable test questions, and 4 out-of-scope test questions. Out-of-scope questions are excluded from the retrieval averages and reserved for a future generation/abstention evaluation. **Dense/hybrid model runs and generated-answer quality have not been measured in the committed report.**

The functions named `faithfulness_score` and `relevance_score` are lexical overlap heuristics. They do not detect factual contradictions, validate citations, or establish semantic correctness.

## Run locally

Python 3.10+ is recommended. The verified lightweight run used Python and NumPy versions recorded in the report.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scaffold.py
```

For embeddings, FAISS, and local answer generation, install the optional dependencies:

```bash
python -m pip install -r requirements-models.txt
python scaffold.py --dense
python scaffold.py --generate --question "What does BM25 k1 control?"
```

The first model-backed run requires internet access to download model weights. The code does not install packages automatically. Dependency ranges are compatibility targets, not a fully locked environment.

Models:

- Embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Default generator: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), a compact instruction model. Its suitability for this corpus still needs evaluation.
- Optional smoke-test generator: `--generator sshleifer/tiny-gpt2`. This checks plumbing; it is not suitable for meaningful grounded answers.

The generator uses its chat template when available. Prompt length is checked with the tokenizer and reserves room for new tokens. A configurable similarity threshold can abstain before generation, but it is not a calibrated probability or a guarantee that evidence is sufficient.

## Engineering improvements

- One canonical `chunk_id` across ingestion, citations, and evaluation.
- A real document manifest and relevance labels validated against loaded chunks.
- Correct imports and explicit dependencies; optional models are loaded lazily.
- FAISS returns scores then indices, clamps k, and checks the supplied index rather than rebuilding it during validation.
- Tie-aware NumPy/FAISS comparison and persistence checks.
- No redundant overlap-only trailing chunks.
- Source metadata supported by filters.
- No-context gating and tokenizer-aware context budgeting.
- Separate development and test examples, with out-of-scope questions clearly labeled.
- Tests for source alignment, metrics, persistence, empty inputs, caching, and answer-path integration.

The answer-path tests use test doubles and do not claim to validate a language model's factual accuracy.

## Repository layout

```text
model.py                   Reusable pipeline components
scaffold.py                Corpus loading, CLI demo and retrieval evaluation
data/documents/            12 local technical notes
data/manifest.json         Document titles and provenance
data/eval.json             28 labeled questions with dev/test splits
reports/                   Measured retrieval results
tests/                     Unit and integration checks
requirements.txt           Lightweight retrieval dependencies
requirements-models.txt    Optional model and FAISS dependencies
artifacts/                 Generated embeddings/index/config (Git-ignored)
```

## Next development milestones

1. Run and compare dense, BM25, and hybrid retrieval on a larger sourced corpus; review gold labels and add multi-passage questions.
2. Evaluate generated answers and abstention against human-reviewed evidence, including unanswerable and adversarial document examples.
3. Add a backend with document ingestion, indexing status, question answering, and persistent source records.
4. Add a frontend with a document library, chat, and expandable evidence passages.
5. Add source/version-aware cache invalidation, input validation, per-user document isolation, and deployment checks before supporting multiple users.

## Component reference

The original function names remain available for reuse. Some utilities are intentionally optional rather than being chained into every request.
1. `load_text_file`
2. `load_text_directory`
3. `extract_text_from_html`
4. `normalize_text`
5. `make_document`
6. `chunk_fixed_size`
7. `chunk_by_tokens`
8. `chunk_by_sentences`
9. `chunk_with_overlap`
10. `attach_chunk_metadata`
11. `load_embedding_model`
12. `embed_text`
13. `embed_chunks`
14. `l2_normalize`
15. `save_corpus`
16. `cosine_similarity_search`
17. `top_k_indices`
18. `top_k_chunks`
19. `retrieve`
20. `build_faiss_index`
21. `faiss_search`
22. `compare_faiss_to_numpy`
23. `save_faiss_index`
24. `build_prompt_template`
25. `format_context`
26. `truncate_context`
27. `add_system_instruction`
28. `load_generator`
29. `generate_answer`
30. `rag_answer`
31. `track_source_chunk_ids`
32. `append_source_references`
33. `query_rewrite`
34. `hyde_retrieve`
35. `reciprocal_rank_fusion`
36. `bm25_search`
37. `hybrid_search`
38. `rerank_cross_encoder`
39. `maximal_marginal_relevance`
40. `filter_by_metadata`
41. `build_eval_set`
42. `hit_rate_at_k`
43. `recall_at_k`
44. `mean_reciprocal_rank`
45. `faithfulness_score`
46. `relevance_score`
47. `handle_no_context`
48. `deduplicate_chunks`
49. `cache_query_embedding`
50. `update_chat_memory`
51. `rewrite_followup`
