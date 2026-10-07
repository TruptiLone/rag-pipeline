# Technical Knowledge Assistant — RAG Pipeline

A modular Python application for searching technical documents and generating answers with traceable source passages. The project implements document ingestion, chunking, lexical and semantic retrieval, optional local language-model generation, and reproducible retrieval evaluation.

**Verified:** 13 passing tests · 12-document corpus · 28 labeled evaluation questions · BM25 MRR@3 of 0.958 on 12 answerable test questions.

The repository emphasizes reliable integration: consistent source identifiers, validated evaluation labels, persisted vector indexes, and explicit context budgets. Its current interface is a command-line application. Real-model validation now covers dense/hybrid retrieval and 16 generated-answer requests. Retrieval succeeds on the small corpus, while the configured generator shows significant answer-quality limitations; see the [baseline validation](reports/VALIDATION.md).

[Quick start](#run-locally) · [Workflow](#workflow) · [Results](#recorded-results) · [Engineering](#engineering-improvements) · [Roadmap](#next-development-milestones)

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

The full evaluation file contains 12 development questions, 12 answerable test questions, and 4 out-of-scope test questions. Out-of-scope questions are excluded from the retrieval averages and evaluated separately in the real-model baseline. **Real-model retrieval and generation have now been run separately; see the [baseline validation report](reports/VALIDATION.md) for results and failure examples.**

The functions named `faithfulness_score` and `relevance_score` are lexical overlap heuristics. They do not detect factual contradictions, validate citations, or establish semantic correctness.

## Real-model validation

Both configured Hugging Face models were run locally. On the 12 answerable test questions, all three retrievers achieved hit@3 and recall@3 of 1.000; MRR@3 was **0.958 for BM25, 0.903 for dense retrieval, and 1.000 for hybrid retrieval**.

Good retrieval did not guarantee good answers. An AI-assisted qualitative review of SmolLM2's responses found **1 fully correct, 1 partial, 2 incorrect, and 8 unnecessary refusals** out of 12 answerable questions, even though the supporting passage was in every prompt. The retrieval gate rejected all four out-of-scope questions before generation. This small synthetic baseline identifies generation as the next area to improve; it is not a production accuracy claim.

- [Validation findings and per-question audit](reports/VALIDATION.md)
- [All answers, supplied passages, timings, and runtime configuration](reports/generation_baseline.json)
- [Qualitative review rubric and annotations](reports/generation_review.json)
- [BM25/dense/hybrid retrieval comparison](reports/retrieval_test_all.json)

Reproduce with `python validate_models.py` after installing the model dependencies. The exact tested environment is recorded in `requirements-validation.txt`. The generator and pipeline were not tuned during this baseline run.

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
- Default generator: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), a compact instruction model. Baseline evaluation found frequent unnecessary refusals and two incorrect answers; it is not yet a validated choice for the final UI.
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

## Technical background and design rationale

### Pipeline stages and responsibilities

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
| Groundedness | Review whether each substantive claim is supported by the supplied passages | Whether generation invents or contradicts evidence | Baseline answers reviewed qualitatively; word overlap is only a heuristic |
| Answer correctness and completeness | Compare against reviewed reference answers and required facts | Whether the response answers the question accurately and sufficiently | Baseline qualitatively reviewed; no automated semantic assessor |
| Citation correctness | Check whether each cited passage supports its associated claim | Whether references provide evidence rather than decoration | Source records returned; substantive baseline answers omitted citation markers |
| Abstention behavior | Measure refusal on unanswerable questions and unnecessary refusal on answerable ones separately | Whether the assistant knows when it lacks evidence | Baseline: 4/4 out-of-scope queries gated; 8/12 answerable queries unnecessarily refused |
| Latency and cost | Measure retrieval, time to first token, total response time, token use, and compute cost | Whether the system is usable under realistic load | Local batch timing recorded; no matched retrieval warm-up or production load/cost benchmark |

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

### Engineering challenges and tradeoffs

| Challenge | Engineering response | Evidence or limitation in this project |
|---|---|---|
| Broken provenance across stages | Standardize identifiers and check labels against the corpus | Fixed `id` versus `chunk_id`; labels now resolve to actual passages |
| Misleading evaluation | Align questions with source content and separate development/test examples | Replaced unrelated demo questions; synthetic scope is explicitly disclosed |
| Backend disagreement | Validate scores and results after index persistence; account for equal-score ties | Tests check the supplied FAISS index and detect an incorrectly ordered index |
| Missing context versus model failure | Inspect retrieval first, then the prompt, then generated claims | Separate retrieval reporting and mocked generation-path tests |
| Prompt budget overflow | Count tokens and reserve space for output before generation | The answer path admits whole passages only when they fit |
| Stale documents and embeddings | Version the corpus, configuration, and model; invalidate affected artifacts | Corpus checksum and model/chunking config exist; automatic refresh is still future work |
| Exact terms versus paraphrases | Compare lexical, dense, and hybrid retrieval under the same labels | BM25, dense, and hybrid measured on the small synthetic corpus |
| Cost and latency | Reuse models, cache embeddings, shortlist before expensive reranking, and profile each stage | Cache helper exists; no production load benchmark yet |

In a multi-user system, document permissions must be enforced before evidence enters the prompt. Retrieved text is untrusted data and can contain prompt-injection instructions. A “use only the context” prompt is not a security boundary. Source validation, retrieval-time authorization, safe rendering, and adversarial tests are additional requirements, not features this local prototype already provides. [OWASP RAG security guidance](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html).

### Where RAG is used and how it is evolving

**Research checked October 6, 2026.** RAG is an established application architecture, while methods for choosing and managing evidence continue to evolve. Common use cases include technical documentation assistants, internal knowledge search, product-support answers, and research tools that need attributable source passages. Enterprise guidance now addresses permission-aware retrieval and operational security rather than treating retrieval as just a vector lookup. [OWASP RAG security guidance](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html).

Two relevant directions are:

- **Context-aware indexing and hybrid reranking:** preserve document context around a chunk, combine semantic and lexical retrieval, and rerank candidates. Anthropic's contextual retrieval work illustrates this approach; its benchmark results should not be assumed to transfer to this corpus. [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).
- **Agentic retrieval:** plan multiple searches for a complex question and combine their evidence. Current Azure documentation describes multi-query planning and reranking, with some capabilities still marked preview. Extra planning also introduces latency and evaluation complexity. This project uses a fixed retrieval flow, not an autonomous agent. [Azure agentic retrieval](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview).

A larger context window does not remove the need to decide what information is current, relevant, and authorized. RAG and fine-tuning also solve different problems: retrieval supplies external evidence at query time; fine-tuning changes model behavior through training. They can be combined. For this small corpus, a whole-corpus prompt is a useful future baseline to test whether retrieval adds value at all.


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

## Directory reference: where everything lives

The repository separates reusable code, source documents, evaluation labels, automated checks, and measured results.

```text
rag-pipeline/
├── README.md
├── model.py
├── scaffold.py
├── requirements.txt
├── requirements-models.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── manifest.json
│   ├── eval.json
│   └── documents/
│       ├── rag.txt
│       ├── chunking.txt
│       ├── embeddings.txt
│       ├── cosine.txt
│       ├── faiss.txt
│       ├── bm25.txt
│       ├── hybrid.txt
│       ├── reranking.txt
│       ├── metrics.txt
│       ├── grounding.txt
│       ├── persistence.txt
│       └── memory.txt
├── tests/
│   └── test_pipeline.py
├── reports/
│   └── retrieval_test_bm25.json
├── docs/
│   └── index.html
└── artifacts/                  # Created by a model-backed run; Git-ignored
```

### Application code and configuration

| File | Responsibility |
|---|---|
| `README.md` | Project purpose, workflow, measured results, limitations, and setup instructions |
| `model.py` | Reusable functions for ingestion, chunking, embeddings, retrieval, generation, and evaluation; despite its name, this is more than a model definition |
| `validate_models.py` | Reproducible real-model baseline batch; saves every test answer, source passage, and timing |
| `requirements-validation.txt` | Exact dependencies from the isolated validation environment |
| `scaffold.py` | Command-line entry point that connects the helpers, loads the corpus, runs evaluation, and optionally generates an answer |
| `requirements.txt` | NumPy dependency for the lightweight retrieval workflow |
| `requirements-models.txt` | Additional dependencies for embeddings, FAISS, and local generation, including PyTorch and Transformers |
| `.gitignore` | Excludes generated artifacts, caches, virtual environments, and Python bytecode from version control |
| `docs/index.html` | Pre-existing static project page; separate from the Python pipeline and not a live question-answering frontend. The root README describes the updated implementation. |

To follow execution, start with `main()` in `scaffold.py`, then inspect the functions it calls in `model.py`.

### `data/`: source knowledge and evaluation labels

The source documents are the information the assistant searches. Evaluation questions are the examples used to assess that search; they are not added to the searchable corpus.

| Path | Contents and purpose |
|---|---|
| `data/documents/` | Twelve project-authored UTF-8 notes about RAG concepts; no webpages are scraped for this demo |
| `data/manifest.json` | Catalog of document titles, relative file paths, source IDs, and provenance; used to locate and identify documents |
| `data/eval.json` | Questions, reference answers, relevant chunk IDs, development/test splits, and answerability labels |
| `data/README.md` | Corpus provenance and limitations of the synthetic evaluation set |

For example, `data/documents/cosine.txt` has source ID `cosine`. Its first chunk is `cosine::0`, and evaluation questions can name that ID as gold-relevant evidence. This relationship lets the evaluator compare retrieved passages against known supporting passages. Changing documents or chunking can change this relationship, so the labels must be reviewed after such changes.

### `tests/`: software correctness

`tests/test_pipeline.py` contains 13 checks covering text preservation, HTML extraction, chunk boundaries, source IDs, evaluation-label validity, metric calculations, normalization, persistence, caching, conversation memory, and the answer path. The FAISS check runs when its optional dependency is installed; otherwise it is skipped.

```bash
python -m unittest discover -s tests -v
```

Tests check that the code behaves as intended. Retrieval evaluation measures whether it finds useful evidence. Answer evaluation checks whether generated responses follow that evidence. Passing software tests does not establish answer quality; the generation-path tests use test doubles.

### `reports/`: measured results

`reports/retrieval_test_bm25.json` is the committed BM25 experiment report. It records aggregate metrics, retrieved and relevant IDs per question, mean query time, a corpus checksum, and Python/NumPy versions.

The CLI writes files according to the selected run:

- `retrieval_test_bm25.json` or `retrieval_dev_bm25.json`: lexical-only evaluation.
- `retrieval_test_all.json` or `retrieval_dev_all.json`: BM25, dense, and hybrid comparisons when embeddings are enabled.
- `answer_example.json`: one generated answer with its query and source records when `--generate` is used.

Only reports actually generated and inspected should be used to support project claims. These filenames describe possible outputs; they do not imply that every experiment has already run.

### `artifacts/`: reusable generated data

A model-backed run creates this folder locally:

```text
artifacts/
├── embeddings.npy    # Chunk vectors, one vector per row
├── chunks.json       # Corresponding passage text and metadata
├── corpus.faiss      # Persisted vector-search index
└── config.json       # Embedding model identifier and chunking settings
```

Row `i` in the embedding matrix must correspond to chunk `i` in the metadata and index. Misalignment can return a mathematically similar vector but display the wrong passage. The tests check persistence, and the model-backed CLI validates the reloaded FAISS index against NumPy.

Artifacts are Git-ignored because they can be regenerated. Downloaded pretrained model weights are normally stored separately in the Hugging Face cache. Persisting an index is implemented; the current CLI still rebuilds it on each model-backed run rather than providing a persistent serving process.

### How a run connects the folders

Running `python scaffold.py`:

1. Reads `data/manifest.json` to locate the source documents.
2. Loads `data/documents/`, then normalizes, chunks, and attaches metadata using `model.py`.
3. Reads the selected answerable questions and gold IDs from `data/eval.json`.
4. Runs BM25 retrieval and calculates hit rate, recall, and MRR.
5. Saves the experiment in `reports/`.

Adding `--dense` loads the embedding model, builds and saves `artifacts/`, checks the FAISS round trip, and evaluates dense/hybrid retrieval as well. Adding `--generate` enables those steps and also loads the generator to answer one question. No model training occurs in these commands.
