# Real-model baseline validation

This report records the unmodified configured pipeline, run locally on CPU. No prompt, threshold, model, or retrieval settings were tuned after seeing the test answers.

## What ran

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Generator: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- 12 corpus chunks; 12 answerable test questions plus 4 out-of-scope questions.
- Dense answer path: k=3, threshold=0.2, total token cap=1024, output budget=100, greedy decoding.
- Corpus and FAISS index persisted and reloaded; the supplied index agreed with NumPy for all 16 questions.

## Retrieval results

| Method | Hit@3 | Recall@3 | MRR@3 |
|---|---:|---:|---:|
| BM25 | 1.000 | 1.000 | 0.958 |
| Dense | 1.000 | 1.000 | 0.903 |
| Hybrid | 1.000 | 1.000 | 1.000 |

The corpus is deliberately small and synthetic, with one gold chunk per answerable question. Hybrid ranked the gold chunk first on every question in this run; this is not a claim of general superiority. The answer generator still used dense retrieval.

## Answer review

The accompanying qualitative review was performed by the AI assistant against the reference answers and actual supplied passages. It is not independent human review or a calibrated automated accuracy measure.

| Outcome | Count |
|---|---:|
| Fully correct answerable response | 1 / 12 |
| Partially correct / incomplete | 1 / 12 |
| Incorrect | 2 / 12 |
| Unnecessary abstention | 8 / 12 |
| Correct out-of-scope abstention | 4 / 4 |

All 12 answerable prompts included their gold passage. All four out-of-scope questions were rejected before generation because their best retrieval score did not exceed the threshold. Their refusal is not evidence of the generator recognizing its knowledge limits. None of the four substantive answers included the requested bracketed citations. Returned source records remain available, but are not claim-level citations.

## Concrete failures

- Lower MMR lambda: the model answered “relevance more,” contradicting the passage saying diversity.
- Mean reciprocal rank: the model gave a recall-style definition instead of the reciprocal rank of the first relevant result.
- Token-based chunk size: “determined by the tokenizer” omits the controlling token count/budget.

## Timing and reproducibility

Across all 16 requests, including four inexpensive gate refusals, p50 was 0.403s and p95 was 0.669s. These exclude model loading and download time. They are local small-batch measurements, not load-test results.

The retrieval methods ran sequentially without a matched warm-up protocol. Dense timing includes first-query warm-up while hybrid benefits from an already warm encoder, so the recorded means must not be used to claim hybrid is faster.

The existing Anaconda environment failed before model loading due to an OpenSSL-related import conflict (`GEN_EMAIL`). Validation used an isolated Python virtual environment instead. Exact installed packages are captured in `requirements-validation.txt`; model identifiers, generator revision, runtime versions and timings are in `generation_baseline.json`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validation.txt
python validate_models.py
```

The model lock captures the tested Python 3.10/macOS environment and may need adaptation on other platforms. Initial runs download public weights without requiring a paid inference API. Keep caches and artifacts out of Git. The script writes a result after each completed answer; it does not automatically regenerate the qualitative review.

## Next experiment

Retrieval is working on these examples; generation is the immediate bottleneck. Compare a better-capable instruction model and simpler prompts on development questions before selecting a change. Because this test set has now been inspected in detail, use newly reviewed holdout questions for a final claim. Evaluate claim support, unnecessary refusal, and citation correctness separately. Do not tune the threshold on these test results or claim that word-overlap scores establish grounding.

## Per-question audit

| Question | Assessment | Generated answer |
|---|---|---|
| Does using RAG automatically fine-tune the generator? | unnecessary_abstention | I do not know |
| What determines the size of token-based chunks? | partial | Token-based chunks are determined by the tokenizer. |
| Why embed both queries and document chunks? | unnecessary_abstention | I do not know |
| Why handle zero vectors separately during normalization? | unnecessary_abstention | I do not know |
| Is IndexFlatIP exact or approximate? | unnecessary_abstention | I do not know |
| What does the BM25 b parameter control? | correct | The BM25 b parameter controls document-length normalization. |
| Does reciprocal rank fusion combine raw scores or ranks? | unnecessary_abstention | I do not know |
| What does a lower MMR lambda emphasize? | incorrect | A lower lambda emphasizes relevance more. |
| What does mean reciprocal rank measure? | incorrect | mean reciprocal rank measures the fraction of gold-relevant passages retrieved from a query. |
| Does word overlap prove that an answer is factually supported? | unnecessary_abstention | I do not know |
| When should embeddings be rebuilt? | unnecessary_abstention | I do not know |
| Why rewrite a conversational follow-up? | unnecessary_abstention | I do not know |
| What is the weather in Paris today? | correct_abstention | I do not know |
| What is my account password? | correct_abstention | I do not know |
| Who won the latest football championship? | correct_abstention | I do not know |
| What is the population of Tokyo in 2026? | correct_abstention | I do not know |
