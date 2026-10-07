# Evidence passages used in baseline validation

These are the exact chunk texts recorded in the baseline generation report, not newly retrieved passages. The source-document links are pinned to the baseline Git commit so later edits do not change the evidence being referenced.

<a id="passage-1"></a>
## bm25::0

**Document:** [BM25 lexical retrieval](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/bm25.txt#L1)

> BM25 is a lexical ranking function based on query-term matches, term frequency, inverse document frequency, and document length normalization. The k1 parameter controls term-frequency saturation. The b parameter controls document-length normalization. Lexical retrieval is useful for exact names and technical identifiers.

<a id="passage-2"></a>
## chunking::0

**Document:** [Document chunking](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/chunking.txt#L1)

> Chunking divides a document into smaller searchable passages. Token-based chunking controls size using a tokenizer. Overlapping chunks repeat part of the previous passage to preserve boundary context. Too little context can hide an answer, while oversized chunks can include distracting material and exceed the generator input budget.

<a id="passage-3"></a>
## cosine::0

**Document:** [Cosine similarity and normalization](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/cosine.txt#L1)

> Cosine similarity is the dot product of two vectors divided by the product of their L2 norms. After L2 normalization, each nonzero vector has unit length, so its inner product with another normalized vector equals cosine similarity. Zero vectors must be handled separately to avoid division by zero.

<a id="passage-4"></a>
## embeddings::0

**Document:** [Text embeddings](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/embeddings.txt#L1)

> An embedding is a numerical vector representation of text. An embedding model encodes both document chunks and queries into a shared vector space. Related meanings can have similar vectors even when the wording differs. Semantic similarity is not proof that a passage answers a question.

<a id="passage-5"></a>
## faiss::0

**Document:** [FAISS vector search](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/faiss.txt#L1)

> FAISS supports nearest-neighbor search over dense vectors. IndexFlatIP stores vectors and performs exact inner-product search. Normalize corpus vectors and queries when using inner products as cosine similarities. Search returns score arrays followed by index arrays. Equally scoring vectors can have a different ordering across search backends.

<a id="passage-6"></a>
## grounding::0

**Document:** [Grounding and abstention](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/grounding.txt#L1)

> A grounded answer is supported by retrieved evidence. When no useful evidence is available, a RAG assistant should abstain rather than invent an answer. A similarity threshold is only a heuristic and should be validated. Token overlap cannot establish factual support: an answer can repeat context words while contradicting the context.

<a id="passage-7"></a>
## hybrid::0

**Document:** [Hybrid retrieval](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/hybrid.txt#L1)

> Hybrid retrieval combines dense semantic retrieval with lexical retrieval such as BM25. One approach scales both score vectors and computes a weighted sum. Another approach, reciprocal rank fusion, combines ranked lists using reciprocal rank contributions. Mixing retrieval methods can help some queries, but improvements must be measured.

<a id="passage-8"></a>
## memory::0

**Document:** [Conversation memory and caching](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/memory.txt#L1)

> Conversation memory stores user and assistant turns. A follow-up question may need the previous user question to become self-contained. A query embedding cache avoids re-encoding repeated queries. Cache keys must account for the embedding model when multiple models share a cache. Memory and caching do not guarantee answer correctness.

<a id="passage-9"></a>
## metrics::0

**Document:** [Retrieval evaluation](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/metrics.txt#L1)

> Hit rate at k is the fraction of questions with at least one relevant result in their top k. Recall at k measures the fraction of gold-relevant passages retrieved. Mean reciprocal rank averages the reciprocal position of the first relevant result. Evaluation labels must refer to existing corpus chunk IDs. Use separate development and test questions.

<a id="passage-10"></a>
## persistence::0

**Document:** [Corpus persistence and source metadata](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/persistence.txt#L1)

> Persist chunk metadata together with its embedding matrix so vector rows remain aligned with chunk records. A stable chunk_id identifies a passage, and source identifies the document it came from. Record the embedding model and chunking configuration with an index. Rebuild embeddings when the embedding model changes.

<a id="passage-11"></a>
## rag::0

**Document:** [Retrieval-augmented generation](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/rag.txt#L1)

> Retrieval-augmented generation (RAG) combines a retriever with a generator. The retriever finds passages from a document collection. The generator receives those passages as context together with a question. RAG does not by itself train or fine-tune the generator. Answers should be checked against their supporting passages.

<a id="passage-12"></a>
## reranking::0

**Document:** [Reranking and diversity](https://github.com/TruptiLone/technical-docs-assistant-rag-pipeline/blob/7485e6459568615eb420a8799968cc40d01e2d08/data/documents/reranking.txt#L1)

> A cross-encoder reranker jointly scores a query and a candidate passage. It is usually applied to a shortlist from a cheaper retriever. Maximal marginal relevance (MMR) balances query relevance against similarity to already selected passages. A higher lambda weights relevance more; a lower lambda weights diversity more.

