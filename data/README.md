# Corpus and evaluation provenance

These 12 short technical notes were authored for this project. They explain RAG components; they were not scraped or copied from a website. `manifest.json` records every document title, source ID and path. Source URLs are null because there is no upstream webpage for these notes.

The 28 evaluation examples include 12 development questions, 12 held-out answerable questions, and 4 out-of-scope questions. Each answerable question points to an existing chunk. These are small, synthetic sanity checks, not an independent real-world benchmark. The test questions share documents with development questions and are not a test of new-domain generalization. Reference answers are manually specified; lexical overlap is not factual verification.

The application uses 800-character chunks with 100-character overlap. Each current document fits one chunk. If documents or chunking change, review and update the relevance labels. Adding a larger corpus with human-reviewed multi-passage questions is the next evaluation milestone.
