"""Run the configured real models without tuning; preserve all test answers."""
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
import model as m
from scaffold import ROOT, EMBED_MODEL, load_corpus, evaluate

GENERATOR = 'HuggingFaceTB/SmolLM2-360M-Instruct'


def main():
    torch.set_num_threads(4)
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    print('Loading embedding model...', flush=True)
    start = time.perf_counter()
    embedder = m.load_embedding_model(EMBED_MODEL)
    chunks = load_corpus()
    embeddings = m.l2_normalize(m.embed_chunks(embedder, chunks))
    embedding_setup_seconds = time.perf_counter() - start
    retrieval = evaluate(chunks, embedder, embeddings, split='test')
    (reports / 'retrieval_test_all.json').write_text(json.dumps(retrieval, indent=2) + '\n')
    index = m.build_faiss_index(embeddings)
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    m.save_corpus(embeddings, chunks, artifacts)
    restored = m.save_faiss_index(index, artifacts / 'corpus.faiss')
    examples = [e for e in m.build_eval_set() if e['split'] == 'test']
    for item in examples:
        assert m.compare_faiss_to_numpy(m.embed_text(embedder, item['question']), embeddings, restored, 3)
    print('Retrieval and FAISS checks complete. Loading generator...', flush=True)
    start = time.perf_counter()
    generator, tokenizer = m.load_generator(GENERATOR)
    generator_setup_seconds = time.perf_counter() - start
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'embedding_model': EMBED_MODEL, 'generator_model': GENERATOR,
        'generator_revision': getattr(generator.config, '_commit_hash', None),
        'device': str(generator.device), 'torch_threads': 4,
        'config': {'k': 3, 'threshold': 0.2, 'max_context_tokens': 1024, 'max_new_tokens': 100},
        'setup_seconds': {'embedding': embedding_setup_seconds, 'generator': generator_setup_seconds},
        'versions': {p: version(p) for p in ['numpy','torch','transformers','sentence-transformers','faiss-cpu']},
        'python': platform.python_version(), 'faiss_roundtrip_verified': True,
        'evaluation_note': 'Unmodified baseline on small synthetic test set. Lexical overlap is not factual verification. Latency excludes downloads and model loading. No tuning performed.',
        'results': [],
    }
    for number, item in enumerate(examples, 1):
        start = time.perf_counter()
        result = m.rag_answer(item['question'], chunks, embeddings, embedder, generator, tokenizer)
        elapsed = time.perf_counter() - start
        retrieved = m.retrieve(item['question'], embedder, embeddings, chunks, 3)
        report['results'].append({
            'question': item['question'], 'reference_answer': item['answer'],
            'answerable': item['answerable'], 'gold_ids': item['relevant_ids'],
            'answer': result['answer'], 'supplied_sources': result['sources'],
            'retrieved': [{'chunk_id': c['chunk_id'], 'score': score} for c, score in retrieved],
            'elapsed_seconds': elapsed,
            'exact_abstention': result['answer'].strip() == 'I do not know',
            'lexical_faithfulness': m.faithfulness_score(result['answer'], result['sources']),
            'lexical_relevance': m.relevance_score(result['answer'], item['question']),
        })
        (reports / 'generation_baseline.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"[{number}/{len(examples)}] {elapsed:.2f}s {item['question']}\n{result['answer']}", flush=True)
    durations = [r['elapsed_seconds'] for r in report['results']]
    report['latency_seconds'] = {'mean': float(np.mean(durations)), 'p50': float(np.percentile(durations,50)), 'p95': float(np.percentile(durations,95))}
    (reports / 'generation_baseline.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Saved baseline reports.', flush=True)


if __name__ == '__main__':
    main()
