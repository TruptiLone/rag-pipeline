"""Run a reproducible retrieval demo; opt into local generation with --generate."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import model as m

ROOT = Path(__file__).resolve().parent
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_corpus():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    chunks = []
    for document in manifest:
        text = m.normalize_text(m.load_text_file(ROOT / 'data' / document['path']))
        added = m.attach_chunk_metadata(m.chunk_with_overlap(text, 800, 100), document['source'])
        for chunk in added:
            chunk['metadata'].update(title=document['title'], origin=document['origin'])
        chunks.extend(added)
    return chunks


def evaluate(chunks, embed_model=None, embeddings=None, split='test'):
    examples = [e for e in m.build_eval_set() if e['split'] == split and e['answerable']]
    ids = {c['chunk_id'] for c in chunks}
    for example in m.build_eval_set():
        if not set(example['relevant_ids']).issubset(ids):
            raise ValueError('Evaluation references missing chunk IDs; review corpus labels')
    methods = {'bm25': lambda q: m.bm25_search(q, chunks, k=3)}
    if embed_model is not None:
        methods['dense'] = lambda q: [(int(i), float(s)) for i, s in
            enumerate(m.cosine_similarity_search(m.embed_text(embed_model, q), embeddings))]
        methods['hybrid'] = lambda q: m.hybrid_search(q, chunks, embeddings, embed_model, k=3)
    results = {}
    for name, search in methods.items():
        retrieved, gold, details, times = [], [], [], []
        for example in examples:
            start = time.perf_counter()
            hits = sorted(search(m.query_rewrite(example['question'])), key=lambda pair: pair[1], reverse=True)[:3]
            times.append((time.perf_counter() - start) * 1000)
            found = [chunks[i]['chunk_id'] for i, score in hits]
            retrieved.append(found); gold.append(example['relevant_ids'])
            details.append({'question': example['question'], 'retrieved_ids': found, 'relevant_ids': example['relevant_ids']})
        results[name] = {'hit_at_3': m.hit_rate_at_k(retrieved, gold, 3),
                         'recall_at_3': m.recall_at_k(retrieved, gold, 3),
                         'mrr_at_3': m.mean_reciprocal_rank(retrieved, gold),
                         'mean_query_ms': float(np.mean(times)), 'queries': details}
    return {'split': split, 'answerable_questions': len(examples), 'chunks': len(chunks),
            'methods': results, 'generation_evaluated': False,
            'unanswerable_questions_excluded_from_retrieval_average': 4 if split == 'test' else 0,
            'corpus_sha256': hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest(),
            'python': platform.python_version(), 'numpy': np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense', action='store_true', help='Download/load the embedding model and compare dense/hybrid retrieval')
    parser.add_argument('--generate', action='store_true', help='Also download/load an instruction model and generate one answer')
    parser.add_argument('--question', default='Why normalize embeddings before cosine search?')
    parser.add_argument('--generator', default='HuggingFaceTB/SmolLM2-360M-Instruct')
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    args = parser.parse_args()
    chunks = load_corpus()
    embedding_model, embeddings = None, None
    if args.dense or args.generate:
        embedding_model = m.load_embedding_model(EMBED_MODEL)
        embeddings = m.l2_normalize(m.embed_chunks(embedding_model, chunks))
        artifact = ROOT / 'artifacts'; artifact.mkdir(exist_ok=True)
        m.save_corpus(embeddings, chunks, artifact)
        index = m.build_faiss_index(embeddings)
        reloaded = m.save_faiss_index(index, artifact / 'corpus.faiss')
        vector = m.embed_text(embedding_model, args.question)
        if not m.compare_faiss_to_numpy(vector, embeddings, reloaded, 3):
            raise AssertionError('Reloaded FAISS index disagrees with NumPy')
        (artifact / 'config.json').write_text(json.dumps({'embedding_model': EMBED_MODEL, 'chunk_size': 800, 'overlap': 100}, indent=2))
    results = evaluate(chunks, embedding_model, embeddings, args.split)
    reports = ROOT / 'reports'; reports.mkdir(exist_ok=True)
    suffix = 'all' if embedding_model is not None else 'bm25'
    path = reports / f'retrieval_{args.split}_{suffix}.json'
    path.write_text(json.dumps(results, indent=2) + '\n')
    for method, scores in results['methods'].items():
        print(method, {key: value for key, value in scores.items() if key != 'queries'})
    print('Saved', path)
    if args.generate:
        generator, tokenizer = m.load_generator(args.generator)
        answer = m.rag_answer(args.question, chunks, embeddings, embedding_model, generator, tokenizer)
        print(json.dumps(answer, indent=2))
        (reports / 'answer_example.json').write_text(json.dumps(answer, indent=2) + '\n')
    else:
        print('Retrieval-only run: no generated answer or answer-quality claim.')


if __name__ == '__main__':
    main()
