import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import model as m
from scaffold import load_corpus, evaluate


class FakeEmbedder:
    def encode(self, text):
        return np.array([1., 0.], dtype=np.float32)


class FakeTokenizer:
    model_max_length = 2048
    def encode(self, text, **kwargs):
        return text.split()


class PipelineTests(unittest.TestCase):
    def test_utf8_and_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'note.txt'
            p.write_bytes('  café\r\nnext\n'.encode())
            self.assertEqual(m.load_text_file(p), '  café\r\nnext\n')

    def test_html_and_normalization(self):
        self.assertEqual(m.extract_text_from_html('<p>A &amp; B</p><script>bad</script><style>bad</style>'), 'A & B')
        self.assertEqual(m.normalize_text('  Ａ\n B\t'), 'A B')

    def test_no_redundant_overlap_tail(self):
        self.assertEqual(m.chunk_with_overlap('abcdefghij', 10, 2), ['abcdefghij'])
        self.assertEqual(m.chunk_with_overlap('abcdefghijkl', 10, 2), ['abcdefghij', 'ijkl'])

    def test_sources_and_metadata(self):
        chunks = m.attach_chunk_metadata(['a', 'b'], 'doc')
        self.assertEqual(m.track_source_chunk_ids(chunks), ['doc::0', 'doc::1'])
        self.assertEqual(m.filter_by_metadata(chunks, {'source': 'doc'}), chunks)
        self.assertEqual(m.append_source_references('answer', chunks), 'answer\nSources: [doc::0, doc::1]')

    def test_eval_ids_exist(self):
        chunks = load_corpus()
        ids = {c['chunk_id'] for c in chunks}
        self.assertEqual(len(chunks), 12)
        for e in m.build_eval_set():
            self.assertTrue(set(e['relevant_ids']).issubset(ids))
        self.assertEqual(evaluate(chunks)['answerable_questions'], 12)

    def test_retrieval_metrics(self):
        retrieved = [['x', 'a'], ['b'], []]
        relevant = [['a', 'c'], ['b'], []]
        self.assertAlmostEqual(m.hit_rate_at_k(retrieved, relevant, 2), 2/3)
        self.assertAlmostEqual(m.recall_at_k(retrieved, relevant, 2), .5)
        self.assertAlmostEqual(m.mean_reciprocal_rank(retrieved, relevant), .5)

    def test_zero_normalization_and_duplicates(self):
        original = np.array([[0., 0.], [3., 4.]])
        out = m.l2_normalize(original)
        np.testing.assert_allclose(out, [[0, 0], [.6, .8]])
        np.testing.assert_array_equal(original, [[0, 0], [3, 4]])
        kept, matrix = m.deduplicate_chunks([], np.zeros((0, 4)))
        self.assertEqual(matrix.shape, (0, 4))

    def test_corpus_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            chunks = m.attach_chunk_metadata(['évidence'], 'doc')
            matrix = np.array([[1., 0.]], dtype=np.float32)
            result = m.save_corpus(matrix, chunks, d)
            np.testing.assert_array_equal(result['embeddings'], matrix)
            self.assertEqual(result['chunks'], chunks)

    def test_abstention_skips_generator(self):
        with patch.object(m, 'generate_answer') as generate:
            result = m.rag_answer('q', [{'text': 'a', 'source': 's'}], np.array([[0., 1.]]),
                                  FakeEmbedder(), object(), FakeTokenizer())
            self.assertEqual(result['answer'], 'I do not know')
            generate.assert_not_called()

    def test_answer_sources_are_supplied_context(self):
        chunks = m.attach_chunk_metadata(['evidence'], 'doc')
        with patch.object(m, 'generate_answer', return_value='answer [1]') as generate:
            result = m.rag_answer('q', chunks, np.array([[1., 0.]]), FakeEmbedder(), object(), FakeTokenizer())
            self.assertEqual(result['sources'], chunks)
            self.assertIn('evidence', generate.call_args.args[2])
            self.assertEqual(result['query'], 'q')

    def test_too_small_context_budget(self):
        with patch.object(m, 'generate_answer') as generate:
            result = m.rag_answer('q', m.attach_chunk_metadata(['evidence'], 's'), np.array([[1., 0.]]),
                                  FakeEmbedder(), object(), FakeTokenizer(), max_context_tokens=5)
            self.assertEqual(result['sources'], [])
            generate.assert_not_called()

    def test_cache_and_memory(self):
        cache = {}
        with patch.object(m, 'embed_text', return_value=np.array([1.])) as embed:
            a = m.cache_query_embedding('q', None, cache)
            b = m.cache_query_embedding('q', None, cache)
            self.assertIs(a, b)
            embed.assert_called_once()
        history = []
        result = m.update_chat_memory(history, 'What is RAG?', 'A retrieval system.')
        self.assertEqual(history, [])
        self.assertEqual(m.rewrite_followup('Why?', result), 'What is RAG? Why?')

    def test_faiss_checks_supplied_index(self):
        try:
            import faiss
        except ImportError:
            self.skipTest('Optional faiss-cpu not installed')
        matrix = np.array([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
        index = m.build_faiss_index(matrix)
        self.assertTrue(m.compare_faiss_to_numpy(matrix[0], matrix, index, 2))
        wrong = m.build_faiss_index(matrix[::-1])
        self.assertFalse(m.compare_faiss_to_numpy(matrix[0], matrix, wrong, 2))
        with tempfile.TemporaryDirectory() as d:
            loaded = m.save_faiss_index(index, Path(d) / 'index.faiss')
            scores, ids = m.faiss_search(loaded, matrix[0], 20)
            self.assertEqual(ids.tolist(), [0, 1, 2])
            self.assertEqual(scores.dtype, np.float32)
            self.assertEqual(ids.dtype, np.int64)


if __name__ == '__main__':
    unittest.main()
