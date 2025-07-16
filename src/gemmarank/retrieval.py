import os
from collections import defaultdict

import ir_datasets

from pyserini.search import get_topics
from pyserini.search.lucene import LuceneSearcher
from ranx import Qrels, Run

from gemmarank.config import ExperimentConfig

def load_dataset(ir_dataset_path):
    print(f"loading dataset: {ir_dataset_path}")
    dataset = ir_datasets.load(ir_dataset_path)
    queries = {query.query_id: query.text for query in dataset.queries_iter()}

    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    qrels = Qrels(qrels)

    passages = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

    print(f" loaded {len(queries):,} and {len(passages):,} passages")
    return queries, qrels, passages


class BM25Retriever:
    def __init__(self, config: ExperimentConfig, name: str = 'BM25 Retrieval'):
        self.config = config
        self.searcher = None
        self.name = name
        self.searcher = LuceneSearcher.from_prebuilt_index(self.config.pyserini_index_name)

    def _generate_run(self):
        print("generating BM25 baseline") 
        topics = get_topics(self.config.pyserini_topic_name)
        results = defaultdict(dict)
        
        for q_id, query_data in topics.items():
            query = query_data['title']
            hits = self.searcher.search(query, k=self.config.bm25_k_hits)
            for hit in hits:
                results[q_id][hit.docid] = hit.score
        
        bm25_run_obj = Run(results)
        bm25_run_obj.save(self.config.bm25_run_path, kind="trec")

        print(f"saved baseline to {self.config.bm25_run_path}")

    def retrieve_candidates(self, queries: dict):
        """returns a dictionary of initial candidates {query_id: {doc_id: score}}"""
        if not os.path.exists(self.config.bm25_run_path):
            self._generate_run()
        
        print(f"Loading initial retrieval results from {self.config.bm25_run_path}...", end=' ')
        candidates = defaultdict(dict) 
        with open(self.config.bm25_run_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                q_id = parts[0]
                doc_id = parts[2]
                score = float(parts[4])
                if len(candidates[q_id]) < self.config.bm25_load_top_k:
                    candidates[q_id][doc_id] = score

        print(f"retrieved results for {len(candidates)} queries")
        return {q_id: docs for q_id, docs in candidates.items() if q_id in queries}
