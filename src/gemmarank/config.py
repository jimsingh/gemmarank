from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    tfidf_vectorizer_path: str = "tfidf_vectorizer.joblib"
    tfidf_corpus_matrix_path: str = "tfidf_corpus_matrix.npz"
    bm25_run_path: str = "bm25_run.txt"
    ranker_run_path: str = "tfidf_rerank_run.txt"

    ir_dataset_name: str = "msmarco-passage/dev/small"
    pyserini_topic_name: str = "msmarco-passage-dev-subset"
    pyserini_index_name: str = "msmarco-v1-passage"
    bm25_k_hits: int = 1000 
    bm25_load_top_k: int = 100

    # TF-IDF generation (fits into 16GB)
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 1)
    tfidf_sublinear_tf: bool = True
    tfidf_max_df: float = 0.95
    tfidf_min_df: int = 5

    # eval config
    evaluation_metrics: list = field(default_factory=lambda: ["mrr@10", "ndcg@5", "ndcg@10", "map"])
    
    num_queries_to_process: int = 100 # for testing 
