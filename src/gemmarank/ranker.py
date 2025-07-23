import os
import sys
import joblib 
import torch

from collections import defaultdict
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .rankt5_model import register_rankt5_model
from .config import ExperimentConfig


class TFIDFRanker:
    def __init__(self, vectorizer, corpus_matrix, corpus_ids, name: str = None):
        self.name = name if name else "tf-idf ranker"
        self.vectorizer = vectorizer
        self.corpus_matrix = corpus_matrix
        self.corpus_ids = corpus_ids
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        print(f"coprus matrix shape: {self.corpus_matrix.shape}")

    @classmethod
    def from_corpus(cls, passages: dict, config):
        if os.path.exists(config.tfidf_vectorizer_path) and os.path.exists(config.tfidf_corpus_matrix_path):
            vectorizer = joblib.load(config.tfidf_vectorizer_path)
            corpus_matrix = load_npz(config.tfidf_corpus_matrix_path)
        else:
            print("generating TF-IDF vocab and corpus... ")
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=config.tfidf_max_features,
                ngram_range=config.tfidf_ngram_range,
                sublinear_tf=config.tfidf_sublinear_tf,
                max_df=config.tfidf_max_df,
                min_df=config.tfidf_min_df
            )

            vectorizer.fit(passages.values())
            corpus_matrix = vectorizer.transform(passages.values())
            
            print(f"saving vocab to {config.tfidf_vectorizer_path} and corpus to {config.tfidf_corpus_matrix_path}")
            joblib.dump(vectorizer, config.tfidf_vectorizer_path)
            save_npz(config.tfidf_corpus_matrix_path, corpus_matrix)
        
        corpus_ids = list(passages.keys())
        return cls(vectorizer, corpus_matrix, corpus_ids)

    def rank_passages(self, query, passage_ids, passages_dict=None):
        qvec = self.vectorizer.transform([query])

        indices = [self.doc_id_to_idx[pid] for pid in passage_ids]
        pvecs = self.corpus_matrix[indices]
        
        scores = (pvecs * qvec.T).toarray().flatten()
        return {pid: scores[i] for i, pid in enumerate(passage_ids)}


class RankT5Ranker:
    def __init__(self, model_path, device='cuda', name="RankT5 Ranker"):
        self.name = name
        self.device = device
        self.model_path = model_path
        
        register_rankt5_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()

    def rank_passages(self, query, passage_ids, passages_dict):
        texts = [f"Query: {query}\nDocument: {passages_dict[pid]}" for pid in passage_ids]
        
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            scores = self.model.predict(inputs.input_ids, inputs.attention_mask)
        
        return {pid: scores[i].item() for i, pid in enumerate(passage_ids)}


def rank_documents(ranker, queries, candidates, passages):
    ranked = {}
    items = [(qid, text) for qid, text in queries.items() if qid in candidates]

    for i, (qid, qtext) in enumerate(items, 1):
        if i % 100 == 0: print(f"reranking query {i}/{len(items)}")
        
        passage_ids = list(candidates[qid].keys())
        ranked[qid] = ranker.rank_passages(qtext, passage_ids, passages)
    
    return ranked
