from ranx import Qrels
from gemmarank.config import ExperimentConfig
from gemmarank.retrieval import load_dataset, BM25Retriever
from gemmarank.ranker import TFIDFRanker, rank_documents, RankT5Ranker
from gemmarank.eval import score_results


def sample_dataset(queries, qrels, n_queries):
    if n_queries <= 0:
        return queries, qrels
    
    sampled_ids = list(queries.keys())[:n_queries]
    sampled_queries = {qid: queries[qid] for qid in sampled_ids}
    sampled_qrels_dict = {qid: docs for qid, docs in qrels.qrels.items() if qid in sampled_ids}
    sampled_qrels = Qrels(sampled_qrels_dict)
    
    return sampled_queries, sampled_qrels


def main():
    config = ExperimentConfig()

    queries, qrels, passages = load_dataset(config.ir_dataset_name)
    queries, qrels = sample_dataset(queries, qrels, config.num_queries_to_process)

    retriever = BM25Retriever(config, name="Initial Retrieval (BM25 Baseline)")
    candidate_results = retriever.retrieve_candidates(queries) 
    print(f"loaded {len(candidate_results)} candidates")

    retrieval_metrics = score_results(qrels, candidate_results, config, run_name=retriever.name)

    print("ranking candidates")
    #ranker = TFIDFRanker.from_corpus(passages, config)
    #ranked_results = rank_documents(ranker, queries, candidate_results) 
    
    print("ranking candidates with neural ranker")
    from gemmarank.ranker import RankT5Ranker  # Add this import
    ranker = RankT5Ranker(config.rankt5_model_path, name="RankT5 Neural Ranker")
    ranked_results = rank_documents(ranker, queries, candidate_results, passages)

    ranked_metrics = score_results(qrels, ranked_results, config, run_name=ranker.name, save_path=config.ranker_run_path)


    print(f"\n{retrieval_metrics['run_name']}:")
    for metric, score in retrieval_metrics.items():
        if metric == 'run_name': continue
        print(f"- {metric.upper()}: {score:.4f}")

    print(f"\n{ranked_metrics['run_name']}:")
    for metric, score in ranked_metrics.items():
        if metric == 'run_name': continue
        print(f"- {metric.upper()}: {score:.4f}")


if __name__ == "__main__":
    main()
