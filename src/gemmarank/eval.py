from ranx import Qrels, Run, evaluate

from gemmarank.config import ExperimentConfig

def score_results(qrels: Qrels, run_info: dict, config: ExperimentConfig, run_name: str, save_path: str = None):
    print(f"\nevaluating {run_name}...")
    run = Run(run_info)
    results = evaluate(qrels, run, config.evaluation_metrics, make_comparable=True)

    print(f"\n{run_name} results on {len(run_info)} queries:")
    for metric, score in results.items():
        print(f"{metric.upper()}: {score:.4f}")

    if save_path:
        run.save(save_path, kind="trec")
        print(f"Saved {run_name} run to {save_path}")

    results['run_name'] = run_name    
    return results
