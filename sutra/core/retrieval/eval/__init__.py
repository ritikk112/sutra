"""
Retrieval eval harness (P12).

Every retrieval change from P15 onward is unfalsifiable without this:
checked-in query datasets, recall@k / MRR metrics, and a harness that
runs any retriever and A/Bs two of them.
"""
from sutra.core.retrieval.eval.dataset import EvalCase, load_dataset, load_datasets
from sutra.core.retrieval.eval.harness import EvalReport, QueryEval, compare, run_eval
from sutra.core.retrieval.eval.metrics import first_hit_rank, recall_at_k, reciprocal_rank

__all__ = [
    "EvalCase",
    "EvalReport",
    "QueryEval",
    "compare",
    "first_hit_rank",
    "load_dataset",
    "load_datasets",
    "recall_at_k",
    "reciprocal_rank",
    "run_eval",
]
