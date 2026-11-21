from rouge_score import rouge_scorer


class RougeEvaluator:
    """ROUGE-1/2/L F-measure against a reference answer.

    Uses the `rouge-score` package directly (the same implementation that
    HuggingFace `evaluate` wraps), so nothing is downloaded at runtime.
    Porter stemming is on, so "contracts" and "contract" count as a match.

    ROUGE rewards word overlap, not correctness: it is only meaningful when
    the response is about as long as the reference, which is why every
    dataset asks the model for a short answer.
    """

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score(self, prediction: str, reference: str) -> dict:
        result = self.scorer.score(reference, prediction)
        return {
            "rouge1": round(result["rouge1"].fmeasure, 4),
            "rouge2": round(result["rouge2"].fmeasure, 4),
            "rougeL": round(result["rougeL"].fmeasure, 4),
        }
