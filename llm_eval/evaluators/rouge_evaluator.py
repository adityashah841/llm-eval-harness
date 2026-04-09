import evaluate as hf_evaluate


class RougeEvaluator:
    def __init__(self):
        self.rouge = hf_evaluate.load("rouge")

    def score(self, prediction: str, reference: str) -> dict:
        result = self.rouge.compute(
            predictions=[prediction],
            references=[reference],
        )
        return {
            "rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4),
        }
