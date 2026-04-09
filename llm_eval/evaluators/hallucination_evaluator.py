from transformers import pipeline


class HallucinationEvaluator:
    """
    Uses a local NLI model (DeBERTa-v3-small) to detect hallucinations.
    Downloads once (~85 MB) and runs fully on CPU — no API key needed.
    """

    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,  # CPU
        )

    def score(self, response: str, reference: str) -> dict:
        # Truncate to avoid token limit issues
        ref = reference[:500]
        resp = response[:500]
        result = self.nli(f"{ref} [SEP] {resp}")[0]
        label = result["label"].lower()
        confidence = round(result["score"], 4)
        return {
            "nli_label": label,
            "nli_confidence": confidence,
            "hallucination_flag": label == "contradiction",
        }
