"""Sentence-level grounding check with a local NLI model.

How it works, in plain terms: the response is split into sentences ("claims").
Each claim is checked against a premise — the source text the answer should
be grounded in — by a natural-language-inference (NLI) model, which says
whether the premise *entails* the claim, *contradicts* it, or is *neutral*
(neither supports nor refutes it).

* For summarization, the premise is the source passage being summarized.
* For QA, the premise is the reference answer.

Reported per response:

* ``hallucination_flag`` — True when at least one claim is contradicted by
  the premise. This is the strict, low-false-positive signal.
* ``contradiction_rate`` / ``entailment_rate`` — share of claims in each class.
  ``1 - entailment_rate`` is the share of claims the premise does not back up.
  For summarization that is a useful "unsupported content" signal; for QA it
  mostly reflects extra detail beyond a one-sentence reference, so read it
  with care.
* ``nli_label`` — ``contradiction`` if any claim is contradicted, else
  ``entailment`` if every claim is entailed, else ``neutral``.
* ``nli_confidence`` — the model's probability for that label on the claim
  that decided it.

The NLI model is cross-encoder/nli-deberta-v3-small (~0.1B parameters, runs
on CPU). Its label order is contradiction, entailment, neutral.
"""
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_FALLBACK_LABELS = ["contradiction", "entailment", "neutral"]

MAX_CLAIMS = 20
MIN_CLAIM_WORDS = 3

# A classifier takes (premise, hypothesis) pairs and returns, per pair, a
# mapping of label -> probability. Injectable so tests can run without the
# model download.
Classifier = Callable[[Sequence[Tuple[str, str]]], List[Dict[str, float]]]

_CODE_BLOCK_RE = re.compile(r"```.*?(```|$)", flags=re.DOTALL)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|#{1,6})\s+")


def split_claims(text: str, max_claims: int = MAX_CLAIMS) -> List[str]:
    """Split a response into checkable sentences.

    Drops code blocks and markdown decoration, splits on line breaks and
    sentence-ending punctuation, and skips fragments shorter than three words
    (headings, labels) that an NLI model cannot judge meaningfully.
    """
    text = _CODE_BLOCK_RE.sub(" ", text)
    claims: List[str] = []
    for line in text.splitlines():
        line = _MARKDOWN_PREFIX_RE.sub("", line)
        line = line.replace("**", "").replace("__", "").replace("`", "").strip()
        if not line:
            continue
        for sentence in _SENTENCE_END_RE.split(line):
            sentence = sentence.strip()
            if len(sentence.split()) >= MIN_CLAIM_WORDS:
                claims.append(sentence)
    return claims[:max_claims]


def _load_deberta_classifier() -> Classifier:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if set(id2label.values()) != set(_FALLBACK_LABELS):
        # Some exports ship generic LABEL_0..2 names; use the documented order.
        id2label = dict(enumerate(_FALLBACK_LABELS))

    def classify(pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        if not pairs:
            return []
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        with torch.no_grad():
            enc = tokenizer(
                premises,
                hypotheses,
                padding=True,
                # Long premises (source passages) are cut first so the claim
                # being checked is never truncated.
                truncation="only_first",
                max_length=512,
                return_tensors="pt",
            )
            probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
        return [{id2label[i]: p for i, p in enumerate(row)} for row in probs]

    return classify


class HallucinationEvaluator:
    def __init__(self, classifier: Optional[Classifier] = None):
        self.classify = classifier or _load_deberta_classifier()

    def score(self, response: str, premise: str) -> dict:
        claims = split_claims(response)
        if not claims or not premise.strip():
            return {
                "nli_label": None,
                "nli_confidence": None,
                "entailment_rate": None,
                "contradiction_rate": None,
                "num_claims": len(claims),
                "hallucination_flag": None,
            }

        scores = self.classify([(premise, c) for c in claims])
        labels = [max(s, key=s.get) for s in scores]
        n = len(labels)
        contradicted = [s for s, lab in zip(scores, labels) if lab == "contradiction"]
        entailed = [s for s, lab in zip(scores, labels) if lab == "entailment"]

        if contradicted:
            label = "contradiction"
            confidence = max(s["contradiction"] for s in contradicted)
        elif len(entailed) == n:
            label = "entailment"
            confidence = min(s["entailment"] for s in entailed)
        else:
            label = "neutral"
            neutral = [s["neutral"] for s, lab in zip(scores, labels) if lab == "neutral"]
            confidence = max(neutral)

        return {
            "nli_label": label,
            "nli_confidence": round(confidence, 4),
            "entailment_rate": round(len(entailed) / n, 4),
            "contradiction_rate": round(len(contradicted) / n, 4),
            "num_claims": n,
            "hallucination_flag": bool(contradicted),
        }
