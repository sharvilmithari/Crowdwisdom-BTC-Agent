"""
Sentiment Analysis Engine.
- FinBERT (ProsusAI/finbert) sentiment model implementation where hardware/environment permits.
- Clear, distinct, labelled Lexicon/VADER baseline fallback when FinBERT is unavailable.
- Computes positive, negative, and neutral probabilities.
- Formula: sentiment_score = positive_probability - negative_probability.
- Never mislabels fallback results as FinBERT.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import re
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("sentiment_engine")


class BaseSentimentAnalyzer(ABC):
    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a batch of texts and returns:
        - positive: float [0, 1]
        - negative: float [0, 1]
        - neutral: float [0, 1]
        - sentiment_score: float [-1, 1] (positive - negative)
        - sentiment_label: str ('positive' | 'negative' | 'neutral')
        - model_name: str
        - is_fallback: bool
        """
        pass


class LexiconFallbackAnalyzer(BaseSentimentAnalyzer):
    """Rule & financial lexicon based analyzer.
    Used for instant fallback, testing, and resource-constrained environments.
    Strictly labelled as 'lexicon_fallback' - never disguised as FinBERT.
    """

    BULLISH_WORDS = {
        "bullish", "moon", "hodl", "breakout", "surge", "rally", "buy", "gain",
        "accumulate", "pump", "support", "ath", "long", "rocket", "profit", "uptrend"
    }

    BEARISH_WORDS = {
        "bearish", "dump", "crash", "weak", "drop", "sell", "retest", "dip",
        "fall", "short", "fud", "loss", "bagholder", "resistance", "downtrend"
    }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
            pos_matches = len(words.intersection(self.BULLISH_WORDS))
            neg_matches = len(words.intersection(self.BEARISH_WORDS))
            total = pos_matches + neg_matches

            if total == 0:
                pos_prob = 0.1
                neg_prob = 0.1
                neu_prob = 0.8
            else:
                pos_prob = round(0.1 + 0.8 * (pos_matches / total), 4)
                neg_prob = round(0.1 + 0.8 * (neg_matches / total), 4)
                neu_prob = max(0.0, round(1.0 - (pos_prob + neg_prob), 4))
                # Normalize to sum to 1.0
                denom = pos_prob + neg_prob + neu_prob
                pos_prob = round(pos_prob / denom, 4)
                neg_prob = round(neg_prob / denom, 4)
                neu_prob = round(1.0 - pos_prob - neg_prob, 4)

            score = round(pos_prob - neg_prob, 4)
            if score > 0.05:
                label = "positive"
            elif score < -0.05:
                label = "negative"
            else:
                label = "neutral"

            results.append({
                "positive": pos_prob,
                "negative": neg_prob,
                "neutral": neu_prob,
                "sentiment_score": score,
                "sentiment_label": label,
                "model_name": "lexicon_fallback_v1",
                "is_fallback": True
            })
        return results


class FinBERTAnalyzer(BaseSentimentAnalyzer):
    """ProsusAI/finbert transformer model.
    Falls back gracefully to LexiconFallbackAnalyzer if transformers / torch are not present.
    """

    def __init__(self):
        self._pipeline = None
        self._init_error = None
        self._fallback = LexiconFallbackAnalyzer()

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            model_id = "ProsusAI/finbert"
            logger.info("Initializing FinBERT model...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self._pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
            logger.info("FinBERT initialized successfully.")
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"FinBERT unavailable ({e}). Defaulting to Lexicon fallback.")

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self._pipeline is None:
            return self._fallback.analyze_batch(texts)

        try:
            raw_scores = self._pipeline(texts, truncation=True, max_length=128)
            results = []
            for item in raw_scores:
                score_dict = {entry["label"].lower(): entry["score"] for entry in item}
                pos = score_dict.get("positive", 0.0)
                neg = score_dict.get("negative", 0.0)
                neu = score_dict.get("neutral", 0.0)
                score = round(pos - neg, 4)
                label = max(score_dict, key=score_dict.get)
                results.append({
                    "positive": round(pos, 4),
                    "negative": round(neg, 4),
                    "neutral": round(neu, 4),
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "model_name": "ProsusAI/finbert",
                    "is_fallback": False
                })
            return results
        except Exception as e:
            logger.error(f"FinBERT batch inference error: {e}. Falling back to lexicon.")
            return self._fallback.analyze_batch(texts)


def get_sentiment_analyzer(prefer_finbert: bool = False) -> BaseSentimentAnalyzer:
    """Factory to retrieve sentiment analyzer."""
    if prefer_finbert:
        return FinBERTAnalyzer()
    return LexiconFallbackAnalyzer()

