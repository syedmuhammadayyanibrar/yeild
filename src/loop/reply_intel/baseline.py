from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from loop.reply_intel.taxonomy import CLASS_TO_INDEX, INDEX_TO_CLASS
from loop.rng import get_rng
from loop.schema import REPLY_CLASSES, ReplyClass
from loop.simulate.replies import SPLIT_CORE_IDS, compose_reply_text


class BaselineClassifier:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.classes_: list[ReplyClass] = list(REPLY_CLASSES)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raw_probs = self.pipeline.predict_proba(texts)
        model_classes = list(self.pipeline.classes_)
        n_samples = len(texts)
        aligned = np.zeros((n_samples, len(REPLY_CLASSES)), dtype=float)
        for col_idx, cls_name in enumerate(model_classes):
            if cls_name in CLASS_TO_INDEX:
                target_col = CLASS_TO_INDEX[cls_name]
                aligned[:, target_col] = raw_probs[:, col_idx]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums

    def predict(self, texts: list[str]) -> list[ReplyClass]:
        probs = self.predict_proba(texts)
        indices = np.argmax(probs, axis=1)
        return [INDEX_TO_CLASS[int(idx)] for idx in indices]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineClassifier":
        pipeline = joblib.load(path)
        return cls(pipeline)

def generate_train_items_from_cores(seed: int = 42, n_per_cell: int = 40) -> list[dict[str, Any]]:
    rng = get_rng(seed)
    train_cores = SPLIT_CORE_IDS["train"]
    items: list[dict[str, Any]] = []
    from loop.schema import LANGUAGES

    for cls in REPLY_CLASSES:
        for lang in LANGUAGES:
            for i in range(n_per_cell):
                core_id = train_cores[i % len(train_cores)]
                text, _ = compose_reply_text(cls, lang, core_id, rng, apply_perturbations=True)
                items.append(
                    {
                        "text": text,
                        "true_class": cls,
                        "true_language": lang,
                    }
                )
    return items

def train_baseline(items: list[dict[str, Any]] | None = None) -> BaselineClassifier:
    if items is None:
        items = generate_train_items_from_cores()

    texts = [str(it["text"]) for it in items]
    labels = [str(it["true_class"]) for it in items]

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(1, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(texts, labels)
    return BaselineClassifier(pipeline)
