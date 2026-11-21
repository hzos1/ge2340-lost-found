"""Rank live lost or found records. Training photos are not used here."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from src.campus import location_closeness, location_distance, time_closeness, time_difference
from src.scoring import clip_fusion, linear_probability

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


class Model:
    def __init__(self, text, weights, coefficients, bias, metrics):
        self.text = text
        self.weights = weights
        self.coefficients = coefficients
        self.bias = bias
        self.metrics = metrics


def load_model() -> Model:
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(ART / "embeddings.npz")
    text = {}
    for key, vector in zip(arrays["text_keys"].tolist(), arrays["text_vecs"]):
        colour, category = str(key).split("|", 1)
        text[(colour, category)] = np.asarray(vector, dtype=np.float32)
    weights_json = metrics["fusion_weights"]
    weights = (weights_json["visual"], weights_json["location"], weights_json["time"])
    coefficients = np.array(
        [
            metrics["linear_model"]["coefficients"]["category_match"],
            metrics["linear_model"]["coefficients"]["colour_match"],
            metrics["linear_model"]["coefficients"]["location_distance"],
            metrics["linear_model"]["coefficients"]["time_difference"],
        ],
        dtype=np.float32,
    )
    return Model(
        text=text,
        weights=weights,
        coefficients=coefficients,
        bias=float(metrics["linear_model"]["bias"]),
        metrics=metrics,
    )


def _visual_row(model: Model, record: dict, clip_by_id: dict[str, np.ndarray]) -> np.ndarray:
    vector = clip_by_id.get(record["id"])
    if vector is not None:
        return np.asarray(vector, dtype=np.float32)
    colour = record.get("colour")
    category = record.get("category")
    if (colour, category) in model.text:
        return model.text[(colour, category)]
    return np.zeros(next(iter(model.text.values())).shape, dtype=np.float32)


def _top(gallery, scores, extra, limit=5) -> list[dict]:
    if not gallery:
        return []
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    rows = []
    for rank, index in enumerate(order[:limit], start=1):
        index = int(index)
        record = gallery[index]
        row = {
            "rank": rank,
            "id": record["id"],
            "kind": record.get("kind"),
            "score": float(scores[index]),
            "category": record["category"],
            "category_label": record["category_label"],
            "colour": record["colour"],
            "location": record["location"],
            "x": record["x"],
            "y": record["y"],
            "has_photo": bool(record.get("has_photo")),
            "note": record.get("note") or "",
            "pred_category": record.get("pred_category", record["category"]),
            "pred_category_label": record.get("pred_category_label", record["category_label"]),
            "pred_colour": record.get("pred_colour", record["colour"]),
        }
        if record.get("kind") == "lost" or "lost_time" in record:
            row["lost_time"] = record.get("lost_time")
        if record.get("kind") == "found" or "found_time" in record:
            row["found_time"] = record.get("found_time")
        for key, values in extra.items():
            row[key] = float(values[index])
        rows.append(row)
    return rows


_encoder_lock = threading.Lock()


def embed_upload(path: Path) -> np.ndarray:
    from src.vision import ClipEncoder

    with _encoder_lock:
        encoder = embed_upload.encoder
        if encoder is None:
            encoder = ClipEncoder()
            embed_upload.encoder = encoder
        return encoder.encode_images([path])[0]


embed_upload.encoder = None


def _empty(visual_source: str, against: str) -> dict:
    weights = (0.45, 0.25, 0.30)
    return {
        "visual_source": visual_source,
        "against": against,
        "message": "No records to match yet. Upload a found item first." if against == "found" else "No lost reports to match yet.",
        "weights": {"visual": weights[0], "location": weights[1], "time": weights[2]},
        "clip": [],
        "linear": [],
    }


def search_found(
    model: Model,
    gallery: list[dict],
    clip_by_id: dict[str, np.ndarray],
    *,
    category: str,
    colour: str,
    location: str,
    lost_time: str,
    image_vector=None,
) -> dict:
    """Match a lost-item query against found records."""
    if (colour, category) not in model.text:
        raise ValueError("Choose a category and a colour from the project lists.")
    if not gallery:
        source = "image" if image_vector is not None else "text"
        result = _empty(source, "found")
        result["weights"] = {"visual": model.weights[0], "location": model.weights[1], "time": model.weights[2]}
        return result
    lost = datetime.fromisoformat(lost_time)
    return _rank(
        model,
        gallery,
        clip_by_id,
        category=category,
        colour=colour,
        query_location=location,
        image_vector=image_vector,
        lost_times=[lost] * len(gallery),
        found_times=[datetime.fromisoformat(record["found_time"]) for record in gallery],
        against="found",
    )


def search_lost(
    model: Model,
    gallery: list[dict],
    clip_by_id: dict[str, np.ndarray],
    *,
    category: str,
    colour: str,
    location: str,
    found_time: str,
    image_vector=None,
) -> dict:
    """Match a found-item query against lost reports."""
    if (colour, category) not in model.text:
        raise ValueError("Choose a category and a colour from the project lists.")
    if not gallery:
        source = "image" if image_vector is not None else "text"
        result = _empty(source, "lost")
        result["weights"] = {"visual": model.weights[0], "location": model.weights[1], "time": model.weights[2]}
        return result
    found = datetime.fromisoformat(found_time)
    return _rank(
        model,
        gallery,
        clip_by_id,
        category=category,
        colour=colour,
        query_location=location,
        image_vector=image_vector,
        lost_times=[datetime.fromisoformat(record["lost_time"]) for record in gallery],
        found_times=[found] * len(gallery),
        against="lost",
    )


def _rank(
    model: Model,
    gallery: list[dict],
    clip_by_id: dict[str, np.ndarray],
    *,
    category: str,
    colour: str,
    query_location: str,
    image_vector,
    lost_times: list[datetime],
    found_times: list[datetime],
    against: str,
) -> dict:
    location_close = np.array(
        [location_closeness(query_location, record["location"]) for record in gallery],
        dtype=np.float32,
    )
    location_far = np.array(
        [location_distance(query_location, record["location"]) for record in gallery],
        dtype=np.float32,
    )
    time_close = np.array(
        [time_closeness(lost, found) for lost, found in zip(lost_times, found_times)],
        dtype=np.float32,
    )
    time_far = np.array(
        [time_difference(lost, found) for lost, found in zip(lost_times, found_times)],
        dtype=np.float32,
    )
    matrix = np.stack([_visual_row(model, record, clip_by_id) for record in gallery]).astype(np.float32)
    if image_vector is None:
        visual = matrix @ model.text[(colour, category)]
        visual_source = "text"
    else:
        vector = np.asarray(image_vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise ValueError("Could not read a visual embedding from that photo.")
        visual = matrix @ (vector / norm)
        visual_source = "image"
    fused, visual_norm = clip_fusion(visual, location_close, time_close, model.weights)
    category_match = np.array(
        [1.0 if record.get("pred_category", record["category"]) == category else 0.0 for record in gallery],
        dtype=np.float32,
    )
    colour_match = np.array(
        [1.0 if record.get("pred_colour", record["colour"]) == colour else 0.0 for record in gallery],
        dtype=np.float32,
    )
    features = np.stack([category_match, colour_match, location_far, time_far], axis=1)
    probability = linear_probability(features, model.coefficients, model.bias)
    weights = model.weights
    return {
        "visual_source": visual_source,
        "against": against,
        "weights": {"visual": weights[0], "location": weights[1], "time": weights[2]},
        "clip": _top(
            gallery,
            fused,
            {
                "visual": visual,
                "visual_norm": visual_norm,
                "place_score": location_close,
                "time_score": time_close,
                "visual_part": weights[0] * visual_norm,
                "location_part": weights[1] * location_close,
                "time_part": weights[2] * time_close,
            },
        ),
        "linear": _top(
            gallery,
            probability,
            {
                "category_match": category_match,
                "colour_match": colour_match,
                "place_score": location_close,
                "time_score": time_close,
                "location_distance": location_far,
                "time_difference": time_far,
            },
        ),
    }
