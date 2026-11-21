"""Build the found-item index, train the linear heads, and evaluate retrieval.

Run from the project folder:

    python -m src.pipeline
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.campus import PLACES, location_closeness, location_distance, nearest_other, time_closeness, time_difference
from src.scoring import clip_fusion, linear_probability
from src.vision import (
    CATEGORIES,
    COLOURS,
    DISPLAY_CATEGORY,
    ClipEncoder,
    CnnEncoder,
    can_read,
    dominant_colour,
    list_images,
    read_rgb_array,
    save_thumb,
    text_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 26
QUOTA = {
    "umbrella": {"train": 100, "val": 36, "test": 36},
    "bottle": {"train": 100, "val": 36, "test": 36},
    "key": {"train": 48, "val": 12, "test": 12},
}
SOURCES = {
    "key": "Kaggle Key-Images (Faique Ali, 2024)",
    "bottle": "Kaggle Bottles Synthetic Images (Fih, 2022)",
    "umbrella": "images.cv Umbrella Image Classification Dataset (2026)",
}


def take(paths: list[Path], count: int, rng: np.random.Generator) -> list[Path]:
    order = np.array(paths, dtype=object)
    rng.shuffle(order)
    chosen: list[Path] = []
    for path in order:
        if can_read(Path(path)):
            chosen.append(Path(path))
        if len(chosen) >= count:
            return chosen
    raise RuntimeError(f"Needed {count} readable images, found {len(chosen)} in the sampled folders.")


def collect_specs(rng: np.random.Generator) -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    umbrella_root = ROOT / "data"
    for split, count in QUOTA["umbrella"].items():
        folder = umbrella_root / split / "umbrella"
        specs.extend(("umbrella", split, path) for path in take(list_images(folder), count, rng))
    for category, folder in (("bottle", ROOT / "Water Bottle"), ("key", ROOT / "KEY")):
        total = sum(QUOTA[category].values())
        chosen = take(list_images(folder), total, rng)
        offset = 0
        for split in ("train", "val", "test"):
            count = QUOTA[category][split]
            specs.extend((category, split, path) for path in chosen[offset : offset + count])
            offset += count
    return specs


def build_records(rng: np.random.Generator, thumb_dir: Path) -> list[dict]:
    specs = collect_specs(rng)
    counters = {category: 0 for category in CATEGORIES}
    records = []
    for category, split, path in specs:
        counters[category] += 1
        item_id = f"{category}_{counters[category]:04d}"
        place = PLACES[int(rng.integers(0, len(PLACES)))]
        found = datetime(2026, 10, 20, int(rng.integers(8, 21)), int(rng.choice([0, 15, 30, 45])))
        found += timedelta(days=int(rng.integers(0, 28)))
        colour = dominant_colour(read_rgb_array(path, 96))
        save_thumb(path, thumb_dir / f"{item_id}.jpg")
        records.append(
            {
                "id": item_id,
                "category": category,
                "category_label": DISPLAY_CATEGORY[category],
                "colour": colour,
                "split": split,
                "location": place["name"],
                "location_id": place["id"],
                "x": place["x"],
                "y": place["y"],
                "found_time": found.isoformat(timespec="minutes"),
                "source_rel": path.relative_to(ROOT).as_posix(),
                "source": SOURCES[category],
            }
        )
    return records


def build_queries(records: list[dict], split: str, rng: np.random.Generator) -> list[dict]:
    items = sorted((record for record in records if record["split"] == split), key=lambda record: record["id"])
    queries = []
    place_names = [place["name"] for place in PLACES]
    for index, item in enumerate(items):
        roll = float(rng.random())
        if roll < 0.70:
            location = item["location"]
        elif roll < 0.90:
            location = nearest_other(item["location"])
        else:
            options = [name for name in place_names if name != item["location"]]
            location = options[int(rng.integers(0, len(options)))]
        found = datetime.fromisoformat(item["found_time"])
        lost = found - timedelta(hours=float(rng.uniform(1.0, 36.0)))
        queries.append(
            {
                "id": f"{split}_{index:03d}",
                "target_id": item["id"],
                "category": item["category"],
                "colour": item["colour"],
                "location": location,
                "lost_time": lost.isoformat(timespec="minutes"),
            }
        )
    return queries


def metadata_arrays(query: dict, gallery: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lost = datetime.fromisoformat(query["lost_time"])
    location_close = np.array(
        [location_closeness(query["location"], record["location"]) for record in gallery],
        dtype=np.float32,
    )
    time_close = np.array(
        [
            time_closeness(lost, datetime.fromisoformat(record["found_time"]))
            for record in gallery
        ],
        dtype=np.float32,
    )
    location_far = np.array(
        [location_distance(query["location"], record["location"]) for record in gallery],
        dtype=np.float32,
    )
    time_far = np.array(
        [time_difference(lost, datetime.fromisoformat(record["found_time"])) for record in gallery],
        dtype=np.float32,
    )
    return location_close, time_close, location_far, time_far


def evaluate(queries: list[dict], gallery: list[dict], score_of) -> dict:
    ranks = []
    by_category: dict[str, list[int]] = {}
    for query in queries:
        scores = np.asarray(score_of(query), dtype=np.float32)
        order = np.argsort(-scores, kind="mergesort")
        target = next(index for index, record in enumerate(gallery) if record["id"] == query["target_id"])
        rank = int(np.where(order == target)[0][0]) + 1
        ranks.append(rank)
        by_category.setdefault(query["category"], []).append(rank)
    return _summarise(ranks, by_category)


def _summarise(ranks: list[int], by_category: dict[str, list[int]]) -> dict:
    array = np.asarray(ranks, dtype=np.float32)
    count = max(1, len(ranks))

    def pack(values: np.ndarray) -> dict:
        return {
            "n": int(len(values)),
            "recall_at_1": float(np.mean(values <= 1)),
            "recall_at_5": float(np.mean(values <= 5)),
            "mrr": float(np.mean(1.0 / values)),
            "median_rank": float(np.median(values)),
        }

    summary = pack(array)
    summary["by_category"] = {}
    for category in CATEGORIES:
        values = np.asarray(by_category.get(category, []), dtype=np.float32)
        if len(values):
            summary["by_category"][category] = pack(values)
    summary["n"] = int(count if ranks else 0)
    return summary


def random_summary(query_count: int, gallery_size: int, by_category_counts: dict[str, int]) -> dict:
    harmonic = float(np.sum(1.0 / np.arange(1, gallery_size + 1)))
    summary = {
        "n": query_count,
        "recall_at_1": 1.0 / gallery_size,
        "recall_at_5": min(5, gallery_size) / gallery_size,
        "mrr": harmonic / gallery_size,
        "median_rank": (gallery_size + 1) / 2.0,
        "by_category": {},
    }
    for category, count in by_category_counts.items():
        summary["by_category"][category] = {
            "n": count,
            "recall_at_1": summary["recall_at_1"],
            "recall_at_5": summary["recall_at_5"],
            "mrr": summary["mrr"],
            "median_rank": summary["median_rank"],
        }
    return summary


def fit_softmax(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=800,
        class_weight="balanced",
        random_state=SEED,
        C=0.4,
        solver="lbfgs",
    )
    model.fit(features, labels)
    return model


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    scores = {}
    for label in labels:
        mask = y_true == label
        if not np.any(mask):
            continue
        scores[label] = float(np.mean(y_pred[mask] == label))
    return scores


def train_heads(features: np.ndarray, records: list[dict], train_index: np.ndarray, gallery_index: np.ndarray):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[train_index])
    train_category = np.array([records[i]["category"] for i in train_index])
    train_colour = np.array([records[i]["colour"] for i in train_index])
    category_model = fit_softmax(train_x, train_category)
    colour_model = fit_softmax(train_x, train_colour)
    gallery_x = scaler.transform(features[gallery_index])
    predicted_category = category_model.predict(gallery_x)
    predicted_colour = colour_model.predict(gallery_x)
    return scaler, category_model, colour_model, predicted_category, predicted_colour


def classification_report(scaler, category_model, colour_model, features, records, indexes) -> dict:
    report = {}
    for name, index in indexes.items():
        transformed = scaler.transform(features[index])
        true_category = np.array([records[i]["category"] for i in index])
        true_colour = np.array([records[i]["colour"] for i in index])
        pred_category = category_model.predict(transformed)
        pred_colour = colour_model.predict(transformed)
        report[name] = {
            "n": int(len(index)),
            "category_accuracy": float(accuracy_score(true_category, pred_category)),
            "colour_accuracy": float(accuracy_score(true_colour, pred_colour)),
            "category_recall": per_class_recall(true_category, pred_category, CATEGORIES),
            "colour_recall": per_class_recall(true_colour, pred_colour, COLOURS),
        }
        if name == "test":
            report["confusion_category"] = {
                "labels": CATEGORIES,
                "matrix": confusion_matrix(true_category, pred_category, labels=CATEGORIES).tolist(),
            }
    return report


def fit_linear_matcher(val_queries: list[dict], gallery: list[dict], pred_category, pred_colour, rng):
    rows = []
    labels = []
    id_to_index = {record["id"]: index for index, record in enumerate(gallery)}
    for query in val_queries:
        positive = id_to_index[query["target_id"]]
        _, _, location_far, time_far = metadata_arrays(query, gallery)
        rows.append(
            [
                1.0 if pred_category[positive] == query["category"] else 0.0,
                1.0 if pred_colour[positive] == query["colour"] else 0.0,
                float(location_far[positive]),
                float(time_far[positive]),
            ]
        )
        labels.append(1)
        choices = [index for index in range(len(gallery)) if index != positive]
        negatives = rng.choice(choices, size=8, replace=False)
        for index in negatives:
            rows.append(
                [
                    1.0 if pred_category[index] == query["category"] else 0.0,
                    1.0 if pred_colour[index] == query["colour"] else 0.0,
                    float(location_far[index]),
                    float(time_far[index]),
                ]
            )
            labels.append(0)
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )
    model.fit(np.asarray(rows, dtype=np.float32), np.asarray(labels))
    return model.coef_.ravel().astype(np.float32), float(model.intercept_[0])


def tune_fusion(val_queries, gallery, visual_of) -> tuple[tuple[float, float, float], list[dict]]:
    trials = []
    for visual_weight in (0.45, 0.55, 0.65, 0.75, 0.85):
        for location_weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            time_weight = round(1.0 - visual_weight - location_weight, 2)
            if time_weight < 0.05:
                continue
            weights = (visual_weight, location_weight, time_weight)

            def score_of(query, weights=weights):
                visual = visual_of(query)
                location_close, time_close, _, _ = metadata_arrays(query, gallery)
                fused, _ = clip_fusion(visual, location_close, time_close, weights)
                return fused

            summary = evaluate(val_queries, gallery, score_of)
            trials.append(
                {
                    "weights": {
                        "visual": visual_weight,
                        "location": location_weight,
                        "time": time_weight,
                    },
                    "recall_at_5": summary["recall_at_5"],
                    "mrr": summary["mrr"],
                }
            )
    trials.sort(key=lambda trial: (trial["recall_at_5"], trial["mrr"], trial["weights"]["visual"]), reverse=True)
    best = trials[0]["weights"]
    return (best["visual"], best["location"], best["time"]), trials


def oracle_scores(query, gallery) -> np.ndarray:
    location_close, time_close, _, _ = metadata_arrays(query, gallery)
    category = np.array(
        [1.0 if record["category"] == query["category"] else 0.0 for record in gallery],
        dtype=np.float32,
    )
    colour = np.array(
        [1.0 if record["colour"] == query["colour"] else 0.0 for record in gallery],
        dtype=np.float32,
    )
    return 0.42 * category + 0.33 * colour + 0.15 * location_close + 0.10 * time_close


def top_case(query, gallery, scores, method: str) -> dict:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    target = query["target_id"]
    ranked = []
    hit_rank = None
    for position, index in enumerate(order[:5], start=1):
        record = gallery[int(index)]
        ranked.append(
            {
                "rank": position,
                "id": record["id"],
                "score": float(scores[int(index)]),
                "is_target": record["id"] == target,
            }
        )
        if record["id"] == target:
            hit_rank = position
    if hit_rank is None:
        full_rank = int(np.where(np.array([record["id"] for record in gallery])[order] == target)[0][0]) + 1
    else:
        full_rank = hit_rank
    return {
        "method": method,
        "rank": full_rank,
        "hit_at_5": full_rank <= 5,
        "query": query,
        "top": ranked,
    }


def write_findings(metrics: dict) -> list[str]:
    test = metrics["test"]
    methods = test["methods"]
    clip = methods["clip_fusion"]
    visual = methods["clip_visual"]
    linear = methods["cnn_linear"]
    oracle = methods["metadata_oracle"]
    random_rank = methods["random"]
    weights = metrics["fusion_weights"]
    coefficients = metrics["linear_model"]["coefficients"]
    classification = metrics["classification"]["test"]
    lines = [
        (
            f"The test set has {test['n_queries']} lost-item queries. Each query is ranked against "
            f"{test['gallery_size']} found records that the classifiers never trained on. A search counts "
            "as successful when the correct record appears in the top 5."
        ),
        (
            f"CLIP with location and time reaches Recall@5 of {clip['recall_at_5']:.1%} "
            f"(MRR {clip['mrr']:.3f}). CLIP visual similarity alone reaches {visual['recall_at_5']:.1%}. "
            f"The CNN plus linear classifier reaches {linear['recall_at_5']:.1%}. "
            f"Random ranking reaches {random_rank['recall_at_5']:.1%}. "
            f"An oracle that already knows the true category and colour reaches {oracle['recall_at_5']:.1%}."
        ),
    ]
    if clip["recall_at_5"] >= visual["recall_at_5"]:
        lines.append(
            "Adding place and time lifts CLIP above visual similarity alone. "
            f"The validation search set the weights to {weights['visual']:.2f} visual, "
            f"{weights['location']:.2f} place, and {weights['time']:.2f} time. "
            "Many photos share a colour name, so the short text prompt cannot pick one record by itself, and place and time break those ties."
        )
    else:
        lines.append(
            "On this test split, CLIP visual similarity alone is at least as strong as the fused score. "
            "Location and time still help when several photos share a colour and a category, but the weight search "
            "kept most of the weight on the image-text similarity."
        )
    if clip["recall_at_5"] >= linear["recall_at_5"]:
        lines.append(
            "CLIP is the stronger matcher here. It can compare a colour-and-category phrase with a photo directly, "
            "so it still works when the CNN colour head is unsure. The linear model is easier to explain: its four inputs "
            "are a category agreement, a colour agreement, a location distance, and a time gap."
        )
    else:
        lines.append(
            "The CNN plus linear classifier is ahead of CLIP on this split. Once the photo is reduced to category and colour, "
            "the learned weights can lean on place and time. That decision is easy to explain, and it can only use the colour names seen in training."
        )
    coef_bits = []
    descriptions = {
        "category_match": "category agreement",
        "colour_match": "colour agreement",
        "location_distance": "location distance",
        "time_difference": "time difference",
    }
    for name, value in coefficients.items():
        direction = "raises" if value >= 0 else "lowers"
        coef_bits.append(f"{descriptions[name]} {direction} the probability (weight {value:.2f})")
    lines.append("In the linear matcher, " + "; ".join(coef_bits) + ".")
    lines.append(
        f"On held-out photos, the category softmax reaches {classification['category_accuracy']:.1%} accuracy and "
        f"the colour softmax reaches {classification['colour_accuracy']:.1%}. "
        "Colour labels come from an automatic dominant-colour reading, so outdoor backgrounds can make the umbrella labels noisy. "
        "The bottle photos are synthetic product shots, and the key set is small, so similar items remain hard to separate."
    )
    lines.append(
        "Campus place and time are simulated, which matches the data plan in the proposal. "
        "The system runs locally, and the public photos do not include student cards or faces. "
        "Phones, cables, earphones, and computers from the proposal are not in the collected image folders, so this version covers keys, umbrellas, and water bottles."
    )
    return lines


def save_figures(metrics: dict, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#1c2430",
            "axes.labelcolor": "#1c2430",
            "xtick.color": "#1c2430",
            "ytick.color": "#1c2430",
            "text.color": "#1c2430",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    order = [
        ("random", "Random"),
        ("metadata_oracle", "True attributes"),
        ("clip_visual", "CLIP visual"),
        ("clip_fusion", "CLIP + place + time"),
        ("cnn_linear", "CNN + linear"),
    ]
    colours = ["#8d8478", "#d08a45", "#1f6f78", "#0f3d3e", "#b6402a"]
    values = [metrics["test"]["methods"][key]["recall_at_5"] * 100 for key, _ in order]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = ax.bar([label for _, label in order], values, color=colours, width=0.72)
    ax.set_ylabel("Recall@5 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Held-out lost queries: correct found item inside the top 5")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "recall_at_5.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    category_labels = [DISPLAY_CATEGORY[category].title() for category in CATEGORIES]
    x = np.arange(len(CATEGORIES))
    width = 0.25
    series = [
        ("clip_fusion", "CLIP + place + time", "#0f3d3e"),
        ("cnn_linear", "CNN + linear", "#b6402a"),
        ("metadata_oracle", "True attributes", "#d08a45"),
    ]
    for offset, (key, label, colour) in enumerate(series):
        heights = [
            metrics["test"]["methods"][key]["by_category"][category]["recall_at_5"] * 100
            for category in CATEGORIES
        ]
        ax.bar(x + (offset - 1) * width, heights, width=width, label=label, color=colour)
    ax.set_xticks(x, category_labels)
    ax.set_ylabel("Recall@5 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Recall@5 by item category")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "recall_by_category.png", dpi=150)
    plt.close(fig)

    coef = metrics["linear_model"]["coefficients"]
    names = ["Category\nmatch", "Colour\nmatch", "Location\ndistance", "Time\ndifference"]
    keys = ["category_match", "colour_match", "location_distance", "time_difference"]
    coef_values = [coef[key] for key in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bar_colours = ["#0f3d3e" if value >= 0 else "#b6402a" for value in coef_values]
    ax.bar(names, coef_values, color=bar_colours, width=0.7)
    ax.axhline(0, color="#1c2430", linewidth=0.8)
    ax.set_ylabel("Learned weight")
    ax.set_title("Linear match classifier")
    fig.tight_layout()
    fig.savefig(figure_dir / "linear_weights.png", dpi=150)
    plt.close(fig)

    matrix = np.array(metrics["classification"]["confusion_category"]["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="YlGn")
    ax.set_xticks(range(3), [DISPLAY_CATEGORY[label].title() for label in CATEGORIES], rotation=20, ha="right")
    ax.set_yticks(range(3), [DISPLAY_CATEGORY[label].title() for label in CATEGORIES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True category")
    ax.set_title("Category softmax on the test photos")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="#1c2430")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figure_dir / "category_confusion.png", dpi=150)
    plt.close(fig)


def pythonize(value):
    if isinstance(value, dict):
        return {str(key): pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonize(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(pythonize(payload), indent=2), encoding="utf-8")


def print_table(metrics: dict) -> None:
    print("\nTest retrieval", flush=True)
    print(f"{'method':<22}{'R@1':>8}{'R@5':>8}{'MRR':>8}{'median':>8}", flush=True)
    for name, summary in metrics["test"]["methods"].items():
        print(
            f"{name:<22}{summary['recall_at_1']*100:7.1f}%{summary['recall_at_5']*100:7.1f}%"
            f"{summary['mrr']:8.3f}{summary['median_rank']:8.1f}",
            flush=True,
        )
    test_cls = metrics["classification"]["test"]
    print(
        f"\nTest classification  category {test_cls['category_accuracy']*100:.1f}%   "
        f"colour {test_cls['colour_accuracy']*100:.1f}%",
        flush=True,
    )
    print("Fusion weights", metrics["fusion_weights"], flush=True)
    print("Linear weights", metrics["linear_model"], flush=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    art = ROOT / "artifacts"
    thumb_dir = art / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    print("Sampling photos and simulating campus reports...", flush=True)
    records = build_records(rng, thumb_dir)
    print(f"Records: {len(records)}", flush=True)
    print("Colours:", dict(Counter(record["colour"] for record in records)), flush=True)
    print("Categories:", dict(Counter(record["category"] for record in records)), flush=True)

    val_queries = build_queries(records, "val", rng)
    test_queries = build_queries(records, "test", rng)
    gallery = [record for record in records if record["split"] != "train"]
    gallery.sort(key=lambda record: record["id"])
    record_index = {record["id"]: index for index, record in enumerate(records)}
    train_index = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    gallery_index = np.array([record_index[record["id"]] for record in gallery])

    print("Encoding frozen MobileNet features...", flush=True)
    cnn = CnnEncoder()
    cnn_features = cnn.encode_images([ROOT / record["source_rel"] for record in records])
    scaler, category_model, colour_model, pred_category, pred_colour = train_heads(
        cnn_features, records, train_index, gallery_index
    )
    for record, category, colour in zip(gallery, pred_category, pred_colour):
        record["pred_category"] = str(category)
        record["pred_colour"] = str(colour)
        record["pred_category_label"] = DISPLAY_CATEGORY[str(category)]

    print("Encoding CLIP gallery and text prompts...", flush=True)
    clip = ClipEncoder()
    clip_images = clip.encode_images([ROOT / record["source_rel"] for record in gallery])
    text_keys = []
    prompts = []
    for category in CATEGORIES:
        for colour in COLOURS:
            text_keys.append(f"{colour}|{category}")
            prompts.append(text_prompt(colour, category))
    clip_text = clip.encode_texts(prompts)
    text_lookup = {key: clip_text[index] for index, key in enumerate(text_keys)}

    def visual_of(query: dict) -> np.ndarray:
        vector = text_lookup[f"{query['colour']}|{query['category']}"]
        return clip_images @ vector

    print("Choosing fusion weights on the validation queries...", flush=True)
    weights, trials = tune_fusion(val_queries, gallery, visual_of)
    print("Fitting the linear match classifier on validation pairs...", flush=True)
    coefficients, bias = fit_linear_matcher(
        val_queries, gallery, pred_category, pred_colour, np.random.default_rng(SEED + 1)
    )

    def clip_visual_scores(query):
        return visual_of(query)

    def clip_fused_scores(query):
        location_close, time_close, _, _ = metadata_arrays(query, gallery)
        fused, _ = clip_fusion(visual_of(query), location_close, time_close, weights)
        return fused

    def linear_scores(query):
        _, _, location_far, time_far = metadata_arrays(query, gallery)
        features = np.stack(
            [
                (pred_category == query["category"]).astype(np.float32),
                (pred_colour == query["colour"]).astype(np.float32),
                location_far,
                time_far,
            ],
            axis=1,
        )
        return linear_probability(features, coefficients, bias)

    print("Evaluating...", flush=True)
    splits = {"val": val_queries, "test": test_queries}
    methods = {}
    for split_name, queries in splits.items():
        category_counts = dict(Counter(query["category"] for query in queries))
        methods[split_name] = {
            "n_queries": len(queries),
            "gallery_size": len(gallery),
            "methods": {
                "random": random_summary(len(queries), len(gallery), category_counts),
                "metadata_oracle": evaluate(queries, gallery, lambda query: oracle_scores(query, gallery)),
                "clip_visual": evaluate(queries, gallery, clip_visual_scores),
                "clip_fusion": evaluate(queries, gallery, clip_fused_scores),
                "cnn_linear": evaluate(queries, gallery, linear_scores),
            },
        }

    classification = classification_report(
        scaler,
        category_model,
        colour_model,
        cnn_features,
        records,
        {
            "val": np.array([record_index[record["id"]] for record in gallery if record["split"] == "val"]),
            "test": np.array([record_index[record["id"]] for record in gallery if record["split"] == "test"]),
        },
    )

    successes = []
    misses = []
    for query in test_queries:
        scores = clip_fused_scores(query)
        case = top_case(query, gallery, scores, "clip_fusion")
        if case["hit_at_5"]:
            successes.append(case)
        else:
            misses.append(case)
    successes.sort(key=lambda case: case["rank"])
    misses.sort(key=lambda case: case["rank"], reverse=True)
    cases = []
    seen_categories = set()
    for case in successes:
        if case["query"]["category"] in seen_categories:
            continue
        seen_categories.add(case["query"]["category"])
        cases.append(case)
        if len(cases) == 3:
            break
    if misses:
        cases.append(misses[0])

    counts = {
        "total_photos": len(records),
        "gallery": len(gallery),
        "by_split": dict(Counter(record["split"] for record in records)),
        "by_category": dict(Counter(record["category"] for record in records)),
        "by_colour": dict(Counter(record["colour"] for record in records)),
        "sources": SOURCES,
    }
    metrics = {
        "group": {
            "course": "GE2340",
            "group": "26",
            "title": "Lost-and-Found Image/Text Matcher",
            "members": [
                {"name": "So Shing Hin", "id": "59687520"},
                {"name": "Tu Kwan Yiu", "id": "60175669"},
                {"name": "Chan Kai Shun", "id": "59696515"},
            ],
        },
        "seed": SEED,
        "dataset": counts,
        "models": {
            "clip": clip.name,
            "cnn": cnn.name,
            "heads": "Multinomial logistic regression (softmax) on frozen CNN features",
            "matcher": "Logistic regression on category match, colour match, location distance, and time difference",
        },
        "fusion_weights": {"visual": weights[0], "location": weights[1], "time": weights[2]},
        "weight_search": trials,
        "linear_model": {
            "coefficients": {
                "category_match": float(coefficients[0]),
                "colour_match": float(coefficients[1]),
                "location_distance": float(coefficients[2]),
                "time_difference": float(coefficients[3]),
            },
            "bias": bias,
        },
        "classification": classification,
        "val": methods["val"],
        "test": methods["test"],
        "cases": cases,
        "examples": [
            {
                "label": f"{case['query']['colour'].title()} {DISPLAY_CATEGORY[case['query']['category']]}",
                "query": case["query"],
            }
            for case in cases
            if case["hit_at_5"]
        ][:3],
    }
    metrics["findings"] = write_findings(metrics)

    gallery_out = []
    for record in gallery:
        item = dict(record)
        gallery_out.append(item)
    dump_json(art / "gallery.json", gallery_out)
    dump_json(art / "queries_val.json", val_queries)
    dump_json(art / "queries_test.json", test_queries)
    dump_json(art / "metrics.json", metrics)
    dump_json(
        art / "backend.json",
        {"clip": clip.name, "cnn": cnn.name, "text_prompt": "a photo of a {colour} {category}"},
    )
    np.savez_compressed(
        art / "embeddings.npz",
        ids=np.array([record["id"] for record in gallery]),
        clip=clip_images.astype(np.float32),
        text_keys=np.array(text_keys),
        text_vecs=clip_text.astype(np.float32),
    )
    save_figures(metrics, art / "figures")
    print_table(metrics)
    print(f"\nSaved artifacts to {art}", flush=True)


if __name__ == "__main__":
    main()
