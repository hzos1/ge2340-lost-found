"""Live lost-and-found records for the web app.

Training photos stay in artifacts/. This store only holds user uploads.
"""

from __future__ import annotations

import hmac
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from src.campus import PLACE_BY_NAME
from src.vision import CATEGORIES, COLOURS, DISPLAY_CATEGORY, save_thumb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
RECORDS_PATH = DATA / "records.json"
PHOTOS = DATA / "photos"
VECTORS_PATH = DATA / "vectors.npz"
ADMIN_PASSWORD = "12345678"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class Store:
    def __init__(self) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        PHOTOS.mkdir(parents=True, exist_ok=True)
        if not RECORDS_PATH.is_file():
            _atomic_write(RECORDS_PATH, "[]\n")
        self.records: list[dict] = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
        self.clip_by_id: dict[str, np.ndarray] = {}
        if VECTORS_PATH.is_file():
            arrays = np.load(VECTORS_PATH)
            ids = [str(item) for item in arrays["ids"].tolist()]
            for item_id, vector in zip(ids, arrays["clip"]):
                self.clip_by_id[item_id] = np.asarray(vector, dtype=np.float32)

    def _persist_records(self) -> None:
        _atomic_write(RECORDS_PATH, json.dumps(self.records, ensure_ascii=False, indent=2) + "\n")

    def _persist_vectors(self) -> None:
        if not self.clip_by_id:
            if VECTORS_PATH.is_file():
                VECTORS_PATH.unlink()
            return
        ids = list(self.clip_by_id)
        clip = np.stack([self.clip_by_id[item_id] for item_id in ids]).astype(np.float32)
        tmp = VECTORS_PATH.with_name(VECTORS_PATH.stem + ".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(ids), clip=clip)
        tmp.replace(VECTORS_PATH)

    def _next_id(self, kind: str) -> str:
        used = {record["id"] for record in self.records}
        index = 1
        while True:
            item_id = f"{kind}_{index:04d}"
            if item_id not in used:
                return item_id
            index += 1

    def public(self, record: dict) -> dict:
        item = {
            "id": record["id"],
            "kind": record["kind"],
            "category": record["category"],
            "category_label": record["category_label"],
            "colour": record["colour"],
            "location": record["location"],
            "location_id": record["location_id"],
            "x": record["x"],
            "y": record["y"],
            "note": record.get("note") or "",
            "has_photo": bool(record.get("has_photo")),
            "created_at": record["created_at"],
            "pred_category": record.get("pred_category", record["category"]),
            "pred_category_label": record.get("pred_category_label", record["category_label"]),
            "pred_colour": record.get("pred_colour", record["colour"]),
        }
        if record["kind"] == "found":
            item["found_time"] = record["found_time"]
        else:
            item["lost_time"] = record["lost_time"]
        return item

    def list_public(self) -> list[dict]:
        with _lock:
            return [self.public(record) for record in self.records]

    def counts(self) -> dict:
        with _lock:
            lost = sum(1 for record in self.records if record["kind"] == "lost")
            found = sum(1 for record in self.records if record["kind"] == "found")
            photos = sum(1 for record in self.records if record.get("has_photo"))
            return {"lost": lost, "found": found, "photos": photos, "total": len(self.records)}

    def found_for_search(self) -> tuple[list[dict], dict[str, np.ndarray]]:
        with _lock:
            found = [dict(record) for record in self.records if record["kind"] == "found"]
            vectors = {record["id"]: self.clip_by_id[record["id"]] for record in found if record["id"] in self.clip_by_id}
            return found, vectors

    def lost_for_search(self) -> tuple[list[dict], dict[str, np.ndarray]]:
        with _lock:
            lost = [dict(record) for record in self.records if record["kind"] == "lost"]
            vectors = {record["id"]: self.clip_by_id[record["id"]] for record in lost if record["id"] in self.clip_by_id}
            return lost, vectors

    def photo_path(self, item_id: str) -> Path | None:
        path = PHOTOS / f"{item_id}.jpg"
        return path if path.is_file() else None

    def add(
        self,
        *,
        kind: str,
        category: str,
        colour: str,
        location: str,
        event_time: str,
        note: str = "",
        image_path: Path | None = None,
        clip_vector=None,
    ) -> dict:
        if kind not in {"lost", "found"}:
            raise ValueError("Choose lost or found.")
        if category not in CATEGORIES:
            raise ValueError("Choose a category from the project list.")
        if colour not in COLOURS:
            raise ValueError("Choose a colour from the project list.")
        place = PLACE_BY_NAME.get(location)
        if place is None:
            raise ValueError("Choose a campus location from the list.")
        parsed = datetime.fromisoformat(event_time)
        event_iso = parsed.replace(microsecond=0).isoformat()
        if kind == "found" and image_path is None:
            raise ValueError("A photo is required when reporting a found item.")

        with _lock:
            item_id = self._next_id(kind)
            record = {
                "id": item_id,
                "kind": kind,
                "category": category,
                "category_label": DISPLAY_CATEGORY[category],
                "colour": colour,
                "location": place["name"],
                "location_id": place["id"],
                "x": place["x"],
                "y": place["y"],
                "note": (note or "").strip()[:400],
                "has_photo": False,
                "created_at": _now(),
                "pred_category": category,
                "pred_category_label": DISPLAY_CATEGORY[category],
                "pred_colour": colour,
            }
            if kind == "found":
                record["found_time"] = event_iso
            else:
                record["lost_time"] = event_iso
            if image_path is not None:
                dest = PHOTOS / f"{item_id}.jpg"
                save_thumb(image_path, dest)
                record["has_photo"] = True
                if clip_vector is not None:
                    vector = np.asarray(clip_vector, dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(vector))
                    if norm >= 1e-8:
                        self.clip_by_id[item_id] = vector / norm
            self.records.append(record)
            self._persist_records()
            self._persist_vectors()
            return self.public(record)

    def delete(self, item_id: str, password: str) -> dict:
        if not hmac.compare_digest(str(password or ""), ADMIN_PASSWORD):
            raise PermissionError("Wrong admin password.")
        with _lock:
            index = next((i for i, record in enumerate(self.records) if record["id"] == item_id), None)
            if index is None:
                raise ValueError("That record is not in the database.")
            record = self.records.pop(index)
            self.clip_by_id.pop(item_id, None)
            photo = PHOTOS / f"{item_id}.jpg"
            if photo.is_file():
                photo.unlink()
            self._persist_records()
            self._persist_vectors()
            return self.public(record)
