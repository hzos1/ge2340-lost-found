"""Live lost-and-found records for the web app.

Training photos stay in artifacts/. This store only holds user uploads.
"""

from __future__ import annotations

import hmac
import json
import shutil
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
DEMO_SEED_PATH = DATA / ".test_dataset_seeded"
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
        if not DEMO_SEED_PATH.exists():
            if self.records:
                DEMO_SEED_PATH.write_text("existing database preserved\n", encoding="utf-8")
            else:
                self._seed_test_dataset()

    def _seed_test_dataset(self) -> None:
        """Load the held-out gallery and synthetic test queries as demo records once."""
        art = ROOT / "artifacts"
        gallery_path = art / "gallery.json"
        queries_path = art / "queries_test.json"
        embeddings_path = art / "embeddings.npz"
        if not all(path.is_file() for path in (gallery_path, queries_path, embeddings_path)):
            return

        gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        arrays = np.load(embeddings_path)
        source_vectors = {
            str(item_id): np.asarray(vector, dtype=np.float32)
            for item_id, vector in zip(arrays["ids"].tolist(), arrays["clip"])
        }
        found_id_by_source: dict[str, str] = {}
        seeded: list[dict] = []
        category_labels = {
            source["category"]: source["category_label"] for source in gallery
        }

        for index, source in enumerate(gallery, start=1):
            item_id = f"found_{index:04d}"
            source_id = source["id"]
            found_id_by_source[source_id] = item_id
            place = PLACE_BY_NAME[source["location"]]
            photo_source = art / "thumbs" / f"{source_id}.jpg"
            has_photo = photo_source.is_file()
            if has_photo:
                shutil.copyfile(photo_source, PHOTOS / f"{item_id}.jpg")
            seeded.append({
                "id": item_id,
                "kind": "found",
                "category": source["category"],
                "category_label": source["category_label"],
                "colour": source["colour"],
                "location": place["name"],
                "location_id": place["id"],
                "x": place["x"],
                "y": place["y"],
                "note": "Demo record from the held-out test dataset.",
                "has_photo": has_photo,
                "created_at": source["found_time"],
                "found_time": source["found_time"],
                "pred_category": source.get("pred_category", source["category"]),
                "pred_category_label": source.get("pred_category_label", source["category_label"]),
                "pred_colour": source.get("pred_colour", source["colour"]),
                "status": "available",
                "contact": "demo@example.com",
            })
            vector = source_vectors.get(source_id)
            if vector is not None:
                norm = float(np.linalg.norm(vector))
                if norm >= 1e-8:
                    self.clip_by_id[item_id] = vector / norm

        for index, query in enumerate(queries, start=1):
            item_id = f"lost_{index:04d}"
            place = PLACE_BY_NAME[query["location"]]
            seeded.append({
                "id": item_id,
                "kind": "lost",
                "category": query["category"],
                "category_label": category_labels[query["category"]],
                "colour": query["colour"],
                "location": place["name"],
                "location_id": place["id"],
                "x": place["x"],
                "y": place["y"],
                "note": "Demo synthetic lost report from the held-out test dataset.",
                "has_photo": False,
                "created_at": query["lost_time"],
                "lost_time": query["lost_time"],
                "pred_category": query["category"],
                "pred_category_label": category_labels[query["category"]],
                "pred_colour": query["colour"],
                "status": "available",
                "contact": "demo@example.com",
                "demo_target_id": found_id_by_source.get(query["target_id"], ""),
            })

        self.records = seeded
        self._persist_records()
        self._persist_vectors()
        DEMO_SEED_PATH.write_text(
            f"Seeded {len(gallery)} found records and {len(queries)} synthetic lost reports.\n",
            encoding="utf-8",
        )


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
            "status": record.get("status", "available"),
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
            found = [dict(record) for record in self.records if record["kind"] == "found" and record.get("status", "available") != "returned"]
            vectors = {record["id"]: self.clip_by_id[record["id"]] for record in found if record["id"] in self.clip_by_id}
            return found, vectors

    def lost_for_search(self) -> tuple[list[dict], dict[str, np.ndarray]]:
        with _lock:
            lost = [dict(record) for record in self.records if record["kind"] == "lost" and record.get("status", "available") != "returned"]
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
        contact: str = "",
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
        contact = contact.strip()
        if kind == "lost" and (len(contact) > 120 or "@" not in contact or "." not in contact.split("@")[-1]):
            raise ValueError("Enter a valid contact email for the lost report.")

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
                "status": "available",
                "contact": contact,
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

    def request_claim(self, found_id: str, lost_id: str, detail: str) -> dict:
        detail = detail.strip()
        if not 10 <= len(detail) <= 300:
            raise ValueError("Enter 10–300 characters describing a private identifying detail.")
        with _lock:
            found = next((r for r in self.records if r["id"] == found_id and r["kind"] == "found"), None)
            lost = next((r for r in self.records if r["id"] == lost_id and r["kind"] == "lost"), None)
            if found is None or lost is None:
                raise ValueError("Both the found item and lost report must exist.")
            if found.get("status", "available") != "available" or lost.get("status", "available") == "returned":
                raise ValueError("This item is no longer available for a claim.")
            found["claim"] = {"lost_id": lost_id, "detail": detail, "requested_at": _now()}
            found["status"] = "claim_pending"
            self._persist_records()
            return self.public(found)

    def list_claims(self, password: str) -> list[dict]:
        self._check_admin(password)
        with _lock:
            claims = []
            for r in self.records:
                if r.get("status") != "claim_pending" or not r.get("claim"):
                    continue
                lost = next((item for item in self.records if item["id"] == r["claim"]["lost_id"]), None)
                claims.append(dict(found_id=r["id"], contact=lost.get("contact", "") if lost else "", **r["claim"]))
            return claims

    def review_claim(self, found_id: str, approve: bool, password: str) -> dict:
        self._check_admin(password)
        with _lock:
            found = next((r for r in self.records if r["id"] == found_id and r["kind"] == "found"), None)
            if found is None or found.get("status") != "claim_pending" or not found.get("claim"):
                raise ValueError("No pending claim exists for this item.")
            lost = next((r for r in self.records if r["id"] == found["claim"]["lost_id"]), None)
            if approve and lost is None:
                raise ValueError("The linked lost report no longer exists.")
            found["status"] = "returned" if approve else "available"
            if approve:
                lost["status"] = "returned"
            found.pop("claim", None)
            self._persist_records()
            return self.public(found)

    def set_lost_status(self, lost_id: str, found: bool, password: str) -> dict:
        self._check_admin(password)
        with _lock:
            lost = next((r for r in self.records if r["id"] == lost_id and r["kind"] == "lost"), None)
            if lost is None:
                raise ValueError("Lost report not found.")
            lost["status"] = "returned" if found else "available"
            self._persist_records()
            return self.public(lost)

    @staticmethod
    def _check_admin(password: str) -> None:
        if not hmac.compare_digest(str(password or ""), ADMIN_PASSWORD):
            raise PermissionError("Wrong admin password.")

    def delete(self, item_id: str, password: str) -> dict:
        self._check_admin(password)
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
