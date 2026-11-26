"""Local web app for Group 26 lost-and-found matching.

    python app/server.py
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.campus import PLACES  # noqa: E402
from src.search import embed_upload, load_model, search_found, search_lost  # noqa: E402
from src.store import Store  # noqa: E402
from src.vision import CATEGORIES, COLOURS, DISPLAY_CATEGORY  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ART = ROOT / "artifacts"
SAFE_ID = re.compile(r"^(lost|found)_\d{4}$")

MODEL = None
STORE = None


def decode_image(encoded: str) -> Path | None:
    if not encoded:
        return None
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=False)
    if not raw:
        return None
    handle, name = tempfile.mkstemp(suffix=".jpg")
    os.close(handle)
    path = Path(name)
    path.write_bytes(raw)
    return path


def bootstrap() -> dict:
    return {
        "places": PLACES,
        "colours": COLOURS,
        "categories": [{"id": category, "label": DISPLAY_CATEGORY[category]} for category in CATEGORIES],
        "records": STORE.list_public(),
        "counts": STORE.counts(),
        "group": MODEL.metrics["group"],
        "fusion_weights": MODEL.metrics["fusion_weights"],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        if str(args[1] if len(args) > 1 else "").startswith("4") or str(args[1] if len(args) > 1 else "").startswith("5"):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 6_000_000:
            self._json({"error": "That photo is too large."}, 413)
            return None
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        data = path.read_bytes()
        guessed = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, data, guessed)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            name = path[len("/static/") :]
            target = (STATIC / name).resolve()
            if STATIC.resolve() not in target.parents and target != STATIC.resolve():
                self._send(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            self._file(target)
            return
        if path.startswith("/media/"):
            item_id = Path(path).stem
            if not SAFE_ID.match(item_id):
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            photo = STORE.photo_path(item_id)
            if photo is None:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            self._file(photo, "image/jpeg")
            return
        if path == "/api/bootstrap":
            self._json(bootstrap())
            return
        if path == "/api/records":
            self._json({"records": STORE.list_public(), "counts": STORE.counts()})
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if body is None:
                return
            if path == "/api/records":
                self._create(body)
                return
            if path == "/api/records/delete":
                self._delete(body)
                return
            if path == "/api/claims":
                self._json({"record": STORE.request_claim(str(body.get("found_id", "")), str(body.get("lost_id", "")), str(body.get("detail", ""))), "records": STORE.list_public()})
                return
            if path == "/api/claims/list":
                self._json({"claims": STORE.list_claims(str(body.get("password", "")))})
                return
            if path == "/api/claims/review":
                self._json({"record": STORE.review_claim(str(body.get("found_id", "")), body.get("approve") is True, str(body.get("password", ""))), "records": STORE.list_public()})
                return
            if path == "/api/records/lost-status":
                self._json({"record": STORE.set_lost_status(str(body.get("id", "")), body.get("found") is True, str(body.get("password", ""))), "records": STORE.list_public()})
                return
            if path == "/api/search":
                self._search(body)
                return
        except PermissionError as exc:
            self._json({"error": str(exc)}, 403)
            return
        except Exception as exc:  # user input, image decoding, validation
            self._json({"error": str(exc)}, 400)
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def _create(self, body: dict) -> None:
        kind = str(body.get("kind", "")).strip()
        encoded = body.get("image_base64") or ""
        temp = decode_image(encoded) if encoded else None
        clip_vector = None
        try:
            if temp is not None:
                clip_vector = embed_upload(temp)
            event_time = body.get("event_time") or body.get("lost_time") or body.get("found_time") or ""
            record = STORE.add(
                kind=kind,
                category=str(body.get("category", "")),
                colour=str(body.get("colour", "")),
                location=str(body.get("location", "")),
                event_time=str(event_time),
                note=str(body.get("note", "")),
                contact=str(body.get("contact", "")),
                image_path=temp,
                clip_vector=clip_vector,
            )
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        matches = {}
        if kind == "lost":
            gallery, vectors = STORE.found_for_search()
            matches = search_found(
                MODEL,
                gallery,
                vectors,
                category=record["category"],
                colour=record["colour"],
                location=record["location"],
                lost_time=record["lost_time"],
                image_vector=clip_vector,
            )
        elif kind == "found":
            gallery, vectors = STORE.lost_for_search()
            matches = search_lost(
                MODEL,
                gallery,
                vectors,
                category=record["category"],
                colour=record["colour"],
                location=record["location"],
                found_time=record["found_time"],
                image_vector=clip_vector,
            )
        self._json(
            {
                "record": record,
                "counts": STORE.counts(),
                "records": STORE.list_public(),
                "matches": matches,
            }
        )

    def _delete(self, body: dict) -> None:
        deleted = STORE.delete(str(body.get("id", "")), str(body.get("password", "")))
        self._json({"deleted": deleted, "counts": STORE.counts(), "records": STORE.list_public()})

    def _search(self, body: dict) -> None:
        encoded = body.get("image_base64") or ""
        temp = decode_image(encoded) if encoded else None
        image_vector = None
        try:
            if temp is not None:
                image_vector = embed_upload(temp)
            gallery, vectors = STORE.found_for_search()
            result = search_found(
                MODEL,
                gallery,
                vectors,
                category=str(body.get("category", "")),
                colour=str(body.get("colour", "")),
                location=str(body.get("location", "")),
                lost_time=str(body.get("lost_time") or body.get("event_time", "")),
                image_vector=image_vector,
            )
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        self._json(result)


def main() -> None:
    global MODEL, STORE
    if not (ART / "metrics.json").is_file():
        raise SystemExit("Missing artifacts. Run: python -m src.pipeline")
    print("Loading trained matcher weights...", flush=True)
    MODEL = load_model()
    STORE = Store()
    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
