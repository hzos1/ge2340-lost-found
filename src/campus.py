"""Campus places and the location / time scores from the project proposal."""

from __future__ import annotations

from datetime import datetime

# Pin positions are fractions of the official CityUHK location plan
# (app/static/campus-map.jpg, 1024 x 586).
# Yeung = AC1, Li = AC2, Lau Ming Wai = AC3. Canteens sit inside those buildings.
PLACES = [
    {
        "id": "library",
        "name": "Run Run Shaw Library",
        "short": "Library",
        "x": 0.215,
        "y": 0.47,
    },
    {
        "id": "ac1",
        "name": "Yeung Kin Man Academic Building (AC1)",
        "short": "AC1",
        "x": 0.265,
        "y": 0.505,
    },
    {
        "id": "boc",
        "name": "Bank of China Complex",
        "short": "BOC",
        "x": 0.355,
        "y": 0.50,
    },
    {
        "id": "sports",
        "name": "Hu Fa Kuang Sports Centre",
        "short": "Sports",
        "x": 0.40,
        "y": 0.455,
    },
    {
        "id": "ac2",
        "name": "Li Dak Sum Yip Yio Chin Academic Building (AC2)",
        "short": "AC2",
        "x": 0.45,
        "y": 0.565,
    },
    {
        "id": "ac3",
        "name": "Lau Ming Wai Academic Building (AC3)",
        "short": "AC3",
        "x": 0.43,
        "y": 0.655,
    },
    {
        "id": "hall",
        "name": "Student Residence",
        "short": "Halls",
        "x": 0.68,
        "y": 0.68,
    },
    {
        "id": "cmc",
        "name": "Run Run Shaw Creative Media Centre",
        "short": "CMC",
        "x": 0.825,
        "y": 0.575,
    },
]

PLACE_BY_ID = {place["id"]: place for place in PLACES}
PLACE_BY_NAME = {place["name"]: place for place in PLACES}
for alias, place_id in {
    "Yeung Kin Man Academic Building": "ac1",
    "AC1 Canteen": "ac1",
    "Li Dak Sum Yip Yio Chin Academic Building": "ac2",
    "AC2 Canteen": "ac2",
    "Lau Ming Wai Academic Building": "ac3",
    "AC3 Canteen": "ac3",
    "Bank of China (Hong Kong) Complex": "boc",
    "Student Residence Halls": "hall",
    "Hu Fa Kuang Sports Centre": "sports",
}.items():
    PLACE_BY_NAME[alias] = PLACE_BY_ID[place_id]

# A report filed several hours before the item was lost cannot be that item.
TIME_EARLY_HOURS = 6.0
TIME_WINDOW_HOURS = 72.0
# Map diagonal is about 1.0; this scale makes a neighbouring building useful
# and a far building close to zero.
LOCATION_SCALE = 0.75


def location_distance(place_a: str, place_b: str) -> float:
    """Euclidean distance scaled to [0, 1]. 0 means the same place."""
    first = PLACE_BY_NAME[place_a]
    second = PLACE_BY_NAME[place_b]
    raw = ((first["x"] - second["x"]) ** 2 + (first["y"] - second["y"]) ** 2) ** 0.5
    return float(min(1.0, raw / LOCATION_SCALE))


def location_closeness(place_a: str, place_b: str) -> float:
    return 1.0 - location_distance(place_a, place_b)


def time_difference(lost_time: datetime, found_time: datetime) -> float:
    """Absolute gap scaled to [0, 1]. A found time far before the loss is 1."""
    delta_hours = (found_time - lost_time).total_seconds() / 3600.0
    if delta_hours < -TIME_EARLY_HOURS:
        return 1.0
    return float(min(1.0, abs(delta_hours) / TIME_WINDOW_HOURS))


def time_closeness(lost_time: datetime, found_time: datetime) -> float:
    return 1.0 - time_difference(lost_time, found_time)


def nearest_other(place_name: str) -> str:
    others = [place["name"] for place in PLACES if place["name"] != place_name]
    others.sort(key=lambda name: location_distance(place_name, name))
    return others[0]
