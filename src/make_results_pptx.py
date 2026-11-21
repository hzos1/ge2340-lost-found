"""Build the Group 26 results slides. The web app keeps the live demo only."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

from src.vision import DISPLAY_CATEGORY

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
THUMB = ROOT / "artifacts" / "thumbs"
METRICS = ROOT / "artifacts" / "metrics.json"
OUT = ROOT / "slides" / "GE2340_Group26_Results.pptx"

W = Inches(13.333)
H = Inches(7.5)
PINE = RGBColor(0x12, 0x38, 0x32)
PINE2 = RGBColor(0x1D, 0x6A, 0x5F)
COPPER = RGBColor(0xC4, 0x62, 0x2D)
CLAY = RGBColor(0x8D, 0x3D, 0x32)
INK = RGBColor(0x1C, 0x24, 0x20)
MUTED = RGBColor(0x5D, 0x67, 0x5F)
CREAM = RGBColor(0xF4, 0xEF, 0xE6)
WHITE = RGBColor(0xFF, 0xFD, 0xF9)


def set_run(run, text, size=18, bold=False, color=INK, font="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def fill_solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_rect(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    fill_solid(shape, color)
    return shape


def add_round(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    fill_solid(shape, color)
    return shape


def textbox(slide, left, top, width, height, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, font="Calibri"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(_ensure_run(p), text, size, bold, color, font)
    return box


def _ensure_run(paragraph):
    if paragraph.runs:
        return paragraph.runs[0]
    return paragraph.add_run()


def load_metrics() -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def short_place(name: str) -> str:
    if "(" in name:
        head, rest = name.split("(", 1)
        return f"{' '.join(head.strip().split()[:3])} ({rest}"
    return name


def heading(slide, title, kicker="Results"):
    add_rect(slide, 0, 0, W, Inches(0.18), PINE)
    textbox(slide, Inches(0.6), Inches(0.28), Inches(12), Inches(0.28), kicker.upper(), 12, True, COPPER)
    textbox(slide, Inches(0.6), Inches(0.5), Inches(12), Inches(0.6), title, 32, True, PINE, font="Georgia")


def footer(slide, page, total):
    textbox(
        slide,
        Inches(0.6),
        Inches(7.12),
        Inches(9),
        Inches(0.28),
        "GE2340 Group 26  ·  Lost-and-Found Image/Text Matcher",
        11,
        False,
        MUTED,
    )
    textbox(slide, Inches(11.2), Inches(7.12), Inches(1.5), Inches(0.28), f"{page} / {total}", 11, False, MUTED, PP_ALIGN.RIGHT)


def set_cell(cell, text, size=14, bold=False, color=INK, fill=None, align=PP_ALIGN.LEFT):
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    else:
        cell.fill.background()
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = _ensure_run(p)
    set_run(run, text, size, bold, color)
    for border in ("lnL", "lnR", "lnT", "lnB"):
        el = cell._tc.get_or_add_tcPr()
        ln = etree.SubElement(el, qn(f"a:{border}"))
        ln.set("w", "6350")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        srgb = etree.SubElement(sf, qn("a:srgbClr"))
        srgb.set("val", "DDD4C6")


def blank():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs, prs.slide_layouts[6]


def title_slide(prs, layout, metrics):
    test = metrics["test"]
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, PINE)
    add_rect(slide, 0, Inches(5.9), W, Inches(1.6), RGBColor(0x0E, 0x2C, 0x28))
    textbox(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.4), "GE2340  ·  GROUP 26  ·  CITYUHK", 14, True, RGBColor(0xD7, 0xC4, 0xA4))
    textbox(slide, Inches(0.8), Inches(1.85), Inches(11.5), Inches(1.8), "Lost-and-Found\nImage/Text Matcher", 48, True, WHITE, font="Georgia")
    textbox(slide, Inches(0.8), Inches(4.0), Inches(11), Inches(0.5), "Held-out retrieval results", 22, False, RGBColor(0xD9, 0xE4, 0xDF))
    names = "  ·  ".join(member["name"] for member in metrics["group"]["members"])
    textbox(
        slide,
        Inches(0.8),
        Inches(6.15),
        Inches(11),
        Inches(0.9),
        f"{names}\n{test['n_queries']} test queries  ·  {test['gallery_size']} found records  ·  success = correct item in the top 5",
        16,
        False,
        RGBColor(0xD9, 0xE4, 0xDF),
    )


def setup_slide(prs, layout, page, total, metrics):
    test = metrics["test"]
    n_classes = len(metrics["dataset"]["by_category"])
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "How we scored the system")
    cards = [
        (str(test["n_queries"]), "lost queries on the test split"),
        (str(test["gallery_size"]), "found records never used to train the heads"),
        ("Top 5", "counts as a hit, as in the proposal"),
        (f"{n_classes} classes", "keys, bottles, earphones, laptops, watches"),
    ]
    for i, (value, label) in enumerate(cards):
        left = Inches(0.6 + i * 3.15)
        add_round(slide, left, Inches(1.4), Inches(2.95), Inches(1.7), WHITE)
        textbox(slide, left + Inches(0.15), Inches(1.55), Inches(2.65), Inches(0.8), value, 32, True, PINE, font="Georgia")
        textbox(slide, left + Inches(0.15), Inches(2.3), Inches(2.65), Inches(0.65), label, 14, False, MUTED)
    bullets = [
        "CLIP ViT-B/32 is not fine-tuned. It matches “a photo of a {colour} {category}” to each found photo, then adds place and time.",
        "The CNN keeps frozen MobileNetV3 features. Softmax heads predict category and colour. A linear classifier scores the pair.",
        "Place and time are simulated on ten CityUHK locations. Colour names are read from the photo automatically.",
    ]
    top = Inches(3.4)
    for line in bullets:
        add_round(slide, Inches(0.6), top, Inches(12.1), Inches(1.0), WHITE)
        textbox(slide, Inches(0.85), top + Inches(0.22), Inches(11.6), Inches(0.65), line, 16, False, INK)
        top += Inches(1.08)
    footer(slide, page, total)


def headline_slide(prs, layout, page, total, metrics):
    methods = metrics["test"]["methods"]
    cnn = methods["cnn_linear"]
    fusion = methods["clip_fusion"]
    visual = methods["clip_visual"]
    random = methods["random"]
    oracle = methods["metadata_oracle"]
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Headline numbers on the test set")
    big = [
        (pct(cnn["recall_at_5"]), "CNN + linear", "Recall@5", COPPER),
        (pct(fusion["recall_at_5"]), "CLIP + place + time", "Recall@5", PINE2),
        (pct(visual["recall_at_5"]), "CLIP visual only", "Recall@5", RGBColor(0x3A, 0x7A, 0x82)),
        (pct(random["recall_at_5"]), "Random ranking", "Recall@5", MUTED),
    ]
    for i, (value, name, metric, color) in enumerate(big):
        left = Inches(0.6 + i * 3.15)
        add_round(slide, left, Inches(1.45), Inches(2.95), Inches(2.35), WHITE)
        textbox(slide, left + Inches(0.15), Inches(1.65), Inches(2.65), Inches(0.9), value, 36, True, color, font="Georgia")
        textbox(slide, left + Inches(0.15), Inches(2.55), Inches(2.65), Inches(0.4), name, 16, True, INK)
        textbox(slide, left + Inches(0.15), Inches(2.95), Inches(2.65), Inches(0.4), metric, 13, False, MUTED)
    add_round(slide, Inches(0.6), Inches(4.05), Inches(12.1), Inches(2.7), WHITE)
    textbox(slide, Inches(0.9), Inches(4.25), Inches(11.5), Inches(0.4), "What this means", 16, True, PINE)
    notes = [
        f"Place and time lift CLIP from {pct(visual['recall_at_5'])} to {pct(fusion['recall_at_5'])}. Many photos share a colour name, so the short text prompt cannot pick one record by itself.",
        f"The linear matcher is ahead of CLIP here, and close to an oracle that already knows the true category and colour ({pct(oracle['recall_at_5'])}).",
        f"Median rank is {cnn['median_rank']:.0f} for both matchers. Random ranking has a median rank of {random['median_rank']:.1f}.",
    ]
    top = Inches(4.75)
    for line in notes:
        textbox(slide, Inches(0.9), top, Inches(11.5), Inches(0.5), "•  " + line, 15, False, INK)
        top += Inches(0.52)
    footer(slide, page, total)


def table_slide(prs, layout, page, total, metrics):
    methods = metrics["test"]["methods"]

    def row(name, key):
        item = methods[key]
        return [name, pct(item["recall_at_1"]), pct(item["recall_at_5"]), f"{item['mrr']:.3f}", f"{item['median_rank']:.1f}"]

    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Full retrieval table")
    rows = [
        ["Method", "Recall@1", "Recall@5", "MRR", "Median rank"],
        row("CNN + linear", "cnn_linear"),
        row("CLIP + place + time", "clip_fusion"),
        row("True category and colour", "metadata_oracle"),
        row("CLIP visual only", "clip_visual"),
        row("Random ranking", "random"),
    ]
    table = slide.shapes.add_table(len(rows), 5, Inches(0.8), Inches(1.5), Inches(11.7), Inches(4.2)).table
    widths = [Inches(3.6), Inches(2.0), Inches(2.0), Inches(2.0), Inches(2.1)]
    for i, width in enumerate(widths):
        table.columns[i].width = width
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            header = r == 0
            best = r == 1
            align = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            fill = PINE if header else (RGBColor(0xE8, 0xF2, 0xEF) if best else WHITE)
            color = WHITE if header else (PINE if best else INK)
            set_cell(table.cell(r, c), value, 16 if header else 16, True if header or best else False, color, fill, align)
    textbox(
        slide,
        Inches(0.8),
        Inches(5.9),
        Inches(11.7),
        Inches(0.9),
        "True category and colour is an upper bound, not a third system: it already knows the labels. The highlighted row is the stronger matcher we can run from a photo.",
        15,
        False,
        MUTED,
    )
    footer(slide, page, total)


def chart_slide(prs, layout, page, total, title, image, caption):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, title)
    slide.shapes.add_picture(str(image), Inches(0.7), Inches(1.35), width=Inches(12.0))
    textbox(slide, Inches(0.7), Inches(6.55), Inches(12.0), Inches(0.45), caption, 14, False, MUTED)
    footer(slide, page, total)


def weights_slide(prs, layout, page, total, metrics):
    coef = metrics["linear_model"]["coefficients"]
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Why the linear matcher is easy to read")
    slide.shapes.add_picture(str(FIG / "linear_weights.png"), Inches(0.5), Inches(1.35), width=Inches(7.4))
    add_round(slide, Inches(8.1), Inches(1.45), Inches(4.6), Inches(5.2), WHITE)
    textbox(slide, Inches(8.35), Inches(1.65), Inches(4.15), Inches(0.4), "Learned pair features", 16, True, PINE)
    facts = [
        (signed(coef["category_match"]), "Category match raises the probability."),
        (signed(coef["colour_match"]), "Colour match helps, but less. Colour labels are noisy."),
        (signed(coef["location_distance"]), "A farther campus place lowers the score."),
        (signed(coef["time_difference"]), "A larger time gap lowers the score the most."),
    ]
    top = Inches(2.2)
    for value, line in facts:
        display = value.replace("-", "−")
        textbox(slide, Inches(8.35), top, Inches(4.15), Inches(0.4), display, 20, True, PINE2 if value.startswith("+") else CLAY, font="Georgia")
        textbox(slide, Inches(8.35), top + Inches(0.38), Inches(4.15), Inches(0.55), line, 14, False, INK)
        top += Inches(1.05)
    footer(slide, page, total)


def classify_slide(prs, layout, page, total, metrics):
    test_cls = metrics["classification"]["test"]
    labels = metrics["classification"]["confusion_category"]["labels"]
    matrix = metrics["classification"]["confusion_category"]["matrix"]
    key_idx = labels.index("key")
    key_row = matrix[key_idx]
    key_total = sum(key_row)
    key_wrong = key_total - key_row[key_idx]
    wrong_label = labels[int(max(range(len(key_row)), key=lambda i: key_row[i] if i != key_idx else -1))]
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Category is easy. Colour is not.")
    slide.shapes.add_picture(str(FIG / "category_confusion.png"), Inches(0.45), Inches(1.3), height=Inches(5.4))
    add_round(slide, Inches(7.3), Inches(1.45), Inches(5.4), Inches(5.2), WHITE)
    textbox(slide, Inches(7.55), Inches(1.7), Inches(5.0), Inches(0.4), "Held-out photo labels", 16, True, PINE)
    stats = [
        (pct(test_cls["category_accuracy"]), "Category softmax accuracy"),
        (pct(test_cls["colour_accuracy"]), "Colour softmax accuracy"),
        (pct(test_cls["category_recall"]["laptop"]), "Laptop photos labelled correctly"),
        (f"{key_wrong} / {key_total}", f"Keys confused with an {wrong_label}"),
    ]
    top = Inches(2.3)
    for value, label in stats:
        textbox(slide, Inches(7.55), top, Inches(5.0), Inches(0.45), value, 24, True, PINE, font="Georgia")
        textbox(slide, Inches(7.55), top + Inches(0.42), Inches(5.0), Inches(0.35), label, 14, False, MUTED)
        top += Inches(0.95)
    footer(slide, page, total)


def cases_slide(prs, layout, page, total, metrics):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "CLIP + place + time: three hits, one miss")
    for i, case in enumerate(metrics["cases"][:4]):
        query = case["query"]
        hit = case["hit_at_5"]
        status = f"{'Hit' if hit else 'Miss'} · rank {case['rank']}"
        name = f"{query['colour'].title()} {DISPLAY_CATEGORY[query['category']]}"
        place = short_place(query["location"])
        item_id = query["target_id"]
        left = Inches(0.5 + i * 3.2)
        add_round(slide, left, Inches(1.4), Inches(3.0), Inches(5.25), WHITE)
        photo = THUMB / f"{item_id}.jpg"
        if photo.is_file():
            slide.shapes.add_picture(str(photo), left + Inches(0.18), Inches(1.55), width=Inches(2.64), height=Inches(2.64))
        color = PINE2 if hit else CLAY
        textbox(slide, left + Inches(0.18), Inches(4.3), Inches(2.64), Inches(0.35), status, 13, True, color)
        textbox(slide, left + Inches(0.18), Inches(4.65), Inches(2.64), Inches(0.7), name, 18, True, INK)
        textbox(slide, left + Inches(0.18), Inches(5.35), Inches(2.64), Inches(0.9), place, 13, False, MUTED)
    footer(slide, page, total)


def takeaway_slide(prs, layout, page, total, metrics):
    methods = metrics["test"]["methods"]
    fusion_w = metrics["fusion_weights"]
    cls = metrics["classification"]["test"]
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Takeaways for the report")
    items = [
        (
            "CLIP needs place and time",
            f"Visual similarity alone is {pct(methods['clip_visual']['recall_at_5'])} Recall@5. Fusion weights chosen on validation are {fusion_w['visual']:.2f} visual, {fusion_w['location']:.2f} place, and {fusion_w['time']:.2f} time.",
        ),
        (
            "The linear model wins on this split",
            "Once the photo is reduced to category and colour, learned weights can lean on campus place and time. That decision is easy to explain.",
        ),
        (
            "Colour remains the weak label",
            f"Category accuracy is {pct(cls['category_accuracy'])}. Colour accuracy is {pct(cls['colour_accuracy'])}. Synthetic bottles make similar items hard to split.",
        ),
        (
            "Scope of the data",
            "Keys, water bottles, earphones, laptops, and watches are in this split. Phones and charging cables from the proposal are still missing.",
        ),
    ]
    for i, (title, body) in enumerate(items):
        col = i % 2
        row = i // 2
        left = Inches(0.6 + col * 6.35)
        top = Inches(1.4 + row * 2.55)
        add_round(slide, left, top, Inches(6.1), Inches(2.35), WHITE)
        textbox(slide, left + Inches(0.3), top + Inches(0.25), Inches(5.5), Inches(0.5), title, 20, True, PINE, font="Georgia")
        textbox(slide, left + Inches(0.3), top + Inches(0.85), Inches(5.5), Inches(1.2), body, 15, False, INK)
    footer(slide, page, total)


def limits_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Limits we should say out loud")
    limits = [
        "Campus place and time are simulated. They help ranking, but they are not real lost-and-found logs.",
        "Similar objects of the same colour stay hard to tell apart from a short phrase.",
        "Bottle images are synthetic, and key photos are few.",
        "The system runs on a local computer. ID cards, student cards, and faces are out of scope.",
        "Live demo: the web app at http://127.0.0.1:8765 shows matching and the found inventory, not these slides.",
    ]
    top = Inches(1.45)
    for i, line in enumerate(limits, start=1):
        add_round(slide, Inches(0.6), top, Inches(12.1), Inches(0.9), WHITE)
        textbox(slide, Inches(0.85), top + Inches(0.12), Inches(0.5), Inches(0.6), f"{i:02d}", 20, True, COPPER, font="Georgia")
        textbox(slide, Inches(1.5), top + Inches(0.22), Inches(10.9), Inches(0.55), line, 16, False, INK)
        top += Inches(1.02)
    footer(slide, page, total)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    prs, layout = blank()
    total = 11
    title_slide(prs, layout, metrics)
    setup_slide(prs, layout, 2, total, metrics)
    headline_slide(prs, layout, 3, total, metrics)
    table_slide(prs, layout, 4, total, metrics)
    chart_slide(
        prs,
        layout,
        5,
        total,
        "Correct found item inside the top 5",
        FIG / "recall_at_5.png",
        "Higher is better. True attributes already know category and colour, so they are an upper bound.",
    )
    chart_slide(
        prs,
        layout,
        6,
        total,
        "The same test, split by item type",
        FIG / "recall_by_category.png",
        "Earphones, laptops, and watches now sit in the same evaluation. CLIP still leans on place and time; the linear matcher stays ahead on Recall@5.",
    )
    weights_slide(prs, layout, 7, total, metrics)
    classify_slide(prs, layout, 8, total, metrics)
    cases_slide(prs, layout, 9, total, metrics)
    takeaway_slide(prs, layout, 10, total, metrics)
    limits_slide(prs, layout, 11, total)
    prs.save(OUT)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
