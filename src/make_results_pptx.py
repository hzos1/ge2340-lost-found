"""Build the Group 26 results slides. The web app keeps the live demo only."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
THUMB = ROOT / "artifacts" / "thumbs"
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


def title_slide(prs, layout):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, PINE)
    add_rect(slide, 0, Inches(5.9), W, Inches(1.6), RGBColor(0x0E, 0x2C, 0x28))
    textbox(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.4), "GE2340  ·  GROUP 26  ·  CITYUHK", 14, True, RGBColor(0xD7, 0xC4, 0xA4))
    textbox(slide, Inches(0.8), Inches(1.85), Inches(11.5), Inches(1.8), "Lost-and-Found\nImage/Text Matcher", 48, True, WHITE, font="Georgia")
    textbox(slide, Inches(0.8), Inches(4.0), Inches(11), Inches(0.5), "Held-out retrieval results", 22, False, RGBColor(0xD9, 0xE4, 0xDF))
    textbox(
        slide,
        Inches(0.8),
        Inches(6.15),
        Inches(11),
        Inches(0.9),
        "So Shing Hin  ·  Tu Kwan Yiu  ·  Chan Kai Shun\n84 test queries  ·  168 found records  ·  success = correct item in the top 5",
        16,
        False,
        RGBColor(0xD9, 0xE4, 0xDF),
    )


def setup_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "How we scored the system")
    cards = [
        ("84", "lost queries on the test split"),
        ("168", "found records never used to train the heads"),
        ("Top 5", "counts as a hit, as in the proposal"),
        ("3 classes", "keys, umbrellas, water bottles"),
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


def headline_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Headline numbers on the test set")
    big = [
        ("94.0%", "CNN + linear", "Recall@5", COPPER),
        ("77.4%", "CLIP + place + time", "Recall@5", PINE2),
        ("13.1%", "CLIP visual only", "Recall@5", RGBColor(0x3A, 0x7A, 0x82)),
        ("3.0%", "Random ranking", "Recall@5", MUTED),
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
        "Place and time lift CLIP from 13.1% to 77.4%. Many photos share a colour name, so the short text prompt cannot pick one record by itself.",
        "The linear matcher is ahead of CLIP here, and close to an oracle that already knows the true category and colour (95.2%).",
        "Median rank is 2 for both matchers. Random ranking has a median rank of 84.5.",
    ]
    top = Inches(4.75)
    for line in notes:
        textbox(slide, Inches(0.9), top, Inches(11.5), Inches(0.5), "•  " + line, 15, False, INK)
        top += Inches(0.52)
    footer(slide, page, total)


def table_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Full retrieval table")
    rows = [
        ["Method", "Recall@1", "Recall@5", "MRR", "Median rank"],
        ["CNN + linear", "48.8%", "94.0%", "0.663", "2.0"],
        ["CLIP + place + time", "41.7%", "77.4%", "0.571", "2.0"],
        ["True category and colour", "83.3%", "95.2%", "0.887", "1.0"],
        ["CLIP visual only", "4.8%", "13.1%", "0.117", "24.0"],
        ["Random ranking", "0.6%", "3.0%", "0.034", "84.5"],
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


def weights_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Why the linear matcher is easy to read")
    slide.shapes.add_picture(str(FIG / "linear_weights.png"), Inches(0.5), Inches(1.35), width=Inches(7.4))
    add_round(slide, Inches(8.1), Inches(1.45), Inches(4.6), Inches(5.2), WHITE)
    textbox(slide, Inches(8.35), Inches(1.65), Inches(4.15), Inches(0.4), "Learned pair features", 16, True, PINE)
    facts = [
        ("+3.48", "Category match raises the probability."),
        ("+0.46", "Colour match helps, but less. Colour labels are noisy."),
        ("−3.15", "A farther campus place lowers the score."),
        ("−6.31", "A larger time gap lowers the score the most."),
    ]
    top = Inches(2.2)
    for value, line in facts:
        textbox(slide, Inches(8.35), top, Inches(4.15), Inches(0.4), value, 20, True, PINE2 if value.startswith("+") else CLAY, font="Georgia")
        textbox(slide, Inches(8.35), top + Inches(0.38), Inches(4.15), Inches(0.55), line, 14, False, INK)
        top += Inches(1.05)
    footer(slide, page, total)


def classify_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Category is easy. Colour is not.")
    slide.shapes.add_picture(str(FIG / "category_confusion.png"), Inches(0.45), Inches(1.3), height=Inches(5.4))
    add_round(slide, Inches(7.3), Inches(1.45), Inches(5.4), Inches(5.2), WHITE)
    textbox(slide, Inches(7.55), Inches(1.7), Inches(5.0), Inches(0.4), "Held-out photo labels", 16, True, PINE)
    stats = [
        ("96.4%", "Category softmax accuracy"),
        ("59.5%", "Colour softmax accuracy"),
        ("100%", "Umbrella photos labelled correctly"),
        ("1 / 12", "Keys confused with a water bottle"),
    ]
    top = Inches(2.3)
    for value, label in stats:
        textbox(slide, Inches(7.55), top, Inches(5.0), Inches(0.45), value, 24, True, PINE, font="Georgia")
        textbox(slide, Inches(7.55), top + Inches(0.42), Inches(5.0), Inches(0.35), label, 14, False, MUTED)
        top += Inches(0.95)
    footer(slide, page, total)


def cases_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "CLIP + place + time: three hits, one miss")
    cases = [
        ("Hit · rank 1", "Gray water bottle", "Student Residence Halls", "bottle_0141", True),
        ("Hit · rank 1", "Black key", "Li Dak Sum Yip Yio Chin", "key_0061", True),
        ("Hit · rank 1", "Gray umbrella", "Lau Ming Wai", "umbrella_0139", True),
        ("Miss · rank 54", "Gray umbrella", "Bank of China Complex", "umbrella_0143", False),
    ]
    for i, (status, name, place, item_id, hit) in enumerate(cases):
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


def takeaway_slide(prs, layout, page, total):
    slide = prs.slides.add_slide(layout)
    add_rect(slide, 0, 0, W, H, CREAM)
    heading(slide, "Takeaways for the report")
    items = [
        ("CLIP needs place and time", "Visual similarity alone is 13.1% Recall@5. Fusion weights chosen on validation are 0.45 visual, 0.25 place, 0.30 time."),
        ("The linear model wins on this split", "Once the photo is reduced to category and colour, learned weights can lean on campus place and time. That decision is easy to explain."),
        ("Colour remains the weak label", "Category accuracy is 96.4%. Colour accuracy is 59.5%. Outdoor umbrellas and synthetic bottles make similar items hard to split."),
        ("Scope of the data", "Keys, umbrellas, and water bottles from the public sets named in the proposal. Phones, cables, earphones, and computers were not in the collected folders."),
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
        "Bottle images are synthetic, key photos are few, and umbrella scenes are uncontrolled.",
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
    prs, layout = blank()
    total = 11
    title_slide(prs, layout)
    setup_slide(prs, layout, 2, total)
    headline_slide(prs, layout, 3, total)
    table_slide(prs, layout, 4, total)
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
        "Keys are easy for every matcher. Umbrellas are where CLIP falls behind the linear model.",
    )
    weights_slide(prs, layout, 7, total)
    classify_slide(prs, layout, 8, total)
    cases_slide(prs, layout, 9, total)
    takeaway_slide(prs, layout, 10, total)
    limits_slide(prs, layout, 11, total)
    prs.save(OUT)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
