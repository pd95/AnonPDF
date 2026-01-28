#!/usr/bin/env python
import base64
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"

UNICODE_SENTENCE = "Zürich, Straße, Öl, € – naïve façade"


def write_dummy_png(path: Path) -> None:
    # 10x10 blue PNG
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFElEQVR4nGNkSDvDgBsw"
        "4ZEbwdIAfL0BRupNMTYAAAAASUVORK5CYII="
    )
    path.write_bytes(base64.b64decode(png_b64))


def _draw_words(c, x: float, y: float, words: list[str], font_name: str, font_size: int) -> None:
    cursor = x
    space_w = pdfmetrics.stringWidth(" ", font_name, font_size)
    for idx, word in enumerate(words):
        c.drawString(cursor, y, word)
        word_w = pdfmetrics.stringWidth(word, font_name, font_size)
        cursor += word_w
        if idx != len(words) - 1:
            cursor += space_w


def _draw_address_block(c, x: float, y: float, font_name: str = "DejaVuSans", font_size: int = 12) -> float:
    c.setFont(font_name, font_size)
    line_h = 14
    _draw_words(c, x, y, ["Herr"], font_name, font_size)
    y -= line_h
    _draw_words(c, x, y, ["John", "Doe"], font_name, font_size)
    y -= line_h
    _draw_words(c, x, y, ["Hauptrasse", "67"], font_name, font_size)
    y -= line_h
    _draw_words(c, x, y, ["8001", "Zürich"], font_name, font_size)
    y -= 24
    return y


def generate_pdf(path: Path) -> None:
    rl_config.defaultCompression = 0
    font_path = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))

    c = canvas.Canvas(str(path), pagesize=LETTER, pageCompression=0)
    c.setTitle("Office Tool Test: PDF")
    c.setAuthor("OfficeToolTests")
    c.setSubject(UNICODE_SENTENCE)

    width, height = LETTER
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "Office Tool Test: PDF")
    y -= 24

    c.setFont("DejaVuSans", 12)
    c.drawString(72, y, UNICODE_SENTENCE)
    y -= 36

    y = _draw_address_block(c, 72, y)

    rows, cols = 10, 4
    cell_w, cell_h = 100, 18
    x0, y0 = 72, y - (rows * cell_h)
    c.setStrokeColor(colors.black)
    for r in range(rows + 1):
        c.line(x0, y0 + r * cell_h, x0 + cols * cell_w, y0 + r * cell_h)
    for c_idx in range(cols + 1):
        c.line(x0 + c_idx * cell_w, y0, x0 + c_idx * cell_w, y0 + rows * cell_h)

    c.setFont("Helvetica", 10)
    for r in range(rows):
        for col in range(cols):
            value = f"R{r+1}C{col+1}"
            if r == 6 and col == 2:
                value = "CELL_7_3_OK"
            c.drawString(x0 + 4 + col * cell_w, y0 + (rows - 1 - r) * cell_h + 4, value)

    c.showPage()
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, "Second page")
    c.setFont("DejaVuSans", 12)
    c.drawString(72, height - 96, UNICODE_SENTENCE)
    _draw_address_block(c, 72, height - 130)
    c.save()
    with path.open("ab") as f:
        f.write(("\n% UNICODE: " + UNICODE_SENTENCE + "\n").encode("utf-8"))


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    generate_pdf(FIXTURES / "sample.pdf")
    generate_pdf(FIXTURES / "address_split.pdf")
    print("Fixtures generated in", FIXTURES)


if __name__ == "__main__":
    main()
