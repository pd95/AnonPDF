#!/usr/bin/env python3
import argparse
import io
import os
import re
import tempfile
from typing import Iterable, List, Tuple

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf._cmap import build_char_map_from_dict
from pypdf.generic import (
    ContentStream,
    NameObject,
    TextStringObject,
    ByteStringObject,
    IndirectObject,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _pick_font() -> str:
    # Prefer a Unicode-capable font if available.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont("AnonSans", path))
            return "AnonSans"
    return "Helvetica"


def _find_matches(chars: List[dict], needle: str) -> List[List[dict]]:
    text = "".join(c.get("text", "") for c in chars)
    matches = []
    start = 0
    nlen = len(needle)
    if nlen == 0:
        return matches
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        matches.append(chars[idx : idx + nlen])
        start = idx + nlen
    return matches


def _group_by_line(match_chars: List[dict], tol: float = 2.0) -> List[List[dict]]:
    groups: List[List[dict]] = []
    if not match_chars:
        return groups
    current = [match_chars[0]]
    current_top = match_chars[0].get("top", 0.0)
    for ch in match_chars[1:]:
        top = ch.get("top", 0.0)
        if abs(top - current_top) <= tol:
            current.append(ch)
        else:
            groups.append(current)
            current = [ch]
            current_top = top
    groups.append(current)
    return groups


def _group_bounds(group: List[dict]) -> Tuple[float, float, float, float, float, int]:
    x0 = min(c.get("x0", 0.0) for c in group)
    x1 = max(c.get("x1", 0.0) for c in group)
    top = min(c.get("top", 0.0) for c in group)
    bottom = max(c.get("bottom", 0.0) for c in group)
    sizes = [c.get("size") for c in group if c.get("size") is not None]
    size = float(sum(sizes) / len(sizes)) if sizes else max(1.0, bottom - top)
    return x0, x1, top, bottom, size, len(group)


def _draw_overlays(page_width: float, page_height: float, boxes: Iterable[Tuple[float, float, float, float, float, int]], font_name: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    for x0, x1, top, bottom, size, n_chars in boxes:
        width = max(0.0, x1 - x0)
        height = max(0.0, bottom - top)
        y_bottom = page_height - bottom
        # Cover original text.
        c.setFillColorRGB(1, 1, 1)
        c.rect(x0, y_bottom, width, height, stroke=0, fill=1)
        # Draw replacement x's.
        c.setFillColorRGB(0, 0, 0)
        c.setFont(font_name, max(1.0, size))
        text_y = y_bottom + max(0.0, (height - size) * 0.3)
        c.drawString(x0, text_y, "x" * n_chars)
    c.showPage()
    c.save()
    return buf.getvalue()

def _regex_flags_from_string(flag_string: str) -> int:
    flags = 0
    for ch in flag_string:
        if ch == "i":
            flags |= re.IGNORECASE
        elif ch == "m":
            flags |= re.MULTILINE
        elif ch == "s":
            flags |= re.DOTALL
        elif ch == "u":
            flags |= re.UNICODE
    return flags


def _replace_all(
    text: str,
    words: List[str],
    regexes: List[str],
    regex_flags: int,
) -> tuple[str, List[int], List[int]]:
    result = text
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)
    for idx_word, needle in enumerate(words):
        if not needle:
            continue
        start = 0
        while True:
            idx = result.find(needle, start)
            if idx == -1:
                break
            repl = "x" * len(needle)
            result = result[:idx] + repl + result[idx + len(needle) :]
            start = idx + len(repl)
            word_counts[idx_word] += 1
    for idx_re, needle in enumerate(regexes):
        if not needle:
            continue
        pattern = re.compile(needle, flags=regex_flags)
        result, n = pattern.subn(lambda m: "x" * len(m.group(0)), result)
        regex_counts[idx_re] += n
    return result, word_counts, regex_counts


def _replace_in_text_operand(operand, words: List[str], regexes: List[str], regex_flags: int):
    if isinstance(operand, TextStringObject):
        text = str(operand)
        replaced, w_counts, r_counts = _replace_all(text, words, regexes, regex_flags)
        if replaced != text:
            return TextStringObject(replaced), w_counts, r_counts
        return operand, [0] * len(words), [0] * len(regexes)
    if isinstance(operand, ByteStringObject):
        # Best-effort for byte strings: try latin-1 to preserve byte length.
        data = bytes(operand)
        try:
            text = data.decode("latin-1")
        except Exception:
            return operand, [0] * len(words), [0] * len(regexes)
        replaced, w_counts, r_counts = _replace_all(text, words, regexes, regex_flags)
        if replaced != text:
            return ByteStringObject(replaced.encode("latin-1")), w_counts, r_counts
        return operand, [0] * len(words), [0] * len(regexes)
    return operand, [0] * len(words), [0] * len(regexes)


def _decode_text_bytes(raw: bytes, font_encoding, font_map: dict) -> str:
    if font_map:
        bytes_per_char = font_map.get(-1, 1)
        out = []
        for i in range(0, len(raw), bytes_per_char):
            chunk = raw[i : i + bytes_per_char]
            if bytes_per_char == 1:
                key = chunk.decode("charmap", "surrogatepass")
            else:
                key = chunk.decode("utf-16-be", "surrogatepass")
            out.append(font_map.get(key, key))
        return "".join(out)
    if isinstance(font_encoding, dict):
        return "".join(font_encoding.get(b, "�") for b in raw)
    if isinstance(font_encoding, str):
        return raw.decode(font_encoding, "surrogatepass")
    return raw.decode("latin-1", "surrogatepass")


def _build_font_maps(font_resource) -> tuple[object, dict, dict[str, bytes]]:
    font_resource = font_resource.get_object() if isinstance(font_resource, IndirectObject) else font_resource
    _, _, font_encoding, font_map = build_char_map_from_dict(200, font_resource)
    # Remove sentinel if present
    font_map = {k: v for k, v in font_map.items() if k != -1}
    font_glyph_byte_map: dict[str, bytes] = {}
    if isinstance(font_encoding, str):
        for key, value in font_map.items():
            if not isinstance(value, str):
                continue
            try:
                font_glyph_byte_map[value] = key.encode(font_encoding)
            except Exception:
                pass
    else:
        for k, v in font_encoding.items():
            font_glyph_byte_map[v] = bytes((k,))
        font_encoding_rev = {v: bytes((k,)) for k, v in font_encoding.items()}
        for key, value in font_map.items():
            if not isinstance(value, str):
                continue
            if isinstance(key, str):
                font_glyph_byte_map[value] = font_encoding_rev.get(key, key.encode("latin-1", "ignore"))
    return font_encoding, font_map, font_glyph_byte_map


def _encode_unicode(text: str, font_encoding, font_glyph_byte_map: dict[str, bytes]) -> bytes:
    if isinstance(font_encoding, dict):
        rev = {v: bytes((k,)) for k, v in font_encoding.items()}
    else:
        rev = {}
    out = []
    for ch in text:
        if ch in font_glyph_byte_map:
            out.append(font_glyph_byte_map[ch])
        elif isinstance(font_encoding, str):
            out.append(ch.encode(font_encoding, "replace"))
        elif isinstance(font_encoding, dict):
            out.append(rev.get(ch, b"?"))
        else:
            out.append(ch.encode("latin-1", "replace"))
    return b"".join(out)


def _process_content_stream(
    stream_obj,
    reader,
    words: List[str],
    regexes: List[str],
    resources,
    regex_flags: int,
) -> tuple[ContentStream, List[int], List[int]]:
    content = ContentStream(stream_obj, reader)
    new_ops = []
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)
    current_font = None
    font_cache: dict[str, tuple[object, dict, dict[str, bytes]]] = {}
    for operands, operator in content.operations:
        op = operator.decode("utf-8") if isinstance(operator, bytes) else operator
        if op == "Tf" and operands:
            current_font = str(operands[0])
        if op in ("Tj", "'", '"'):
            if operands:
                if current_font and resources:
                    font_dict = resources.get("/Font", {})
                    font_res = font_dict.get(current_font)
                    if font_res:
                        if current_font not in font_cache:
                            font_cache[current_font] = _build_font_maps(font_res)
                        font_encoding, font_map, font_glyph_byte_map = font_cache[current_font]
                        if isinstance(operands[0], TextStringObject):
                            raw = str(operands[0]).encode("latin-1", "surrogatepass")
                        elif isinstance(operands[0], ByteStringObject):
                            raw = bytes(operands[0])
                        else:
                            raw = None
                        if raw is not None:
                            decoded = _decode_text_bytes(raw, font_encoding, font_map)
                            replaced, w_counts, r_counts = _replace_all(decoded, words, regexes, regex_flags)
                            if replaced != decoded:
                                new_raw = _encode_unicode(replaced, font_encoding, font_glyph_byte_map)
                                operands[0] = ByteStringObject(new_raw)
                                word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                                regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                        else:
                            new_val, w_counts, r_counts = _replace_in_text_operand(operands[0], words, regexes, regex_flags)
                            operands[0] = new_val
                            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                    else:
                        new_val, w_counts, r_counts = _replace_in_text_operand(operands[0], words, regexes, regex_flags)
                        operands[0] = new_val
                        word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                        regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                else:
                    new_val, w_counts, r_counts = _replace_in_text_operand(operands[0], words, regexes, regex_flags)
                    operands[0] = new_val
                    word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                    regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
        elif op == "TJ":
            if operands and hasattr(operands[0], "__iter__"):
                arr = operands[0]
                for idx, item in enumerate(arr):
                    if current_font and resources:
                        font_dict = resources.get("/Font", {})
                        font_res = font_dict.get(current_font)
                        if font_res:
                            if current_font not in font_cache:
                                font_cache[current_font] = _build_font_maps(font_res)
                            font_encoding, font_map, font_glyph_byte_map = font_cache[current_font]
                            if isinstance(item, TextStringObject):
                                raw = str(item).encode("latin-1", "surrogatepass")
                            elif isinstance(item, ByteStringObject):
                                raw = bytes(item)
                            else:
                                raw = None
                            if raw is not None:
                                decoded = _decode_text_bytes(raw, font_encoding, font_map)
                                replaced, w_counts, r_counts = _replace_all(decoded, words, regexes, regex_flags)
                                if replaced != decoded:
                                    new_raw = _encode_unicode(replaced, font_encoding, font_glyph_byte_map)
                                    arr[idx] = ByteStringObject(new_raw)
                                    word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                                    regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                                else:
                                    arr[idx] = item
                            else:
                                new_val, w_counts, r_counts = _replace_in_text_operand(item, words, regexes, regex_flags)
                                arr[idx] = new_val
                                word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                                regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                        else:
                            new_val, w_counts, r_counts = _replace_in_text_operand(item, words, regexes, regex_flags)
                            arr[idx] = new_val
                            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                    else:
                        new_val, w_counts, r_counts = _replace_in_text_operand(item, words, regexes, regex_flags)
                        arr[idx] = new_val
                        word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                        regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
        new_ops.append((operands, operator))
    content.operations = new_ops
    # Force serialization from operations.
    content._data = b""
    content._decoded_data = None
    content._encoded_data = None
    content._compress = None
    content._filters = None
    content._constants = None
    content._resolved_objects = None
    content._is_inline_image = False
    return content, word_counts, regex_counts


def _indirect_key(obj) -> Tuple[int, int] | None:
    if isinstance(obj, IndirectObject):
        return (obj.idnum, obj.generation)
    if hasattr(obj, "indirect_reference") and obj.indirect_reference is not None:
        ref = obj.indirect_reference
        return (ref.idnum, ref.generation)
    return None


def _process_xobjects(
    resources,
    reader,
    words: List[str],
    regexes: List[str],
    visited: set,
    parent_resources=None,
    regex_flags: int = 0,
) -> tuple[List[int], List[int]]:
    if resources is None:
        resources = parent_resources
    if resources is None:
        return [0] * len(words), [0] * len(regexes)
    if isinstance(resources, IndirectObject):
        resources = resources.get_object()
    xobjs = resources.get("/XObject")
    if xobjs is None:
        return [0] * len(words), [0] * len(regexes)
    if isinstance(xobjs, IndirectObject):
        xobjs = xobjs.get_object()
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)
    for _, xobj_ref in xobjs.items():
        xobj = xobj_ref
        if isinstance(xobj_ref, IndirectObject):
            key = _indirect_key(xobj_ref)
            if key and key in visited:
                continue
            if key:
                visited.add(key)
            xobj = xobj_ref.get_object()
        subtype = xobj.get("/Subtype")
        if subtype == "/Form":
            # Process the form XObject's content stream (the XObject itself is a stream).
            content, w_counts, r_counts = _process_content_stream(
                xobj, reader, words, regexes, xobj.get("/Resources") or resources, regex_flags
            )
            xobj.set_data(content.get_data())
            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
            # Recurse into nested resources.
            w_counts, r_counts = _process_xobjects(
                xobj.get("/Resources"), reader, words, regexes, visited, parent_resources=resources, regex_flags=regex_flags
            )
            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
    return word_counts, regex_counts


def anonymize_pdf(
    input_path: str,
    output_path: str,
    words: List[str],
    regexes: List[str],
    mode: str,
    regex_flags: int,
    dry_run: bool,
) -> tuple[List[int], List[int]]:
    font_name = _pick_font()
    reader = PdfReader(input_path)
    writer = PdfWriter()
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)

    if mode == "overlay":
        with pdfplumber.open(input_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_width = float(page.width)
                page_height = float(page.height)
                chars = page.chars or []
                boxes: List[Tuple[float, float, float, float, float, int]] = []
                for needle in words:
                    for match in _find_matches(chars, needle):
                        for group in _group_by_line(match):
                            boxes.append(_group_bounds(group))

                overlay_pdf = _draw_overlays(page_width, page_height, boxes, font_name)
                overlay_reader = PdfReader(io.BytesIO(overlay_pdf))
                overlay_page = overlay_reader.pages[0]
                base_page = reader.pages[i]
                if boxes:
                    base_page.merge_page(overlay_page)
                writer.add_page(base_page)
    else:
        for i, base_page in enumerate(reader.pages):
            # Page content stream
            content, w_counts, r_counts = _process_content_stream(
                base_page.get("/Contents"), reader, words, regexes, base_page.get("/Resources"), regex_flags
            )
            base_page[NameObject("/Contents")] = content
            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
            # XObject form streams (common for text)
            w_counts, r_counts = _process_xobjects(
                base_page.get("/Resources"), reader, words, regexes, visited=set(), regex_flags=regex_flags
            )
            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
            writer.add_page(base_page)

    if not dry_run:
        if os.path.abspath(output_path) == os.path.abspath(input_path):
            out_dir = os.path.dirname(os.path.abspath(output_path))
            with tempfile.NamedTemporaryFile(delete=False, dir=out_dir, suffix=".pdf") as tmp:
                writer.write(tmp)
                temp_path = tmp.name
            os.replace(temp_path, output_path)
        else:
            with open(output_path, "wb") as f:
                writer.write(f)
    return word_counts, regex_counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anonymize a PDF by replacing given strings with same-length x characters."
    )
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("--output", help="Output PDF path (defaults to overwrite input)")
    parser.add_argument("words", nargs="*", help="Verbatim strings to replace (Unicode supported)")
    parser.add_argument(
        "--regex",
        nargs="+",
        default=[],
        help="Regex patterns to replace (case-insensitive, Unicode-aware)",
    )
    parser.add_argument(
        "--regex-flags",
        default="iu",
        help="Regex flags as letters: i, m, s, u (default: iu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and report replacements, but do not write output",
    )
    parser.add_argument(
        "--mode",
        choices=["rewrite", "overlay"],
        default="rewrite",
        help="rewrite = replace text in PDF content streams; overlay = visual overlay",
    )
    args = parser.parse_args()

    output_path = args.output or args.input_pdf
    regex_flags = _regex_flags_from_string(args.regex_flags)
    word_counts, regex_counts = anonymize_pdf(
        args.input_pdf, output_path, args.words, args.regex, args.mode, regex_flags, args.dry_run
    )
    if args.dry_run:
        total = sum(word_counts) + sum(regex_counts)
        print(f"Dry-run: {total} replacements would be made.")
        if args.words:
            for needle, count in zip(args.words, word_counts):
                print(f'  word "{needle}": {count}')
        if args.regex:
            for pattern, count in zip(args.regex, regex_counts):
                print(f'  regex "{pattern}": {count}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
