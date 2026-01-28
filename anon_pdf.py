#!/usr/bin/env python3
"""
AnonPDF: in-place PDF text anonymization without rasterization.

This CLI rewrites PDF content streams and replaces matching text with same-length
characters. It preserves vector graphics and layout by modifying text operands
inside page content streams and Form XObjects.

Usage:
  python anon_pdf.py <INPUT> [--output <FILE>] [<verbatim words...>] [--regex <patterns...>]

Replacement behavior:
  - fixed (default): repeat a single character, e.g. "Zürich" -> "xxxxxx"
  - first-letter: repeat the first character of each match, e.g. "Zürich" -> "ZZZZZZ"

Notes:
  - Matches are applied to decoded text (via /ToUnicode or font encodings).
  - Regex replacements are applied after verbatim word replacements.
  - Matches cross adjacent text operands by default; disable with --no-match-across-operators.
"""
import argparse
import os
import re
import tempfile
from typing import List, Tuple
from pypdf import PdfReader, PdfWriter
from pypdf._cmap import build_char_map_from_dict
from pypdf.generic import (
    ContentStream,
    NameObject,
    TextStringObject,
    ByteStringObject,
    IndirectObject,
)


def _replacement_for_match(match_text: str, mode: str, fixed_char: str) -> str:
    if not match_text:
        return ""
    if mode == "first-letter":
        repl_char = match_text[0]
    else:
        repl_char = fixed_char
    return repl_char * len(match_text)


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
    replace_mode: str,
    replace_char: str,
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
            repl = _replacement_for_match(needle, replace_mode, replace_char)
            result = result[:idx] + repl + result[idx + len(needle) :]
            start = idx + len(repl)
            word_counts[idx_word] += 1
    for idx_re, needle in enumerate(regexes):
        if not needle:
            continue
        pattern = re.compile(needle, flags=regex_flags)
        result, n = pattern.subn(
            lambda m: _replacement_for_match(m.group(0), replace_mode, replace_char),
            result,
        )
        regex_counts[idx_re] += n
    return result, word_counts, regex_counts


def _replace_in_text_operand(
    operand,
    words: List[str],
    regexes: List[str],
    regex_flags: int,
    replace_mode: str,
    replace_char: str,
):
    if isinstance(operand, TextStringObject):
        text = str(operand)
        replaced, w_counts, r_counts = _replace_all(
            text, words, regexes, regex_flags, replace_mode, replace_char
        )
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
        replaced, w_counts, r_counts = _replace_all(
            text, words, regexes, regex_flags, replace_mode, replace_char
        )
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
    replace_mode: str,
    replace_char: str,
    match_across_operators: bool = False,
    match_joiner: str = "space",
) -> tuple[ContentStream, List[int], List[int]]:
    content = ContentStream(stream_obj, reader)
    new_ops = []
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)
    current_font = None
    font_cache: dict[str, tuple[object, dict, dict[str, bytes]]] = {}
    if match_joiner == "space":
        joiner = " "
    else:
        joiner = ""

    def decode_item(item) -> tuple[str, callable] | None:
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
                    return None
                decoded = _decode_text_bytes(raw, font_encoding, font_map)

                def encoder(new_text: str):
                    new_raw = _encode_unicode(new_text, font_encoding, font_glyph_byte_map)
                    return ByteStringObject(new_raw)

                return decoded, encoder
        if isinstance(item, TextStringObject):
            return str(item), lambda s: TextStringObject(s)
        if isinstance(item, ByteStringObject):
            try:
                text = bytes(item).decode("latin-1")
            except Exception:
                return None
            return text, lambda s: ByteStringObject(s.encode("latin-1"))
        return None

    if match_across_operators:
        segments: list[dict] = []
        segment_runs: list[int] = []
        run_id = 0
        text_related_ops = {
            "BT",
            "ET",
            "Tf",
            "Tm",
            "Td",
            "TD",
            "T*",
            "Tc",
            "Tw",
            "Tz",
            "TL",
            "Ts",
            "Tr",
            "Tj",
            "TJ",
            "'",
            '"',
        }
        last_font = None
        for operands, operator in content.operations:
            op = operator.decode("utf-8") if isinstance(operator, bytes) else operator
            if op not in text_related_ops:
                run_id += 1
            if op == "Tf" and operands:
                current_font = str(operands[0])
            if op in ("Tj", "'", '"'):
                if operands:
                    decoded = decode_item(operands[0])
                    if decoded:
                        text, encoder = decoded
                        segments.append(
                            {
                                "text": text,
                                "set": lambda s, ops=operands, enc=encoder: ops.__setitem__(0, enc(s)),
                                "font": current_font,
                            }
                        )
                        if last_font is not None and current_font != last_font:
                            run_id += 1
                        segment_runs.append(run_id)
                        last_font = current_font
            elif op == "TJ":
                if operands and hasattr(operands[0], "__iter__"):
                    arr = operands[0]
                    for idx, item in enumerate(arr):
                        decoded = decode_item(item)
                        if decoded:
                            text, encoder = decoded
                            segments.append(
                                {
                                    "text": text,
                                    "set": lambda s, a=arr, i=idx, enc=encoder: a.__setitem__(i, enc(s)),
                                    "font": current_font,
                                }
                            )
                            if last_font is not None and current_font != last_font:
                                run_id += 1
                            segment_runs.append(run_id)
                            last_font = current_font
            new_ops.append((operands, operator))

        # Run replacements across segments, grouped by run_id.
        for run in sorted(set(segment_runs)):
            run_indices = [i for i, r in enumerate(segment_runs) if r == run]
            if not run_indices:
                continue
            run_segments = [segments[i] for i in run_indices if segments[i]["text"]]
            if not run_segments:
                continue
            virtual_chars: list[str] = []
            mapping: list[tuple[int, int] | None] = []
            for si, seg in enumerate(run_segments):
                for ci, ch in enumerate(seg["text"]):
                    virtual_chars.append(ch)
                    mapping.append((si, ci))
                if joiner and si != len(run_segments) - 1:
                    virtual_chars.append(joiner)
                    mapping.append(None)
            virtual_text = "".join(virtual_chars)
            replaced, w_counts, r_counts = _replace_all(
                virtual_text, words, regexes, regex_flags, replace_mode, replace_char
            )
            if replaced != virtual_text:
                seg_chars = [list(seg["text"]) for seg in run_segments]
                changed = set()
                for idx, map_entry in enumerate(mapping):
                    if map_entry is None:
                        continue
                    si, ci = map_entry
                    new_ch = replaced[idx]
                    if seg_chars[si][ci] != new_ch:
                        seg_chars[si][ci] = new_ch
                        changed.add(si)
                for si in changed:
                    run_segments[si]["set"]("".join(seg_chars[si]))
                    run_segments[si]["text"] = "".join(seg_chars[si])
                word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
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
                            replaced, w_counts, r_counts = _replace_all(
                                decoded, words, regexes, regex_flags, replace_mode, replace_char
                            )
                            if replaced != decoded:
                                new_raw = _encode_unicode(replaced, font_encoding, font_glyph_byte_map)
                                operands[0] = ByteStringObject(new_raw)
                                word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                                regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                        else:
                            new_val, w_counts, r_counts = _replace_in_text_operand(
                                operands[0], words, regexes, regex_flags, replace_mode, replace_char
                            )
                            operands[0] = new_val
                            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                    else:
                        new_val, w_counts, r_counts = _replace_in_text_operand(
                            operands[0], words, regexes, regex_flags, replace_mode, replace_char
                        )
                        operands[0] = new_val
                        word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                        regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                else:
                    new_val, w_counts, r_counts = _replace_in_text_operand(
                        operands[0], words, regexes, regex_flags, replace_mode, replace_char
                    )
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
                                replaced, w_counts, r_counts = _replace_all(
                                    decoded, words, regexes, regex_flags, replace_mode, replace_char
                                )
                                if replaced != decoded:
                                    new_raw = _encode_unicode(replaced, font_encoding, font_glyph_byte_map)
                                    arr[idx] = ByteStringObject(new_raw)
                                    word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                                    regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                                else:
                                    arr[idx] = item
                            else:
                                new_val, w_counts, r_counts = _replace_in_text_operand(
                                    item, words, regexes, regex_flags, replace_mode, replace_char
                                )
                                arr[idx] = new_val
                                word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                                regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                        else:
                            new_val, w_counts, r_counts = _replace_in_text_operand(
                                item, words, regexes, regex_flags, replace_mode, replace_char
                            )
                            arr[idx] = new_val
                            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
                            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
                    else:
                        new_val, w_counts, r_counts = _replace_in_text_operand(
                            item, words, regexes, regex_flags, replace_mode, replace_char
                        )
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
    replace_mode: str = "fixed",
    replace_char: str = "x",
    match_across_operators: bool = False,
    match_joiner: str = "space",
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
                xobj,
                reader,
                words,
                regexes,
                xobj.get("/Resources") or resources,
                regex_flags,
                replace_mode,
                replace_char,
                match_across_operators,
                match_joiner,
            )
            xobj.set_data(content.get_data())
            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
            # Recurse into nested resources.
            w_counts, r_counts = _process_xobjects(
                xobj.get("/Resources"),
                reader,
                words,
                regexes,
                visited,
                parent_resources=resources,
                regex_flags=regex_flags,
                replace_mode=replace_mode,
                replace_char=replace_char,
                match_across_operators=match_across_operators,
                match_joiner=match_joiner,
            )
            word_counts = [a + b for a, b in zip(word_counts, w_counts)]
            regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
    return word_counts, regex_counts


def anonymize_pdf(
    input_path: str,
    output_path: str,
    words: List[str],
    regexes: List[str],
    regex_flags: int,
    dry_run: bool,
    replace_mode: str,
    replace_char: str,
    match_across_operators: bool = False,
    match_joiner: str = "space",
) -> tuple[List[int], List[int]]:
    reader = PdfReader(input_path)
    writer = PdfWriter()
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)

    for i, base_page in enumerate(reader.pages):
        # Page content stream
        content, w_counts, r_counts = _process_content_stream(
            base_page.get("/Contents"),
            reader,
            words,
            regexes,
            base_page.get("/Resources"),
            regex_flags,
            replace_mode,
            replace_char,
            match_across_operators,
            match_joiner,
        )
        base_page[NameObject("/Contents")] = content
        word_counts = [a + b for a, b in zip(word_counts, w_counts)]
        regex_counts = [a + b for a, b in zip(regex_counts, r_counts)]
        # XObject form streams (common for text)
        w_counts, r_counts = _process_xobjects(
            base_page.get("/Resources"),
            reader,
            words,
            regexes,
            visited=set(),
            regex_flags=regex_flags,
            replace_mode=replace_mode,
            replace_char=replace_char,
            match_across_operators=match_across_operators,
            match_joiner=match_joiner,
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
        description=(
            "Rewrite PDF content streams to replace matching text with same-length characters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python anon_pdf.py fixtures/sample.pdf \"Zürich\"\n"
            "  python anon_pdf.py fixtures/sample.pdf \"Zürich\" \"Öl\" --dry-run\n"
            "  python anon_pdf.py fixtures/sample.pdf --regex \"naïve\" --regex-flags iu\n"
            "  python anon_pdf.py fixtures/sample.pdf \"Zürich\" --replacement-char \"*\"\n"
            "  python anon_pdf.py fixtures/sample.pdf \"Zürich\" --replacement-mode first-letter\n"
        ),
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
        "--replacement-char",
        default="x",
        help="Single character used for replacements in fixed mode (default: x)",
    )
    parser.add_argument(
        "--replacement-mode",
        choices=["fixed", "first-letter"],
        default="fixed",
        help="fixed = repeat replacement char; first-letter = repeat first character of each match",
    )
    parser.add_argument(
        "--match-across-operators",
        action="store_true",
        default=True,
        help="Match across adjacent text operators and text objects (default: enabled)",
    )
    parser.add_argument(
        "--no-match-across-operators",
        dest="match_across_operators",
        action="store_false",
        help="Disable cross-operator matching (revert to per-operand matching)",
    )
    parser.add_argument(
        "--match-joiner",
        choices=["space", "none"],
        default="space",
        help="Virtual joiner between adjacent text operands when matching across operators",
    )
    args = parser.parse_args()
    if args.replacement_mode == "fixed" and len(args.replacement_char) != 1:
        parser.error("--replacement-char must be a single character")

    output_path = args.output or args.input_pdf
    regex_flags = _regex_flags_from_string(args.regex_flags)
    word_counts, regex_counts = anonymize_pdf(
        args.input_pdf,
        output_path,
        args.words,
        args.regex,
        regex_flags,
        args.dry_run,
        args.replacement_mode,
        args.replacement_char,
        match_across_operators=args.match_across_operators,
        match_joiner=args.match_joiner,
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
