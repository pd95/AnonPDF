#!/usr/bin/env python3
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
    regex_flags: int,
    dry_run: bool,
) -> tuple[List[int], List[int]]:
    reader = PdfReader(input_path)
    writer = PdfWriter()
    word_counts = [0] * len(words)
    regex_counts = [0] * len(regexes)

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
    args = parser.parse_args()

    output_path = args.output or args.input_pdf
    regex_flags = _regex_flags_from_string(args.regex_flags)
    word_counts, regex_counts = anonymize_pdf(
        args.input_pdf, output_path, args.words, args.regex, regex_flags, args.dry_run
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
