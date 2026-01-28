# AnonPDF

AnonPDF is a small CLI utility that replaces sensitive text inside PDF content streams without rasterizing pages. It rewrites text operators and keeps vector graphics intact.

## Installation

```
python -m pip install -r requirements.txt
```
Minimal runtime dependency is `pypdf`.

## Usage

```
python anon_pdf.py <INPUT> [--output <FILE>] [<verbatim words...>] [--regex <regex patterns...>]
```

If `--output` is omitted, the input file is overwritten (safely, via temp file + replace).
Replacement behavior can be customized with `--replacement-char` and `--replacement-mode`.

### Examples

Verbatim replacements only:
```
python anon_pdf.py fixtures/sample.pdf "Zürich" "Öl"
```

Verbatim + regex (regex applied after verbatim):
```
python anon_pdf.py fixtures/sample.pdf "Zürich" "Öl" --regex "naïve"
```

Dry-run with per-pattern counts:
```
python anon_pdf.py fixtures/sample.pdf "Zürich" "Öl" --regex "naïve" --dry-run
```

Custom regex flags:
```
python anon_pdf.py fixtures/sample.pdf --regex "naïve" --regex-flags "iu"
```

Custom replacement character:
```
python anon_pdf.py fixtures/sample.pdf "Zürich" --replacement-char "*"
```

Derive replacements from the first letter of each match:
```
python anon_pdf.py fixtures/sample.pdf "Zürich" --replacement-mode first-letter
```

## Options

- `--output <FILE>`: Output PDF path. Defaults to overwriting input.
- `--regex <patterns...>`: Regex patterns (case-insensitive + Unicode-aware by default).
- `--regex-flags <flags>`: Regex flags as letters: `i`, `m`, `s`, `u`. Default: `iu`.
- `--dry-run`: Do not write output. Reports total and per-pattern counts.
- `--replacement-char <char>`: Replacement character in fixed mode (default: `x`).
- `--replacement-mode fixed|first-letter`: Choose fixed replacement or repeat the first character of each match.

## How it works

- Parses each page’s content stream.
- Tracks the current font (via `Tf` operators) and uses the font’s `/ToUnicode` or encoding map to decode glyph bytes into Unicode.
- Performs replacements on the decoded Unicode text.
- Re-encodes the updated text back into the original font encoding and writes it into the PDF content stream.
- Also processes Form XObjects, which commonly contain text.

## Limitations (PDF realities)

- Matches only within a single text operand. If text is split across multiple `TJ` chunks or text objects, the current version may miss it.
- If a font lacks a valid `/ToUnicode` map or uses a custom encoding without a reversible mapping, replacements may be incomplete.
- Text that is outlined to vector paths or embedded as images cannot be replaced by this tool.
- Annotation appearance streams are not yet processed.

## Test Fixtures

Generate PDF fixtures for local testing:
```
python gen_fixtures.py
```

This writes PDFs into `fixtures/`.

## License

Internal tool; no license specified.
