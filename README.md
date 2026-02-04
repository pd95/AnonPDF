# AnonPDF

AnonPDF is a small CLI utility that replaces sensitive text inside PDF content streams without rasterizing pages. It rewrites text operators and keeps vector graphics intact.

## Installation

```bash
python -m pip install -r requirements.txt
```

Minimal runtime dependency is `pypdf`.
For the redaction workflow described below, you also need `pdftotext` (Poppler utilities) and `qpdf` available on your PATH.

## Usage

```bash
python anon_pdf.py <INPUT> [--output=<FILE>] [<verbatim words...>] [--regex <regex patterns...>]
```

If `--output` is omitted, the input file is overwritten (safely, via temp file + replace).
Replacement behavior and matching defaults are hardcoded in `anon_pdf.py`.

### Examples

Verbatim replacements only:

```bash
python anon_pdf.py fixtures/sample.pdf "Zürich" "Öl"
```

Verbatim + regex (regex applied after verbatim):

```bash
python anon_pdf.py fixtures/sample.pdf "Zürich" "Öl" --regex "\+41 \d{2} \d{3} \d{2} \d{2}" "80\d\d Zürich"
```

Dry-run with per-pattern counts:

```bash
python anon_pdf.py fixtures/sample.pdf "Zürich" "Öl" --regex "\+41 \d{2} \d{3} \d{2} \d{2}" "80\d\d Zürich" --dry-run
```

Custom regex flags:

```bash
python anon_pdf.py fixtures/sample.pdf --regex "\+41 \d{2} \d{3} \d{2} \d{2}" "80\d\d Zürich" --regex-flags "iu"
```

## Options

- `--output=<FILE>` / `--output <FILE>`: Output PDF path. Defaults to overwriting input.
- `--regex <patterns...>`: Regex patterns (case-insensitive + Unicode-aware by default). Use a single `--regex` followed by one or more patterns (e.g., `--regex "pat1" "pat2"`). Repeating `--regex` is not supported and only the last occurrence is used.
- `--regex-flags <flags>`: Regex flags as letters: `i`, `m`, `s`, `u`. Default: `iu`.
- `--dry-run`: Do not write output. Reports total and per-pattern counts.

### Defaults (hardcoded)

- Unicode normalization: `NFD` (helps match decomposed accents like `u` + U+0308).
- Replacement mode: fixed; replacement character: `x`.
- Replacement fallbacks if `x` is not encodable: `x # * - .`.
- Match across adjacent text operands: enabled; joiner: `space`.
- Whitespace normalization: enabled.

## How it works

- Parses each page’s content stream.
- Tracks the current font (via `Tf` operators) and uses the font’s `/ToUnicode` or encoding map to decode glyph bytes into Unicode.
- Performs replacements on the decoded Unicode text.
- Re-encodes the updated text back into the original font encoding and writes it into the PDF content stream.
- Also processes Form XObjects, which commonly contain text.

## Limitations (PDF realities)

- By default, matches across adjacent text operands.
- If a font lacks a valid `/ToUnicode` map or uses a custom encoding without a reversible mapping, replacements may be incomplete.
- Text that is outlined to vector paths or embedded as images cannot be replaced by this tool.
- Annotation appearance streams are not yet processed.

## Test Fixtures

Generate PDF fixtures for local testing:

```bash
python gen_fixtures.py
```

This writes PDFs into `fixtures/`.

## How to redact PDFs

These lessons are meant to reduce surprises when working with PDF text streams.

### Tools used

These steps generally rely on three capabilities:

1. Text extraction from PDFs (using `pdftotext` from `poppler`).
2. PDF linearization/repairing (to make subsequent rewriting predictable) (using `qpdf`)
3. Targeted redaction/rewrite capability (using `anon_pdf.py` from this project)

### Recommended workflow

1. Extract text to discover what actually exists in the PDF text layer (keep layout to "see" how text correlates):

   ```bash
   pdftotext -layout input.pdf -
   ```

2. Identify verbatim terms and regex patterns to redact (names, addresses, IDs).

3. Linearize the PDF before rewriting to avoid structural surprises during modification:

   ```bash
   qpdf --linearize input.pdf _linearized.pdf
   ```

4. Run a dry-run match report until your terms and patterns hit:

   ```bash
   python anon_pdf.py _linearized.pdf --dry-run "John" "Doe" --regex "80\\d\\d Zürich" "Account\\s+\\d+"
   ```

5. Produce the redacted output PDF:

   ```bash
   python anon_pdf.py _linearized.pdf --output=redacted.pdf "John" "Doe" --regex "80\\d\\d Zürich" "Account\\s+\\d+"
   ```

6. Verify by extracting text again and scanning for any sensitive terms: (we can omit `-layout` as we only care about the plain text)

   ```bash
   pdftotext redacted.pdf -
   ```

   If you still see sensitive text after verification, it may be in annotations or images.

7. Remove temporary files (linearized copy and any extracted text output you saved) after verification.

### Practical lessons

- If a term does not match in a dry-run, adjust it before writing output.
- Literal names often need flexible whitespace: try `John\\s+Doe` when `John Doe` fails.
- Always check the output PDF for missed terms; PDFs frequently split text in unexpected ways.
- Prefer writing the redacted PDF next to the original with a `*-redacted.pdf`.

## License

MIT License.
