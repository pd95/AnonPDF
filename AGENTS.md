# AGENTS

## Project summary
AnonPDF rewrites PDF content streams to replace sensitive text with same-length characters. It avoids rasterization and keeps vector content intact. Regex support and dry-run reporting are implemented.

## Files
- `anon_pdf.py`: main CLI and logic.
- `gen_fixtures.py`: generates sample PDF fixtures in `AnonPDF/fixtures/`.
- `README.md`: usage and limitations.

## Current CLI
```
python anon_pdf.py <INPUT> [--output <FILE>] [<verbatim words...>] [--regex <regex patterns...>]
```
Key flags:
- `--regex-flags` (default `iu`)
- `--dry-run` (prints total and per-pattern counts)
- `--replacement-char` (default `x`)
- `--replacement-mode fixed|first-letter`
- `--match-across-operators` (default on) / `--no-match-across-operators`
- `--match-joiner space|none`

## How replacements work
- Parse page content streams.
- Track current font via `Tf` operator.
- Decode glyph bytes using `/ToUnicode` or encoding map (via `pypdf._cmap.build_char_map_from_dict`).
- Apply verbatim replacements, then regex replacements.
- Re-encode to original font encoding and write back.
- Also processes Form XObjects.

## Known limitations / open points
- **Split text runs:** Cross-operator matching is best-effort. It may miss cases with complex spacing or custom positioning, especially across distinct text blocks.
- **Annotations / appearance streams:** Not processed. Many PDFs store visible text in `/AP` streams (e.g., form fields).
- **Fonts without `/ToUnicode`:** Decoding/re-encoding may be incomplete; could add best-effort byte-level matching.

## Suggested next steps
- Add annotation appearance stream processing.
- Improve cross-operator matching with position-aware spacing or re-chunking.
- Provide JSON output for `--dry-run`.

## Testing
Quick end-to-end verification:
```
python gen_fixtures.py
python anon_pdf.py fixtures/sample.pdf "Zürich" --dry-run
python anon_pdf.py fixtures/sample.pdf "Zürich" --output /workdir/anonpdf-check.pdf
pdftotext /workdir/anonpdf-check.pdf - | sed -n '1,40p'
python anon_pdf.py fixtures/address_split.pdf "Zürich" --dry-run
python anon_pdf.py fixtures/address_split.pdf "Zürich" --output /workdir/anonpdf-address-split.pdf
pdftotext /workdir/anonpdf-address-split.pdf - | sed -n '1,20p'
```
Expected: `Zürich` should appear as `xxxxxx` in the extracted text on both pages. Remove the temp file afterward.

## Environment notes
- Requires `pypdf`. This is already installed in this environment.
- Generated fixtures are in `AnonPDF/fixtures/`.
