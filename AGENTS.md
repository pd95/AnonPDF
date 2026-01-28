# AGENTS

## Project summary
AnonPDF rewrites PDF content streams to replace sensitive text with same-length `x` characters. It avoids rasterization and keeps vector content intact. Regex support and dry-run reporting are implemented.

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

## How replacements work
- Parse page content streams.
- Track current font via `Tf` operator.
- Decode glyph bytes using `/ToUnicode` or encoding map (via `pypdf._cmap.build_char_map_from_dict`).
- Apply verbatim replacements, then regex replacements.
- Re-encode to original font encoding and write back.
- Also processes Form XObjects.

## Known limitations / open points
- **Split text runs:** Matches don’t cross `TJ` array boundaries or separate text objects. A future enhancement could merge `TJ` arrays for matching and re-chunk with original spacing.
- **Annotations / appearance streams:** Not processed. Many PDFs store visible text in `/AP` streams (e.g., form fields).
- **Fonts without `/ToUnicode`:** Decoding/re-encoding may be incomplete; could add best-effort byte-level matching.

## Suggested next steps
- Add annotation appearance stream processing.
- Add cross-`TJ` matching with safe re-chunking.
- Provide JSON output for `--dry-run`.

## Environment notes
- Requires `pypdf`. This is already installed in this environment.
- Generated fixtures are in `AnonPDF/fixtures/`.
