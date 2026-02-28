# EPUB and SRT Translator with LLM

A desktop application for translating **EPUB**, **SRT**, and **TXT** files using local or cloud-hosted language models. Written in Python with a PyQt6 interface.

<img width="1504" height="936" alt="translator" src="https://github.com/user-attachments/assets/8c0dac5b-82f8-46eb-afa9-c61f7d6831bb" />

---

## Supported formats

| Format | Notes |
|--------|-------|
| EPUB | Full HTML structure preserved; two processing modes (see below) |
| SRT | Subtitle blocks with timestamps; translated lines are split proportionally to match original line count |
| TXT | Plain text, paragraph-by-paragraph |

---

## LLM backends

| Backend | Connection | Notes |
|---------|------------|-------|
| LM Studio | `http://localhost:1234/v1/chat/completions` | Default. Any model loaded in LM Studio. |
| Ollama | `http://localhost:11434/api/generate` | Requires model name (e.g. `llama3.2:3b`) |
| OpenRouter | Cloud API | Requires API key and model name (e.g. `openai/gpt-4o`, `openai/gpt-4o:free`) |

OpenRouter requests include automatic rate-limit detection and retry with backoff.

**Quick translation** (no LLM required):

| Service | Notes |
|---------|-------|
| Google Translate | Free, via `deep-translator` |
| DeepL Free | API key required |
| DeepL Pro | API key required |

Quick translation supports single fragments or all checked fragments in bulk. A character-count warning is shown before bulk runs.

---

## EPUB processing

EPUB files contain HTML with inline formatting tags (`<i>`, `<b>`, `<span>`, `<em>`, `<strong>`, `<u>`, `<sup>`, `<sub>`, `<small>`) that should not be sent to an LLM as raw HTML. The application replaces them with numbered placeholders before translation and restores them afterward.

### Inline mode (default)

All inline formatting tags are replaced with paired placeholders: `<p_01>...</p_01>`, `<p_02>...</p_02>`, etc.  
Non-translatable content (padding spaces, empty anchors) is marked as `<nt_01/>`.  
Structural elements (`img`, `code`, `br`, `hr`, `kbd`, `abbr`, `wbr`, `var`, `canvas`, `svg`, `script`, `style`, `math`) become `<id_01>` reserve markers.

After translation, placeholders are resolved back to the original tags in their correct positions.

### Legacy mode

Only structural reserve elements are protected (`<id_xx>`). Inline tags are not replaced with placeholders and are instead re-inserted after translation using a multilingual alignment model (see [Tag alignment](#tag-alignment-epub-legacy-mode)).

### Tags that can be individually skipped

In Inline mode, specific tags can be excluded from placeholder substitution if an EPUB has broken or unreliable markup:  
`<span>` · `<i>` · `<b>` · `<em>` · `<strong>` · `<u>` · `<sup>` · `<sub>` · `<small>`

### Paragraph structure

Multi-paragraph elements can optionally preserve their internal paragraph breaks:
- **Inline mode:** breaks are sent to the LLM as `<ps>` markers and restored after translation.
- **Legacy / TXT:** breaks are restored by proportional word-count split after translation.

When disabled, all newlines within a fragment are flattened before sending.

### Special cases handled on save

- **Drop cap spans** (`span.first-letter`) — the first character is separated from the rest of the element text and wrapped in the correct span after translation.
- **Last-word spans** (`span.last-word`) — the last word is kept in its span after the translated text is inserted.

---

## Tag alignment (EPUB Legacy mode)

When saving an EPUB translated in Legacy mode, the application can run a multilingual transformer model to compute word-level semantic embeddings for the original and translated text, find word correspondences, and insert inline tags (`<i>`, `<b>`, `<span>`, etc.) at the correct positions in the translation.

**Supported models (downloaded from HuggingFace Hub, stored locally):**

| Model | Size | Notes |
|-------|------|-------|
| `bert-base-multilingual-cased` | ~700 MB | CPU-friendly |
| `microsoft/mdeberta-v3-base` | ~1.0 GB | CPU-friendly |
| `xlm-roberta-base` | ~1.1 GB | |
| `xlm-roberta-large` | ~2.4 GB | Default. GPU recommended. |

Models are stored in `<app_directory>/models/<model_name>/`. Each model has its own subfolder; changing the model name does not overwrite previously downloaded models. CUDA is supported.

The alignment step runs after translation, as a batch process at save time. Paragraphs without inline tags and those containing reserve elements are skipped automatically.

---

## Mismatch detection

After each translation the application runs a series of checks and flags the fragment if any check fails. Flagged fragments are shown in red in the list.

| Check | Description | Configurable |
|-------|-------------|:---:|
| Paragraph / line count | Number of paragraphs differs between original and translation | — |
| First character type | Type of the leading character changed (uppercase, lowercase, digit, quote, special) | — |
| Last character type | Ending punctuation type changed | — |
| Length ratio | Translation length is disproportionate to the original | ✓ |
| Quote parity | Odd number of double quotation marks in the translation | — |
| Untranslated | Translation is identical to the original (ignores short texts, proper nouns, URLs, single tokens) | ✓ |
| Reserve elements `<id_xx>` | Structural placeholders missing, duplicated, or spurious closing tags added | — |
| NT markers `<nt_xx/>` | Non-translatable markers missing or extra (Inline mode only) | — |
| Inline formatting `<p_xx>` | Opening/closing tags missing, extra, or unpaired (Inline mode only) | — |

**Configurable thresholds:**

- Length ratio: separate thresholds for short (≤100 chars), medium (≤500 chars), and long (>500 chars) texts. Texts ≤20 chars are always skipped.
- Untranslated ratio: minimum fraction of lowercase-initial words required to trigger the check (default: 0.30).
- Tag position shift for `<id_xx>` / `<nt_xx/>`: default 0.15.
- Tag position shift for `<p_xx>` (inline formatting): default 0.30 — higher tolerance because translated words have different lengths.

A fragment can be manually marked as correct (suppresses mismatch flag) or flagged for review (force-mismatch).

---

## Auto-fix

If mismatch is detected and Auto-fix is enabled, the application retries the translation automatically. On each retry, the specific error details are appended to the prompt so the model can correct them. Temperature increases slightly on each attempt to discourage identical outputs. After all attempts are exhausted, the result with the fewest mismatch errors is returned.

- Max attempts: 1–10 (default: 3)
- Temperature increment: configurable per attempt

---

## Prompts

The application maintains separate prompt variants for each file type and processing mode:

| Variant | Used for |
|---------|----------|
| `epub_inline` | EPUB in Inline mode |
| `epub_legacy` | EPUB in Legacy mode |
| `srt` | SRT files |
| `txt` | TXT files |

For LM Studio and OpenRouter, prompts are split into **System**, **Assistant (context/instruction)**, and **User** roles.  
For Ollama, a single combined prompt is used.

**Single-prompt mode** merges all parts into one string — for instruct-only models that do not support role-based messages (e.g. Gemma instruct). Reduces "Channel Error"-type failures with some local models.

Prompts can be edited directly in the application's built-in editor, saved to disk as `.txt` files, reset to the last saved state, or hard-reset (deletes saved files and restores factory defaults).

---

## Context window

Each fragment is sent to the LLM with optional surrounding context:
- **Previous paragraphs:** 0–∞ (default: 3)
- **Next paragraphs:** 0–∞ (default: 2)

Context is provided as read-only reference — only the current fragment is translated.

---

## Session management

The current translation state can be saved to a JSON file and restored later. The session includes:
- All paragraphs (original text, translated text, mismatch flags, translation status)
- Translation settings (temperature, context window, processing mode, prompt variant)
- Custom prompts

On load, the application re-parses the original file to reconstruct the internal EPUB book object (required for saving) and remaps paragraph IDs by matching normalized original text.

Sessions are stored as plain JSON and are not tied to a specific file path — the original file location is confirmed on load via a file picker.

---

## Requirements

**Python 3.10+**

```
PyQt6
requests
lxml
ebooklib
deep-translator
deepl
openrouter
```

Optional — required only for tag alignment in EPUB Legacy mode:
```
transformers
torch
```

---

## Installation

```bash
git clone https://github.com/Mubumbutu/EPUB-SRT-LLM-Translator.git
cd EPUB-SRT-LLM-Translator
pip install -r requirements.txt
```

With CUDA support for alignment (optional):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Running

```bash
python app.py
```

Make sure your chosen LLM backend is running before starting a translation session.

---

## Workflow

```
Open file → Select fragments → Configure LLM → Translate → Review → Save file
```

1. **Open file** — EPUB, SRT, or TXT via `📂 Open File`.
2. **Options tab** — select backend, enter API keys or model name, click **Save Settings**.
3. **Select fragments** — checkboxes in the list; `Select All`, `Select Untranslated`, `Select Mismatch`, or Shift+click for range selection. Searchable by original or translated text.
4. **Translate** — `▶ Translate Selected`. Status bar shows fragment index, progress count, elapsed time, timeout, and auto-fix attempt number.
5. **Review** — click a fragment to see original and translation side by side. The translation panel is editable; changes are applied immediately.
6. **Handle mismatches** — red fragments have detected problems. Hover for details. Options: edit manually, re-translate, mark as correct (ignore), or flag for review.
7. **Save** — `💾 Save as New File`. For EPUB Legacy with translations, a dialog offers to run tag alignment before saving.

---

## Notes

- **OpenRouter free models (`:free`)** — require enabling *"Allow free endpoints to publish prompts"* in [OpenRouter privacy settings](https://openrouter.ai/settings/privacy). Without it, all requests return 404.
- **Alignment and VRAM** — the alignment model loads into the same GPU as the LLM. Shut down LM Studio or Ollama before running alignment to avoid out-of-memory errors.
- **Switching Inline ↔ Legacy** — if a file is already loaded, the mode change dialog offers to reload immediately. Translating without reloading uses the new prompts but the old placeholder structure.
- **Ruby annotations** — `<rt>` and `<rp>` tags (Japanese furigana) are stripped during EPUB parsing and not included in fragments.

---

## License

[GNU General Public License v3.0](LICENSE)
