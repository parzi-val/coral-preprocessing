# Scientific Caption Refinement & Species Extraction Guide

`refine_captions.py` is a self-contained, multimodal proofreading script that uses an LLM (such as Gemini 2.5 Flash) to audit, correct, and enrich extracted figure captions by comparing them directly against the high-resolution source PDF page.

---

## 1. What It Does

1. **Multimodal OCR & Layout Verification**:
   - Renders each source PDF page at 150 DPI and passes both the page bitmap and extracted PDF text streams to the model.
2. **Scientific & Taxonomic Correction**:
   - Fixes OCR typos (e.g., `"tbeir"` → `"their"`, `"Fi,g."` → `"Fig."`).
   - Corrects garbled figure numbering and punctuation.
   - Restores missing sentences, sub-figure labels `(a, b, c)`, and author citations.
   - Strips extraneous surrounding book prose that accidentally leaked into the caption box.
3. **Structured Species Name Extraction**:
   - Extracts verified Latin binomials (e.g., *Acropora cytherea*, *Dardanus megistos*, *Sabella melanostigma*) into a dedicated `species_name` field.
   - Sets non-taxonomic figures (e.g., habitat diagrams, anatomy charts) to `null`.
4. **Safety & Zero Image Modification**:
   - Only operates on text metadata (`caption`, `fig_num`, `species_name`).
   - Never touches or alters image crops or coordinate bounding boxes.

---

## 2. Prerequisites & Environment Setup

### Install Dependencies
```bash
pip install pymupdf openai pandas tqdm python-dotenv
```

### Configure `.env`
Ensure your `.env` file in the project root defines your LLM proxy or API endpoint:
```ini
OPENAI_BASE_URL=http://127.0.0.1:8787/v1
OPENAI_API_KEY=your_api_key_or_dummy
LLM_MODEL=gemini-2.5-flash
```

---

## 3. Command-Line Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--pdf` | `index.pdf` | Path to the source textbook PDF |
| `--metadata` | `output/figures.json` | Path to the figure metadata ground truth |
| `--output-json` | `output/refined_captions.json` | Detailed per-figure audit report in JSON |
| `--output-csv` | `output/refined_captions.csv` | Summary tabular report for quick CSV inspection |
| `--apply` | *(disabled)* | **Directly applies** verified changes back into `figures.json` |
| `--start-page` | `None` | Filter: start at this page number (inclusive) |
| `--end-page` | `None` | Filter: end at this page number (inclusive) |
| `--pages` | `None` | Comma-separated page numbers (e.g. `26,59,60`) or `'manual'` to target pages with manual figures |
| `--only-manual` | *(disabled)* | Only apply updates to manual added figures (`method == "manual_added"`) |
| `--limit-pages` | `None` | Limit total number of pages to process |

---

## 4. Usage Workflows

### A. Dry-Run Audit (Recommended First Step)
To test on a handful of pages without modifying `figures.json`:
```bash
python refine_captions.py --start-page 21 --end-page 25
```
* Generates `output/refined_captions.json` and `output/refined_captions.csv`.
* Displays a side-by-side comparison of original vs. cleaned captions.
* Leaves `output/figures.json` completely untouched.

### B. Inspect Changes
Open `output/refined_captions.csv` in Excel or pandas to review:
- `original_caption` vs. `cleaned_caption`
- `species_name`
- `has_changes` and `change_summary`
- `confidence`

### C. Full Production Run (Apply Mode)
To refine all pages and save the polished captions and species names directly into `output/figures.json`:
```bash
python refine_captions.py --apply
```
* Saves progress incrementally after every page, so it can be stopped or resumed safely at any time.

---

## 5. Safe Practice Checklist
- [x] Ensure a snapshot backup exists before running with `--apply` (e.g., `output/figures.backup.json` or `output/figures.curated_snapshot.json`).
- [x] Check progress stats in terminal (`tqdm` progress bar with error tolerance and auto-retries).
- [x] Launch the local web app (`python -m uvicorn app:app --port 8000`) to inspect the refined captions in the Review and Page Overview tabs.
