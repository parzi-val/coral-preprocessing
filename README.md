# Coral & Marine Taxonomic Figures Preprocessing & Curation Toolkit

A comprehensive pipeline and local curation tool for extracting, annotating, refining, and publishing high-resolution taxonomic figures and captions from marine biology literature.

---

## 🌟 Features

- **Interactive Local Curator Web App (`app.py` + `static/index.html`)**:
  - **Native PDF Review Mode**: Embedded vector PDF viewing with high-contrast active and context bounding boxes.
  - **300 DPI Seamless Recropper**: Drag-and-drop canvas cropper to recrop figures on demand.
  - **Page Overview View**: Bird's-eye book auditing across all 296 pages with interactive bounding boxes, species tags, and floating hover tooltips.
  - **Missed Pages Queue**: Fast flagging and bounding-box drawing for uncaptured figures with duplicate detection and deletion.
- **Scientific Caption Refinement & Taxonomic Extraction (`refine_captions.py`)**:
  - Multimodal LLM proofreading comparing captions against 150 DPI page bitmaps and PDF text streams.
  - Automatic correction of OCR glitches, line break cutoffs, and punctuation.
  - Structured extraction of verified Latin binomial nomenclature (`species_name`).
- **Publication-Ready Dataset Exporter (`build_dataset.py`)**:
  - Operates in **strictly read-only mode** on ground-truth curation files.
  - Exports clean, standardized 300 DPI crops (`dataset/images/`).
  - Generates Hugging Face Datasets `metadata.jsonl`, standard `metadata.json`, tabular `metadata.csv`, and Multimodal VLM Q&A instruction pairs (`vqa_conversations.jsonl`).
  - Calculates normalized bounding boxes `[0, 1000]` for modern Vision-Language Models (PaliGemma, LayoutLM, Florence-2, Qwen2-VL).

---

## 📁 Repository Structure

```text
├── app.py                      # FastAPI backend for local curation web app
├── static/
│   └── index.html              # Modern dark-mode QA & curation UI
│
├── refine_captions.py          # Multimodal LLM caption refinement & species extraction
├── CAPTION_REFINEMENT.md       # Comprehensive documentation for caption refinement
│
├── build_dataset.py            # Dataset builder & exporter (READ-ONLY on ground truth)
├── dataset/                    # Published final dataset
│   ├── README.md               # Dataset card, taxonomy stats, and usage code
│   ├── metadata.jsonl          # Hugging Face Datasets ready metadata
│   ├── metadata.json           # Structured JSON array
│   ├── metadata.csv            # Tabular format for pandas/Excel
│   ├── vqa_conversations.jsonl # Multimodal VLM instruction tuning format
│   └── images/                 # Standardized 300 DPI high-resolution crops
│
├── output/                     # Curation ground-truth storage
│   ├── figures.json            # Active ground truth metadata
│   ├── figures.csv             # Ground truth tabular export
│   ├── flagged_pages.json      # Missed pages queue state
│   ├── images/                 # 300 DPI original crops
│   └── figures.backup.json     # Safety snapshots
│
├── index.pdf                   # Source textbook PDF
├── .env.example                # Environment variables template
└── .gitignore                  # Git ignore rules (protects .env secrets)
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/parzi-val/coral-preprocessing.git
cd coral-preprocessing
pip install fastapi uvicorn pymupdf pillow pandas tqdm openai python-dotenv
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your LLM credentials:
```bash
cp .env.example .env
```

### 3. Launch the Curator Web App
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 4. Build or Refresh the Dataset
Export the publication dataset at any time:
```bash
python build_dataset.py
```

---

## 📜 Dataset Usage

### With Hugging Face `datasets`
```python
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="dataset")
print(dataset["train"][0])
```

### With Pandas
```python
import pandas as pd

df = pd.read_csv("dataset/metadata.csv")
print(df.head())
```

---

## 📄 License
This project is licensed under the MIT License.
