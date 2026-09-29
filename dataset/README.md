# Coral & Marine Taxonomic Figures Dataset

A high-resolution, expert-curated dataset of marine and coral taxonomic figures, anatomical drawings, and biological field guide specimens extracted directly from scientific literature.

---

## Dataset Summary

| Metric | Value |
| :--- | :--- |
| **Total Figures** | **495** |
| **Source Pages** | **112** |
| **Verified Species Names** | **343** (69.3%) |
| **Unique Genera** | **248** |
| **Unique Species** | **393** |
| **Average Resolution** | **1154 x 737 px** |
| **Native Extraction DPI** | **300 DPI** (Lossless crops) |

---

## Directory Structure

```text
dataset/
├── README.md                  <- This Dataset Card
├── metadata.jsonl             <- Line-delimited JSON (Hugging Face & multimodal ready)
├── metadata.json              <- Full structured JSON array
├── metadata.csv               <- Tabular format for pandas/Excel
├── vqa_conversations.jsonl    <- Multimodal Q&A instruction format (LLaVA / Qwen-VL)
└── images/                    <- Clean, standardized 300 DPI image crops
    ├── coral_fig_001_p021.jpg
    ├── coral_fig_002_p021.jpg
    └── ...
```

---

## Taxonomic Breakdown

### Top Genera
| Genus | Count |
| :--- | :--- |
| *Acropora* | 20 |
| *Cypraea* | 13 |
| *Holothuria* | 11 |
| *Chaetodon* | 11 |
| *Acanthurus* | 7 |
| *Lutjanus* | 6 |
| *Actinopyga* | 5 |
| *Scarus* | 5 |
| *Arothron* | 5 |
| *Pocillopora* | 4 |

### Top Species
| Species | Count |
| :--- | :--- |
| *Acropora sp.* | 5 |
| *Lobophytum sp.* | 3 |
| *Cypraea sp.* | 3 |
| *Holothuria sp.* | 3 |
| *Actinopyga sp.* | 3 |
| *Arothron nigropunctatus* | 3 |
| *Turbinaria ornata* | 2 |
| *Haliclona sp.* | 2 |
| *Millepora sp.* | 2 |
| *Heteractis magnifica* | 2 |

---

## Schema Definition (`metadata.jsonl`)

Each line in `metadata.jsonl` contains:

```json
{
  "figure_id": "coral_fig_001",
  "image_path": "images/coral_fig_001_p021.jpg",
  "page_number": 21,
  "figure_number": "1",
  "species": {
    "scientific_name": "Acropora cytherea",
    "genus": "Acropora",
    "specific_epithet": "cytherea",
    "has_verified_binomial": true
  },
  "caption": {
    "text": "Acropora cytherea (Dana, 1846). Colony showing table form...",
    "is_refined": true
  },
  "bounding_box_300dpi": {
    "x": 229,
    "y": 362,
    "width": 2079,
    "height": 797
  },
  "normalized_box_1000": [108.0, 93.0, 345.0, 938.0],
  "image_dimensions": {
    "width": 2079,
    "height": 797,
    "dpi": 300
  },
  "curation": {
    "review_status": "ok",
    "extraction_method": "gemini_detect"
  }
}
```

---

## Quickstart

### 1. Load with Hugging Face `datasets`
```python
from datasets import load_dataset

# Load directly as an image dataset
dataset = load_dataset("imagefolder", data_dir="dataset")
print(dataset["train"][0])
```

### 2. Load with Pandas
```python
import pandas as pd

df = pd.read_csv("dataset/metadata.csv")
print(f"Loaded {len(df)} figures across {df['scientific_name'].nunique()} unique species.")
```

### 3. Fine-Tuning Vision-Language Models (LLaVA / Qwen2-VL)
The file `vqa_conversations.jsonl` contains standard instruction-tuning format:
```json
{
  "id": "coral_fig_001",
  "image": "images/coral_fig_001_p021.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nIdentify the marine species in this figure and summarize its characteristics."},
    {"from": "gpt", "value": "This figure shows *Acropora cytherea*. Caption: ..."}
  ]
}
```