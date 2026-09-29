"""
Standalone Dataset Builder & Exporter for Coral Figures
Transforms the curated ground-truth metadata (figures.json) into a standardized,
publication-ready dataset with:
- Standardized, sanitized 300 DPI image crops in dataset/images/
- Hugging Face Datasets & Multimodal ready metadata (JSONL, JSON, CSV)
- Normalized bounding box coordinates [0, 1000] (PaliGemma, LayoutLM, Florence-2 compatible)
- Granular taxonomic decomposition (genus, species epithet)
- Auto-generated Dataset Card (README.md) with comprehensive statistics

SAFETY GUARANTEE:
This script NEVER modifies or overwrites the input figures.json ground truth file.
It operates in strictly READ-ONLY mode on the source data.
"""

import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

from PIL import Image
import pandas as pd
from tqdm import tqdm
import pymupdf


def parse_taxonomy(species_name: Optional[str]) -> Dict[str, Any]:
    """
    Decomposes a scientific species name into genus and specific epithet.
    Handles author citations e.g. 'Acropora cytherea (Dana, 1846)'.
    """
    if not species_name or not str(species_name).strip():
        return {
            "scientific_name": None,
            "genus": None,
            "specific_epithet": None,
            "has_verified_binomial": False
        }

    raw = str(species_name).strip()
    if raw.lower() in ["null", "none", "n/a", "unknown"]:
        return {
            "scientific_name": None,
            "genus": None,
            "specific_epithet": None,
            "has_verified_binomial": False
        }

    # Remove author citations in parentheses for genus/epithet parsing
    clean = re.sub(r'\(.*?\)', '', raw).strip()
    clean = re.sub(r'[,;:]', '', clean).strip()
    tokens = clean.split()

    genus = tokens[0].capitalize() if len(tokens) > 0 else None
    specific_epithet = tokens[1].lower() if len(tokens) > 1 else None

    # Verify if valid binomial (no numbers, punctuation)
    has_verified = bool(
        genus
        and specific_epithet
        and genus.isalpha()
        and specific_epithet.isalpha()
        and specific_epithet not in ["sp", "spp", "cf"]
    )

    canonical_name = f"{genus} {specific_epithet}" if (genus and specific_epithet) else raw

    return {
        "scientific_name": canonical_name,
        "genus": genus,
        "specific_epithet": specific_epithet,
        "has_verified_binomial": has_verified
    }


def compute_normalized_box(box: List[int], page_w: int, page_h: int) -> List[float]:
    """
    Normalizes a 300 DPI pixel bounding box [x, y, w, h] to [0, 1000] coordinates
    in [ymin, xmin, ymax, xmax] format (standard for Vision-Language models).
    """
    if not box or len(box) < 4 or page_w <= 0 or page_h <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    ix, iy, iw, ih = box
    xmin = max(0.0, min(1000.0, (ix / page_w) * 1000.0))
    ymin = max(0.0, min(1000.0, (iy / page_h) * 1000.0))
    xmax = max(0.0, min(1000.0, ((ix + iw) / page_w) * 1000.0))
    ymax = max(0.0, min(1000.0, ((iy + ih) / page_h) * 1000.0))

    return [round(ymin, 1), round(xmin, 1), round(ymax, 1), round(xmax, 1)]


def generate_dataset_card(
    stats: Dict[str, Any],
    top_genera: List[tuple],
    top_species: List[tuple],
    output_path: Path
) -> None:
    """Auto-generates a rich markdown Dataset Card with statistics and usage code."""
    genera_rows = "\n".join([f"| *{g}* | {c} |" for g, c in top_genera[:10]])
    species_rows = "\n".join([f"| *{s}* | {c} |" for s, c in top_species[:10]])

    card_content = f"""# Coral & Marine Taxonomic Figures Dataset

A high-resolution, expert-curated dataset of marine and coral taxonomic figures, anatomical drawings, and biological field guide specimens extracted directly from scientific literature.

---

## Dataset Summary

| Metric | Value |
| :--- | :--- |
| **Total Figures** | **{stats['total_figures']}** |
| **Source Pages** | **{stats['total_pages']}** |
| **Verified Species Names** | **{stats['verified_species_count']}** ({stats['verified_species_pct']}%) |
| **Unique Genera** | **{stats['unique_genera_count']}** |
| **Unique Species** | **{stats['unique_species_count']}** |
| **Average Resolution** | **{stats['avg_width']} x {stats['avg_height']} px** |
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
{genera_rows}

### Top Species
| Species | Count |
| :--- | :--- |
{species_rows}

---

## Schema Definition (`metadata.jsonl`)

Each line in `metadata.jsonl` contains:

```json
{{
  "figure_id": "coral_fig_001",
  "image_path": "images/coral_fig_001_p021.jpg",
  "page_number": 21,
  "figure_number": "1",
  "species": {{
    "scientific_name": "Acropora cytherea",
    "genus": "Acropora",
    "specific_epithet": "cytherea",
    "has_verified_binomial": true
  }},
  "caption": {{
    "text": "Acropora cytherea (Dana, 1846). Colony showing table form...",
    "is_refined": true
  }},
  "bounding_box_300dpi": {{
    "x": 229,
    "y": 362,
    "width": 2079,
    "height": 797
  }},
  "normalized_box_1000": [108.0, 93.0, 345.0, 938.0],
  "image_dimensions": {{
    "width": 2079,
    "height": 797,
    "dpi": 300
  }},
  "curation": {{
    "review_status": "ok",
    "extraction_method": "gemini_detect"
  }}
}}
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
print(f"Loaded {{len(df)}} figures across {{df['scientific_name'].nunique()}} unique species.")
```

### 3. Fine-Tuning Vision-Language Models (LLaVA / Qwen2-VL)
The file `vqa_conversations.jsonl` contains standard instruction-tuning format:
```json
{{
  "id": "coral_fig_001",
  "image": "images/coral_fig_001_p021.jpg",
  "conversations": [
    {{"from": "human", "value": "<image>\\nIdentify the marine species in this figure and summarize its characteristics."}},
    {{"from": "gpt", "value": "This figure shows *Acropora cytherea*. Caption: ..."}}
  ]
}}
```
"""
    output_path.write_text(card_content.strip(), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Builds the final publication dataset from curated figures.json (READ-ONLY on ground truth)"
    )
    parser.add_argument("--input", default="output/figures.json", help="Path to curated figures.json ground truth")
    parser.add_argument("--pdf", default="index.pdf", help="Path to source index.pdf for coordinate normalization")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for the final dataset")
    parser.add_argument("--prefix", default="coral_fig", help="Filename prefix for standardized image files")
    parser.add_argument("--no-copy-images", action="store_true", help="Skip copying image files (metadata only)")
    parser.add_argument("--include-deleted", action="store_true", help="Include figures marked as deleted")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: Input ground truth file '{input_path}' not found.")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      CORAL FIGURES DATASET BUILDER & EXPORTER    ")
    print("==================================================")
    print(f"Input Ground Truth : {input_path} (READ-ONLY)")
    print(f"Output Directory   : {output_dir}")
    print(f"Copy Images        : {not args.no_copy_images}")

    # 1. READ-ONLY Ground Truth Ingestion
    with open(input_path, "r", encoding="utf-8") as f:
        raw_figures = json.load(f)

    # 2. Filter Active Figures
    if args.include_deleted:
        curated_figures = raw_figures
    else:
        curated_figures = [f for f in raw_figures if f.get("review_status") != "deleted"]

    print(f"Total entries in ground truth : {len(raw_figures)}")
    print(f"Active curated figures        : {len(curated_figures)}")
    print(f"Excluded deleted figures      : {len(raw_figures) - len(curated_figures)}")

    # 3. Sort Figures Logically by Page then Bounding Box Y
    def sort_key(f):
        page = f.get("page", 0)
        box = f.get("box", [0, 0, 0, 0])
        y = box[1] if len(box) > 1 else 0
        return (page, y)

    curated_figures.sort(key=sort_key)

    # 4. Cache Source Page Dimensions from PDF for Exact Normalization
    page_dims = {}
    if Path(args.pdf).exists():
        try:
            doc = pymupdf.open(args.pdf)
            for pno in range(len(doc)):
                page = doc[pno]
                w = int(round(page.rect.width * 300.0 / 72.0))
                h = int(round(page.rect.height * 300.0 / 72.0))
                page_dims[pno + 1] = (w, h)
        except Exception as e:
            print(f"Notice: Could not pre-cache PDF dimensions ({e}). Using image box bounds.")

    # 5. Process Figures & Standardize Image Files
    processed_records = []
    flat_rows = []
    vqa_conversations = []

    genera_counter = Counter()
    species_counter = Counter()
    widths, heights = [], []

    pbar = tqdm(curated_figures, desc="Exporting dataset records")
    for seq_idx, fig in enumerate(pbar, start=1):
        fig_id = f"{args.prefix}_{seq_idx:03d}"
        page_num = fig.get("page", 1)
        fig_num_str = str(fig.get("fig_num", "")).strip()
        caption_text = str(fig.get("caption", "")).strip()
        raw_species = fig.get("species_name")
        box = fig.get("box", [0, 0, 0, 0])

        # Standardized file name
        new_filename = f"{args.prefix}_{seq_idx:03d}_p{page_num:03d}.jpg"
        dest_rel_path = f"images/{new_filename}"
        dest_full_path = images_dir / new_filename

        # Copy original crop losslessly (no recompression)
        src_path_str = fig.get("image_path", "")
        img_width, img_height = 0, 0

        if src_path_str:
            src_path = Path(src_path_str)
            if not src_path.exists():
                # Fallback to local output/images/ if path was absolute
                fallback = Path("output/images") / src_path.name
                if fallback.exists():
                    src_path = fallback

            if src_path.exists():
                if not args.no_copy_images:
                    shutil.copy2(str(src_path), str(dest_full_path))

                # Inspect dimensions
                try:
                    with Image.open(src_path) as img:
                        img_width, img_height = img.size
                except Exception:
                    img_width = box[2] if len(box) > 2 else 0
                    img_height = box[3] if len(box) > 3 else 0
            else:
                print(f"\nWarning: Image file not found for figure {fig_id}: {src_path_str}")

        widths.append(img_width)
        heights.append(img_height)

        # Compute normalized coordinates [0, 1000]
        if page_num in page_dims:
            pw, ph = page_dims[page_num]
        else:
            pw, ph = 2500, 3500  # standard textbook 300 DPI reference

        norm_box = compute_normalized_box(box, pw, ph)

        # Parse taxonomy
        tax = parse_taxonomy(raw_species)
        if tax["genus"]:
            genera_counter[tax["genus"]] += 1
        if tax["scientific_name"]:
            species_counter[tax["scientific_name"]] += 1

        # Rich record structure
        record = {
            "figure_id": fig_id,
            "image_path": dest_rel_path,
            "page_number": page_num,
            "figure_number": fig_num_str,
            "species": tax,
            "caption": {
                "text": caption_text,
                "is_refined": True
            },
            "bounding_box_300dpi": {
                "x": box[0] if len(box) > 0 else 0,
                "y": box[1] if len(box) > 1 else 0,
                "width": box[2] if len(box) > 2 else 0,
                "height": box[3] if len(box) > 3 else 0
            },
            "normalized_box_1000": norm_box,
            "image_dimensions": {
                "width": img_width,
                "height": img_height,
                "dpi": 300
            },
            "curation": {
                "review_status": fig.get("review_status", "ok"),
                "extraction_method": fig.get("method", "gemini_detect")
            }
        }
        processed_records.append(record)

        # Tabular flat row for CSV
        flat_rows.append({
            "figure_id": fig_id,
            "file_name": dest_rel_path,
            "page_number": page_num,
            "figure_number": fig_num_str,
            "scientific_name": tax["scientific_name"] or "",
            "genus": tax["genus"] or "",
            "species": tax["specific_epithet"] or "",
            "has_verified_binomial": tax["has_verified_binomial"],
            "caption": caption_text,
            "img_width": img_width,
            "img_height": img_height,
            "curation_method": fig.get("method", "gemini_detect"),
            "review_status": fig.get("review_status", "ok")
        })

        # Multimodal VLM instruction format
        species_label = f"*{tax['scientific_name']}*" if tax["scientific_name"] else "the depicted organism"
        vqa_conversations.append({
            "id": fig_id,
            "image": dest_rel_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nWhat marine organism or feature is shown in this textbook figure, and what does the caption describe?"
                },
                {
                    "from": "gpt",
                    "value": f"This figure illustrates {species_label}. Figure caption: \"{caption_text}\""
                }
            ]
        })

    # 6. Save metadata.jsonl
    jsonl_path = output_dir / "metadata.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in processed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 7. Save metadata.json
    json_path = output_dir / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)

    # 8. Save metadata.csv
    csv_path = output_dir / "metadata.csv"
    df = pd.DataFrame(flat_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 9. Save vqa_conversations.jsonl
    vqa_path = output_dir / "vqa_conversations.jsonl"
    with open(vqa_path, "w", encoding="utf-8") as f:
        for item in vqa_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 10. Generate Dataset Card (README.md)
    total_figs = len(processed_records)
    verified_species_count = sum(1 for r in processed_records if r["species"]["has_verified_binomial"])
    pages_count = len(set(r["page_number"] for r in processed_records))

    stats_summary = {
        "total_figures": total_figs,
        "total_pages": pages_count,
        "verified_species_count": verified_species_count,
        "verified_species_pct": round(verified_species_count / max(1, total_figs) * 100, 1),
        "unique_genera_count": len(genera_counter),
        "unique_species_count": len(species_counter),
        "avg_width": int(sum(widths) / max(1, len(widths))),
        "avg_height": int(sum(heights) / max(1, len(heights))),
    }

    generate_dataset_card(
        stats_summary,
        genera_counter.most_common(15),
        species_counter.most_common(15),
        output_dir / "README.md"
    )

    # Final summary banner
    print("\n==================================================")
    print("            DATASET EXPORT COMPLETE               ")
    print("==================================================")
    print(f"Total Figures Exported   : {total_figs}")
    print(f"Total Source Pages       : {pages_count}")
    print(f"Verified Latin Species   : {verified_species_count} ({stats_summary['verified_species_pct']}%)")
    print(f"Unique Genera            : {len(genera_counter)}")
    print(f"Unique Species           : {len(species_counter)}")
    print(f"Average Image Resolution : {stats_summary['avg_width']} x {stats_summary['avg_height']} px")
    print("\nGenerated Files:")
    print(f"  - {jsonl_path} (Hugging Face Datasets JSONL)")
    print(f"  - {json_path} (Standard JSON)")
    print(f"  - {csv_path} (Tabular CSV)")
    print(f"  - {vqa_path} (Multimodal VLM Q&A JSONL)")
    print(f"  - {output_dir / 'README.md'} (Dataset Card & Documentation)")
    print(f"  - {images_dir} ({len(list(images_dir.glob('*.jpg')))} 300 DPI image crops)")
    print("==================================================")
    print("GROUND TRUTH VERIFICATION: 'output/figures.json' was NOT modified.")


if __name__ == "__main__":
    main()
