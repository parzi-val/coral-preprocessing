"""
Scientific Caption Refinement & OCR Fixer for Coral & Marine Figures
Focuses purely on caption text fidelity:
- Compares extracted captions against high-res source PDF page and extracted text blocks.
- Fixes OCR typos, broken words, garbled punctuation, and author citations.
- Validates and corrects scientific binomial nomenclature (genus/species).
- Trims extraneous prose that leaked in from surrounding book text.
Does NOT evaluate image crops or bounding boxes.
"""

import os
import re
import sys
import json
import base64
import argparse
from pathlib import Path

import pymupdf
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy_key")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def parse_llm_json(raw_content: str):
    """Clean markdown fences and parse JSON from LLM response."""
    c = raw_content.strip()
    if c.startswith("```"):
        c = re.sub(r'^```(?:json)?\s*', '', c)
        c = re.sub(r'\s*```$', '', c)
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', c)
    target = match.group(1) if match else c
    try:
        return json.loads(target, strict=False)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', target)
        return json.loads(fixed, strict=False)


def refine_page_captions(doc, page_num: int, figures_on_page: list, client: OpenAI) -> dict:
    """Refines all captions on a given page using Gemini via OpenAI completions format."""
    pno = page_num - 1
    if pno < 0 or pno >= len(doc):
        return {"page": page_num, "figures": []}

    page = doc[pno]

    # 1. High-res page bitmap for visual layout verification
    pix = page.get_pixmap(dpi=150)
    page_b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

    # 2. Text stream blocks directly from the PDF document
    text_blocks = [b[4].strip() for b in page.get_text("blocks") if b[4].strip()]
    raw_page_text = "\n---\n".join(text_blocks)

    # 3. Payload of figures
    figs_payload = []
    for f in figures_on_page:
        figs_payload.append({
            "id": f.get("id"),
            "fig_num": f.get("fig_num", ""),
            "current_caption": f.get("caption", "")
        })

    prompt = (
        "You are an expert scientific editor and taxonomist proofreading figure captions for a field guide on marine biology and corals.\n\n"
        "Your SOLE task is to verify, fix, and complete the CAPTIONS for the figures on this page.\n"
        "Do NOT evaluate image crops or bounding boxes.\n\n"
        "Examine the attached source PDF page image and the extracted raw text stream.\n"
        f"Figures on this page:\n{json.dumps(figs_payload, indent=2)}\n\n"
        f"=== RAW EXTRACTED PAGE TEXT ===\n{raw_page_text}\n\n"
        "INSTRUCTIONS FOR EACH FIGURE:\n"
        "1. Locate where this figure's caption appears in the text or under the figure.\n"
        "2. Check if the current caption:\n"
        "   - Contains OCR typos (e.g. 'tbeir' -> 'their', 'Fi,g.' -> 'Fig.').\n"
        "   - Has garbled punctuation or figure numbering (e.g. 'Fig. ,2'93,.' -> 'Fig. 2.93.').\n"
        "   - Truncated mid-sentence or missed second/third sentences or author citations.\n"
        "   - Has misspelled Latin scientific binomial names (e.g. genus/species names).\n"
        "   - Accidentally absorbed adjacent body prose that is not part of the caption.\n"
        "3. Provide the clean, complete, accurate caption.\n"
        "   - If already accurate and complete, leave it intact with has_changes=false.\n"
        "   - If figure number was misidentified, provide the corrected fig_num.\n"
        "4. Extract the raw scientific/species name (Latin binomial: Genus + species, or Genus sp.) if it exists in the caption or on the page for this specimen (e.g. 'Acanthopleura spiniger', 'Acropora digitifera', 'Sabella melanostigma', 'Sinularia sp.'). If the figure is a general landscape, non-taxonomic diagram, or has no Latin scientific name, set species_name to null.\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        f"  \"page\": {page_num},\n"
        "  \"figures\": [\n"
        "    {\n"
        "      \"id\": 0,\n"
        "      \"fig_num\": \"1\",\n"
        "      \"original_caption\": \"...\",\n"
        "      \"cleaned_caption\": \"...\",\n"
        "      \"species_name\": \"Acropora digitifera\",\n"
        "      \"has_changes\": true,\n"
        "      \"change_summary\": \"Fixed typo 'tbeir' to 'their'\",\n"
        "      \"confidence\": \"high\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Return ONLY the valid JSON object."
    )

    content_parts = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page_b64}"}}
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.1
            )
            return parse_llm_json(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(2)
                continue
            tqdm.write(f"Refinement error on Page {page_num}: {e}")
            return {
                "page": page_num,
                "error": str(e),
                "figures": [
                    {
                        "id": f.get("id"),
                        "fig_num": f.get("fig_num"),
                        "original_caption": f.get("caption"),
                        "cleaned_caption": f.get("caption"),
                        "has_changes": False,
                        "change_summary": f"Error: {e}",
                        "confidence": "none"
                    }
                    for f in figures_on_page
                ]
            }


def main():
    parser = argparse.ArgumentParser(description="Refine and fix coral figure captions using LLM")
    parser.add_argument("--pdf", default="index.pdf", help="Path to input PDF file")
    parser.add_argument("--metadata", default="output/figures.json", help="Path to figures.json metadata")
    parser.add_argument("--output-json", default="output/refined_captions.json", help="Path to output JSON")
    parser.add_argument("--output-csv", default="output/refined_captions.csv", help="Path to output CSV")
    parser.add_argument("--start-page", type=int, default=None, help="Start page filter")
    parser.add_argument("--end-page", type=int, default=None, help="End page filter")
    parser.add_argument("--pages", type=str, default=None, help="Comma-separated page numbers or 'manual' for all pages with manual figures")
    parser.add_argument("--only-manual", action="store_true", help="Only apply updates to manual added figures (method == 'manual_added')")
    parser.add_argument("--limit-pages", type=int, default=None, help="Limit number of pages")
    parser.add_argument("--apply", action="store_true", help="Directly update figures.json with cleaned captions")
    args = parser.parse_args()

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    doc = pymupdf.open(args.pdf)

    with open(args.metadata, "r", encoding="utf-8") as f:
        figures = json.load(f)

    # Group figures by page
    pages_map = {}
    for idx, fig in enumerate(figures):
        fig["id"] = idx
        if fig.get("review_status") == "deleted":
            continue
        p = fig.get("page")
        if p not in pages_map:
            pages_map[p] = []
        pages_map[p].append(fig)

    pages = sorted(list(pages_map.keys()))
    if args.pages:
        if args.pages.strip().lower() == "manual":
            manual_pages = sorted(list(set(f["page"] for f in figures if f.get("method") == "manual_added" and f.get("review_status") != "deleted")))
            pages = [p for p in pages if p in manual_pages]
        else:
            selected_pages = set(int(p.strip()) for p in args.pages.split(",") if p.strip())
            pages = [p for p in pages if p in selected_pages]
    if args.start_page:
        pages = [p for p in pages if p >= args.start_page]
    if args.end_page:
        pages = [p for p in pages if p <= args.end_page]
    if args.limit_pages:
        pages = pages[:args.limit_pages]

    print(f"Refining captions across {len(pages)} pages using {LLM_MODEL}...")

    all_results = []
    flat_rows = []

    pbar = tqdm(pages, desc="Refining captions")
    for page_num in pbar:
        res = refine_page_captions(doc, page_num, pages_map[page_num], client)
        all_results.append(res)

        for fig_res in res.get("figures", []):
            species = fig_res.get("species_name")
            if species and str(species).strip().lower() in ["null", "none", "n/a", ""]:
                species = None
            elif species:
                species = str(species).strip()

            flat_rows.append({
                "page": page_num,
                "figure_id": fig_res.get("id"),
                "fig_num": fig_res.get("fig_num"),
                "species_name": species,
                "has_changes": fig_res.get("has_changes", False),
                "change_summary": fig_res.get("change_summary", ""),
                "original_caption": fig_res.get("original_caption", ""),
                "cleaned_caption": fig_res.get("cleaned_caption", ""),
                "confidence": fig_res.get("confidence", "")
            })

            if args.apply:
                fid = fig_res.get("id")
                if fid is not None and 0 <= fid < len(figures):
                    if args.only_manual and figures[fid].get("method") != "manual_added":
                        continue
                    if fig_res.get("has_changes") and fig_res.get("cleaned_caption"):
                        figures[fid]["caption"] = fig_res["cleaned_caption"]
                    if fig_res.get("fig_num"):
                        figures[fid]["fig_num"] = str(fig_res["fig_num"]).strip()
                    if species:
                        figures[fid]["species_name"] = species

        # Incremental save every page
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if args.apply:
            with open(args.metadata, "w", encoding="utf-8") as f:
                json.dump(figures, f, indent=2, ensure_ascii=False)
            csv_path = Path(args.metadata).with_suffix(".csv")
            try:
                pd.DataFrame(figures).to_csv(csv_path, index=False, encoding="utf-8-sig")
            except Exception:
                pass

    # Save final CSV
    if flat_rows:
        df = pd.DataFrame(flat_rows)
        df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    changed_count = sum(1 for r in flat_rows if r.get("has_changes"))
    print(f"\nDone! Processed {len(flat_rows)} figures across {len(pages)} pages.")
    print(f"Refinements made: {changed_count} / {len(flat_rows)} ({changed_count/max(1, len(flat_rows))*100:.1f}%)")
    print(f"Saved reports to: {args.output_json} and {args.output_csv}")
    if args.apply:
        print(f"Updated figures saved directly to: {args.metadata}")


if __name__ == "__main__":
    main()
