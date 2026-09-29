"""
FastAPI Backend for Coral & Marine Figures QA & Annotation Web App
Provides endpoints to inspect figures, render source pages, recrop at 300 DPI,
edit captions, delete improper crops, and add missed figures.
"""

import os
import json
import shutil
from pathlib import Path
from io import BytesIO
from typing import Optional, List

import cv2
import numpy as np
import pymupdf
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Coral Figures QA & Curator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_PATH = Path("index.pdf")
OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
FIGURES_JSON = OUTPUT_DIR / "figures.json"
BACKUP_JSON = OUTPUT_DIR / "figures.backup.json"
FLAGGED_PAGES_JSON = OUTPUT_DIR / "flagged_pages.json"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Create backup on startup if not already created
if FIGURES_JSON.exists() and not BACKUP_JSON.exists():
    shutil.copy(FIGURES_JSON, BACKUP_JSON)

# In-memory cached PyMuPDF doc
_doc = None

def get_pdf_doc():
    global _doc
    if _doc is None:
        if not PDF_PATH.exists():
            raise HTTPException(status_code=404, detail="index.pdf not found")
        _doc = pymupdf.open(str(PDF_PATH))
    return _doc


def load_figures_data() -> list:
    if not FIGURES_JSON.exists():
        return []
    with open(FIGURES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figures_data(data: list):
    with open(FIGURES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_flagged_pages() -> list:
    if not FLAGGED_PAGES_JSON.exists():
        return []
    with open(FLAGGED_PAGES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_flagged_pages(data: list):
    with open(FLAGGED_PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Cache rendered display pages (150 DPI) in memory
_page_cache = {}

class UpdateFigureRequest(BaseModel):
    caption: Optional[str] = None
    fig_num: Optional[str] = None
    species_name: Optional[str] = None
    status: Optional[str] = None # 'pending', 'ok', 'recropped', 'deleted'
    notes: Optional[str] = None

class RecropRequest(BaseModel):
    # Normalized coordinates [0, 1000] on the page image
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    caption: Optional[str] = None
    fig_num: Optional[str] = None

class AddCropRequest(BaseModel):
    page: int
    fig_num: str
    caption: str
    ymin: float
    xmin: float
    ymax: float
    xmax: float


@app.get("/api/figures")
def get_figures():
    """Returns all figures with review metadata."""
    figures = load_figures_data()
    for idx, f in enumerate(figures):
        f["id"] = idx
        if "review_status" not in f:
            f["review_status"] = "pending"
    return figures


@app.get("/api/flagged_pages")
def get_flagged_pages():
    """Returns list of pages flagged for missed extractions."""
    return load_flagged_pages()


@app.post("/api/flag_page/{page_num}")
def flag_page(page_num: int, body: dict = Body(default={})):
    """Flags or unflags a page for missed extractions."""
    flagged = set(load_flagged_pages())
    action = body.get("action", "toggle")
    if action == "add" or (action == "toggle" and page_num not in flagged):
        flagged.add(page_num)
    elif action == "remove" or (action == "toggle" and page_num in flagged):
        flagged.discard(page_num)
    save_flagged_pages(sorted(list(flagged)))
    return {"page": page_num, "flagged": page_num in flagged, "total_flagged": len(flagged)}


@app.get("/api/page_pdf/{page_num}")
def get_page_pdf(page_num: int, active_id: Optional[int] = None):
    """Generates and serves a single-page PDF with clear, high-contrast bounding boxes."""
    doc = get_pdf_doc()
    pno = page_num - 1
    if pno < 0 or pno >= len(doc):
        raise HTTPException(status_code=404, detail="Page number out of range")

    single = pymupdf.open()
    single.insert_pdf(doc, from_page=pno, to_page=pno)
    page = single[0]

    figures = load_figures_data()
    page_figs = [
        (idx, f) for idx, f in enumerate(figures)
        if f.get("page") == page_num and f.get("review_status") != "deleted"
    ]

    if page_figs:
        pix = page.get_pixmap(dpi=300)
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height

        for idx, fig in page_figs:
            box = fig.get("box")
            if not box or len(box) < 4:
                continue
            ix, iy, iw, ih = box
            r = pymupdf.Rect(ix * scale_x, iy * scale_y, (ix + iw) * scale_x, (iy + ih) * scale_y)
            is_active = (active_id is not None and idx == active_id)
            fig_num = fig.get('fig_num', '')
            species = fig.get('species_name', '')
            if species:
                fig_title = f"Fig. {fig_num}: {species}"
            else:
                fig_title = f"Fig. {fig_num}"

            if is_active:
                # Active Figure: Bold crimson red, thick stroke, red tint
                page.draw_rect(r, color=(0.93, 0.15, 0.15), fill=(0.93, 0.15, 0.15), fill_opacity=0.20, width=3.5)
                label = f"{fig_title} [Active]"
                badge_w = min(220.0, max(60.0, len(label) * 5.8 + 12.0))
                badge_h = 16.0
                badge_y = max(0.0, r.y0 - badge_h)
                badge_rect = pymupdf.Rect(r.x0, badge_y, r.x0 + badge_w, badge_y + badge_h)
                page.draw_rect(badge_rect, color=(0.93, 0.15, 0.15), fill=(0.93, 0.15, 0.15), width=1.0)
                page.insert_text((r.x0 + 4, badge_y + 11.5), label, fontsize=9.5, fontname="helv", color=(1, 1, 1))
            else:
                # Other figures: Vibrant Royal Blue, distinct outline, subtle tint
                page.draw_rect(r, color=(0.14, 0.45, 0.95), fill=(0.14, 0.45, 0.95), fill_opacity=0.14, width=2.5)
                badge_w = min(200.0, max(50.0, len(fig_title) * 5.5 + 10.0))
                badge_h = 14.0
                badge_y = max(0.0, r.y0 - badge_h)
                badge_rect = pymupdf.Rect(r.x0, badge_y, r.x0 + badge_w, badge_y + badge_h)
                page.draw_rect(badge_rect, color=(0.14, 0.45, 0.95), fill=(0.14, 0.45, 0.95), width=1.0)
                page.insert_text((r.x0 + 3, badge_y + 10.5), fig_title, fontsize=8.5, fontname="helv", color=(1, 1, 1))

    pdf_bytes = single.tobytes()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=page_{page_num}.pdf"}
    )


@app.get("/api/page/{page_num}")
def get_page_image(page_num: int, dpi: int = 150):
    """Renders and streams the source PDF page as a high-quality JPEG."""
    cache_key = (page_num, dpi)
    if cache_key in _page_cache:
        return Response(content=_page_cache[cache_key], media_type="image/jpeg")

    doc = get_pdf_doc()
    pno = page_num - 1
    if pno < 0 or pno >= len(doc):
        raise HTTPException(status_code=404, detail="Page number out of range")

    page = doc[pno]
    pix = page.get_pixmap(dpi=dpi)
    jpeg_bytes = pix.tobytes("jpeg")
    _page_cache[cache_key] = jpeg_bytes

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/image/{filename}")
def get_crop_image(filename: str):
    """Serves an extracted cropped image from output/images/."""
    file_path = IMAGES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {filename} not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.post("/api/figure/{index}/update")
def update_figure(index: int, req: UpdateFigureRequest):
    """Updates caption, figure number, and review status of a figure."""
    figures = load_figures_data()
    if index < 0 or index >= len(figures):
        raise HTTPException(status_code=404, detail="Figure index out of range")

    fig = figures[index]
    if req.caption is not None:
        fig["caption"] = req.caption.strip()
    if req.fig_num is not None:
        fig["fig_num"] = req.fig_num.strip()
    if req.status is not None:
        fig["review_status"] = req.status
    if req.species_name is not None:
        fig["species_name"] = req.species_name.strip() if req.species_name.strip() else None
    if req.notes is not None:
        fig["notes"] = req.notes

    save_figures_data(figures)
    return {"status": "success", "figure": fig}


@app.post("/api/figure/{index}/recrop")
def recrop_figure(index: int, req: RecropRequest):
    """
    Re-crops a figure from the source PDF page at 300 DPI using new coordinates [0, 1000].
    Overwrites the crop file on disk and updates figures.json.
    """
    figures = load_figures_data()
    if index < 0 or index >= len(figures):
        raise HTTPException(status_code=404, detail="Figure index out of range")

    fig = figures[index]
    page_num = fig["page"]
    doc = get_pdf_doc()
    page = doc[page_num - 1]

    # Render at 300 DPI for high quality crop
    dpi = 300
    pix = page.get_pixmap(dpi=dpi)
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]

    # Convert normalized [0, 1000] coordinates to pixel coordinates
    ymin = max(0, min(1000, req.ymin))
    xmin = max(0, min(1000, req.xmin))
    ymax = max(0, min(1000, req.ymax))
    xmax = max(0, min(1000, req.xmax))

    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    iy = int(ymin * h / 1000.0)
    ix = int(xmin * w / 1000.0)
    ih = int((ymax - ymin) * h / 1000.0)
    iw = int((xmax - xmin) * w / 1000.0)

    if iw < 10 or ih < 10:
        raise HTTPException(status_code=400, detail="Crop region too small")

    # Crop
    crop = img_bgr[iy:iy+ih, ix:ix+iw]
    img_path = Path(fig["image_path"])

    # Overwrite crop file
    cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Update figure metadata
    fig["box"] = [ix, iy, iw, ih]
    fig["review_status"] = "recropped"
    fig["method"] = "manual_recrop"
    if req.caption is not None:
        fig["caption"] = req.caption.strip()
    if req.fig_num is not None:
        fig["fig_num"] = req.fig_num.strip()

    save_figures_data(figures)
    return {"status": "success", "figure": fig}


@app.post("/api/page/{page_num}/add_crop")
def add_manual_crop(page_num: int, req: AddCropRequest):
    """
    Manually creates a new figure crop on a page from [0, 1000] coordinates.
    Saves new high-res image and appends to figures.json.
    """
    doc = get_pdf_doc()
    pno = page_num - 1
    if pno < 0 or pno >= len(doc):
        raise HTTPException(status_code=404, detail="Page number out of range")

    page = doc[pno]
    dpi = 300
    pix = page.get_pixmap(dpi=dpi)
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]

    ymin = max(0, min(1000, min(req.ymin, req.ymax)))
    xmin = max(0, min(1000, min(req.xmin, req.xmax)))
    ymax = max(0, min(1000, max(req.ymin, req.ymax)))
    xmax = max(0, min(1000, max(req.xmin, req.xmax)))

    iy = int(ymin * h / 1000.0)
    ix = int(xmin * w / 1000.0)
    ih = int((ymax - ymin) * h / 1000.0)
    iw = int((xmax - xmin) * w / 1000.0)

    if iw < 10 or ih < 10:
        raise HTTPException(status_code=400, detail="Crop region too small")

    crop = img_bgr[iy:iy+ih, ix:ix+iw]

    fig_num_clean = req.fig_num.strip().replace(" ", "_")
    filename = f"fig_manual_{fig_num_clean}_p{page_num}.jpg"
    img_path = IMAGES_DIR / filename
    cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    figures = load_figures_data()
    new_fig = {
        "page": page_num,
        "fig_num": req.fig_num.strip(),
        "caption": req.caption.strip(),
        "box": [ix, iy, iw, ih],
        "image_path": str(img_path.resolve()),
        "method": "manual_added",
        "review_status": "ok",
        "id": len(figures)
    }
    figures.append(new_fig)
    save_figures_data(figures)

    return {"status": "success", "figure": new_fig}


@app.get("/api/stats")
def get_stats():
    """Returns curation progress stats."""
    figures = load_figures_data()
    flagged = load_flagged_pages()
    counts = {"total": len(figures), "pending": 0, "ok": 0, "recropped": 0, "deleted": 0, "flagged_pages": len(flagged)}
    for f in figures:
        st = f.get("review_status", "pending")
        counts[st] = counts.get(st, 0) + 1
    return counts


# Serve static web UI
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
