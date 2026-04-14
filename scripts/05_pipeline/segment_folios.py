"""
Voynich Folio Segmentation Pipeline
=====================================
1. Extract each folio page from the Yale PDF as a high-res image
2. Use Apple Vision framework to detect:
   - Contiguous illustration regions (plants, nymphs, diagrams)
   - Text regions
   - Salient objects
3. Save individual folio images + segmentation metadata
4. Build an object catalog for cross-referencing with the lexicon

This is Step 1: PDF extraction + basic segmentation setup.
"""

import json
import re
import subprocess
import os
from pathlib import Path
from collections import defaultdict

# Paths
PDF_PATH = 'source/voynich_scan_yale.pdf'
OUTPUT_DIR = Path('folios')
OUTPUT_DIR.mkdir(exist_ok=True)
SEGMENTS_DIR = Path('segments')
SEGMENTS_DIR.mkdir(exist_ok=True)

# Load our existing data
with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
metadata = nlp['metadata']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}

TYPE_NAMES = {
    'H': 'Herbal', 'A': 'Astro', 'Z': 'Zodiac', 'S': 'Stars',
    'B': 'Bio', 'C': 'Cosmo', 'P': 'Pharma', 'T': 'Text'
}

# =====================================================================
# Step 1: Build PDF page → folio mapping
# =====================================================================

print("Step 1: Building PDF page → folio mapping...")

result = subprocess.run(
    ['/opt/homebrew/bin/pdftotext', '-layout', PDF_PATH, '-'],
    capture_output=True, text=True
)
pages = result.stdout.split('\f')
print(f"  PDF has {len(pages)} pages")

folio_map = {}  # pdf_page (1-indexed) → folio_id
for i, page in enumerate(pages):
    lines = [l.strip() for l in page.strip().split('\n') if l.strip()]
    for line in lines[:5]:
        m = re.match(r'^(\d+[rv]\d?)$', line)
        if m:
            folio_map[i + 1] = f'f{m.group(1)}'
            break

# Filter to actual manuscript folios only (not covers, etc.)
ms_pages = {page: folio for page, folio in folio_map.items()
            if folio.startswith('f') and not folio.startswith('f0')}

print(f"  Mapped {len(ms_pages)} manuscript folios")

# =====================================================================
# Step 2: Extract folio images from PDF
# =====================================================================

print("\nStep 2: Extracting folio images (300 DPI)...")

# Extract all at once with pdftoppm for efficiency
# We'll extract at 300 DPI for good quality while being manageable
DPI = 300
extracted = 0

for pdf_page, folio_id in sorted(ms_pages.items()):
    out_path = OUTPUT_DIR / f'{folio_id}.png'
    if out_path.exists():
        extracted += 1
        continue

    # pdftoppm uses 0-indexed pages for -f/-l but the mapping is 1-indexed
    subprocess.run([
        '/opt/homebrew/bin/pdftoppm',
        '-png', '-r', str(DPI),
        '-f', str(pdf_page), '-l', str(pdf_page),
        '-singlefile',
        PDF_PATH,
        str(OUTPUT_DIR / folio_id)
    ], capture_output=True)

    if out_path.exists():
        extracted += 1
        if extracted % 20 == 0:
            print(f"  Extracted {extracted}/{len(ms_pages)} folios...")

print(f"  Done: {extracted} folio images in {OUTPUT_DIR}/")

# =====================================================================
# Step 3: Run Apple Vision saliency + contour detection
# =====================================================================

print("\nStep 3: Running Apple Vision segmentation...")

import Vision
import Quartz
from PIL import Image
import numpy as np

def load_cgimage(path):
    """Load an image as a CGImage for Vision framework."""
    url = Quartz.CFURLCreateWithFileSystemPath(
        None, str(path), Quartz.kCFURLPOSIXPathStyle, False
    )
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None:
        return None
    cgimage = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    return cgimage

def detect_saliency(cgimage):
    """Detect attention-based saliency regions."""
    request = Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    salient_regions = []
    for obs in results:
        objects = obs.salientObjects()
        if objects:
            for obj in objects:
                bbox = obj.boundingBox()
                salient_regions.append({
                    'x': float(bbox.origin.x),
                    'y': float(bbox.origin.y),
                    'width': float(bbox.size.width),
                    'height': float(bbox.size.height),
                    'confidence': float(obj.confidence()),
                })
    return salient_regions

def detect_contours(cgimage):
    """Detect contour paths in the image."""
    request = Vision.VNDetectContoursRequest.alloc().init()
    request.setContrastAdjustment_(1.5)
    request.setDetectsDarkOnLight_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    contours = []
    for obs in results:
        n_children = obs.childContourCount()
        # Get top-level contour info
        contours.append({
            'child_count': int(n_children),
            'point_count': int(obs.pointCount()),
        })
        # Get child contours (individual objects)
        for i in range(min(n_children, 50)):  # cap at 50
            try:
                child = obs.childContourAtIndex_error_(i, None)
                if child and child[0]:
                    child_contour = child[0]
                    bbox = child_contour.normalizedPath().boundingBox()
                    contours.append({
                        'type': 'child',
                        'index': i,
                        'x': float(bbox.origin.x),
                        'y': float(bbox.origin.y),
                        'width': float(bbox.size.width),
                        'height': float(bbox.size.height),
                        'point_count': int(child_contour.pointCount()),
                    })
            except Exception:
                pass
    return contours

def detect_text_regions(cgimage):
    """Detect text bounding boxes (not OCR — just location)."""
    request = Vision.VNDetectTextRectanglesRequest.alloc().init()
    request.setReportCharacterBoxes_(False)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    text_regions = []
    for obs in results:
        bbox = obs.boundingBox()
        text_regions.append({
            'x': float(bbox.origin.x),
            'y': float(bbox.origin.y),
            'width': float(bbox.size.width),
            'height': float(bbox.size.height),
            'confidence': float(obs.confidence()),
        })
    return text_regions

# Process each folio
segmentation_data = {}
processed = 0

for folio_id in sorted(ms_pages.values()):
    img_path = OUTPUT_DIR / f'{folio_id}.png'
    if not img_path.exists():
        continue

    cgimage = load_cgimage(img_path)
    if cgimage is None:
        print(f"  WARNING: Could not load {img_path}")
        continue

    # Get image dimensions
    width = Quartz.CGImageGetWidth(cgimage)
    height = Quartz.CGImageGetHeight(cgimage)

    # Run detections
    saliency = detect_saliency(cgimage)
    text_regions = detect_text_regions(cgimage)

    # Try contours (may fail on some images)
    try:
        contours = detect_contours(cgimage)
    except Exception as e:
        contours = [{'error': str(e)}]

    sec = folio_type.get(folio_id, '?')
    sec_name = TYPE_NAMES.get(sec, sec)

    segmentation_data[folio_id] = {
        'section': sec,
        'section_name': sec_name,
        'image_width': width,
        'image_height': height,
        'image_path': str(img_path),
        'n_salient_regions': len(saliency),
        'salient_regions': saliency,
        'n_text_regions': len(text_regions),
        'text_regions': text_regions,
        'n_contours': len([c for c in contours if c.get('type') == 'child']),
        'contours_summary': contours[:5],
    }

    processed += 1
    if processed % 20 == 0:
        print(f"  Processed {processed}/{len(ms_pages)} folios...")

print(f"  Done: segmented {processed} folios")

# =====================================================================
# Step 4: Summary statistics
# =====================================================================

print(f"\n{'=' * 80}")
print("SEGMENTATION SUMMARY")
print(f"{'=' * 80}")

for sec_code in ['H', 'A', 'Z', 'S', 'B', 'C', 'P', 'T']:
    sec_name = TYPE_NAMES.get(sec_code, sec_code)
    sec_folios = [f for f, d in segmentation_data.items() if d['section'] == sec_code]
    if not sec_folios:
        continue

    avg_salient = sum(segmentation_data[f]['n_salient_regions'] for f in sec_folios) / len(sec_folios)
    avg_text = sum(segmentation_data[f]['n_text_regions'] for f in sec_folios) / len(sec_folios)
    avg_contours = sum(segmentation_data[f]['n_contours'] for f in sec_folios) / len(sec_folios)

    print(f"  {sec_name:10s}: {len(sec_folios):3d} folios  "
          f"avg salient={avg_salient:.1f}  avg text={avg_text:.1f}  avg contours={avg_contours:.1f}")

# Save
with open('data/visual/segmentation_data.json', 'w') as f:
    json.dump(segmentation_data, f, indent=2)
print(f"\nSegmentation data saved to segmentation_data.json")

# Quick sample: show the most complex folios
print(f"\nMost salient-region-rich folios:")
for folio, data in sorted(segmentation_data.items(),
                          key=lambda x: -x[1]['n_salient_regions'])[:10]:
    print(f"  {folio:10s} [{data['section_name']:8s}]  "
          f"salient={data['n_salient_regions']:3d}  "
          f"text={data['n_text_regions']:3d}  "
          f"contours={data['n_contours']:3d}")

print(f"\nFolio images: {OUTPUT_DIR}/")
print(f"Ready for Apple object detection integration.")
