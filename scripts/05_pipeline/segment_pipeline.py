"""
Voynich Segmentation Pipeline (fixed)
======================================
Process all extracted folio PNGs through Apple Vision:
  - Downscale to ~850px wide for speed
  - Attention saliency → find illustration vs text regions
  - Text rectangle detection → locate text blocks
  - Object-based saliency → find individual objects
  - Save per-folio segmentation metadata
"""

import json
import time
import Vision
import Quartz
from PIL import Image
from pathlib import Path
from collections import defaultdict

FOLIO_DIR = Path('folios')
SEGMENT_DIR = Path('segments')
SEGMENT_DIR.mkdir(exist_ok=True)

# Load existing metadata
with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
metadata = nlp['metadata']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}
TYPE_NAMES = {
    'H': 'Herbal', 'A': 'Astro', 'Z': 'Zodiac', 'S': 'Stars',
    'B': 'Bio', 'C': 'Cosmo', 'P': 'Pharma', 'T': 'Text'
}

def load_and_resize(path, max_width=850):
    """Load a folio image, resize for Vision processing, return CGImage."""
    img = Image.open(path)
    orig_w, orig_h = img.size
    if orig_w > max_width:
        ratio = max_width / orig_w
        new_h = int(orig_h * ratio)
        img = img.resize((max_width, new_h), Image.LANCZOS)

    # Save temp for CGImage loading
    tmp = '/tmp/_voynich_seg.png'
    img.save(tmp)

    url = Quartz.CFURLCreateWithFileSystemPath(None, tmp, Quartz.kCFURLPOSIXPathStyle, False)
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    cgimage = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    return cgimage, orig_w, orig_h, img.width, img.height

def run_saliency(cgimage, saliency_type='attention'):
    """Run saliency detection. Returns list of salient region bboxes."""
    if saliency_type == 'attention':
        request = Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()
    else:
        request = Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init()

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    regions = []
    for obs in results:
        objects = obs.salientObjects()
        if objects:
            for obj in objects:
                bbox = obj.boundingBox()
                regions.append({
                    'x': round(float(bbox.origin.x), 4),
                    'y': round(float(bbox.origin.y), 4),
                    'w': round(float(bbox.size.width), 4),
                    'h': round(float(bbox.size.height), 4),
                    'conf': round(float(obj.confidence()), 4),
                })
    return regions

def run_text_detection(cgimage):
    """Detect text bounding boxes."""
    request = Vision.VNDetectTextRectanglesRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    regions = []
    for obs in results:
        bbox = obs.boundingBox()
        regions.append({
            'x': round(float(bbox.origin.x), 4),
            'y': round(float(bbox.origin.y), 4),
            'w': round(float(bbox.size.width), 4),
            'h': round(float(bbox.size.height), 4),
            'conf': round(float(obs.confidence()), 4),
        })
    return regions

def run_rectangle_detection(cgimage):
    """Detect rectangular regions (jars, frames, etc.)."""
    request = Vision.VNDetectRectanglesRequest.alloc().init()
    request.setMaximumObservations_(20)
    request.setMinimumSize_(0.05)
    request.setMinimumConfidence_(0.3)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    rects = []
    for obs in results:
        bbox = obs.boundingBox()
        rects.append({
            'x': round(float(bbox.origin.x), 4),
            'y': round(float(bbox.origin.y), 4),
            'w': round(float(bbox.size.width), 4),
            'h': round(float(bbox.size.height), 4),
            'conf': round(float(obs.confidence()), 4),
        })
    return rects

# =====================================================================
# Process all folios
# =====================================================================

folio_files = sorted(FOLIO_DIR.glob('f*.png'))
print(f"Processing {len(folio_files)} folio images...")
print(f"(downscaling to 850px width for Vision processing)\n")

all_data = {}
t_start = time.time()

for i, img_path in enumerate(folio_files):
    folio_id = img_path.stem  # e.g. 'f2r'

    cgimage, orig_w, orig_h, proc_w, proc_h = load_and_resize(img_path)
    if cgimage is None:
        continue

    sec = folio_type.get(folio_id, '?')

    # Run detections
    attention = run_saliency(cgimage, 'attention')
    objectness = run_saliency(cgimage, 'objectness')
    text_regions = run_text_detection(cgimage)
    rectangles = run_rectangle_detection(cgimage)

    all_data[folio_id] = {
        'section': sec,
        'section_name': TYPE_NAMES.get(sec, sec),
        'original_size': [orig_w, orig_h],
        'processed_size': [proc_w, proc_h],
        'attention_saliency': attention,
        'object_saliency': objectness,
        'text_regions': text_regions,
        'rectangles': rectangles,
        'n_attention': len(attention),
        'n_objects': len(objectness),
        'n_text': len(text_regions),
        'n_rectangles': len(rectangles),
    }

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        remaining = (len(folio_files) - i - 1) / rate
        print(f"  {i+1}/{len(folio_files)} folios  "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

elapsed = time.time() - t_start
print(f"\nDone: {len(all_data)} folios in {elapsed:.0f}s ({elapsed/len(all_data):.1f}s/folio)")

# =====================================================================
# Summary
# =====================================================================

print(f"\n{'='*80}")
print("SEGMENTATION SUMMARY")
print(f"{'='*80}")

for sec_code in ['H', 'A', 'Z', 'S', 'B', 'C', 'P', 'T']:
    sec_name = TYPE_NAMES.get(sec_code, sec_code)
    sec_folios = [f for f, d in all_data.items() if d['section'] == sec_code]
    if not sec_folios:
        continue

    avg_att = sum(all_data[f]['n_attention'] for f in sec_folios) / len(sec_folios)
    avg_obj = sum(all_data[f]['n_objects'] for f in sec_folios) / len(sec_folios)
    avg_txt = sum(all_data[f]['n_text'] for f in sec_folios) / len(sec_folios)
    avg_rect = sum(all_data[f]['n_rectangles'] for f in sec_folios) / len(sec_folios)

    print(f"  {sec_name:10s}: {len(sec_folios):3d} folios  "
          f"attn={avg_att:.1f}  obj={avg_obj:.1f}  text={avg_txt:.1f}  rect={avg_rect:.1f}")

# Most complex folios
print(f"\nMost object-rich folios (by objectness saliency):")
for folio, data in sorted(all_data.items(), key=lambda x: -x[1]['n_objects'])[:15]:
    print(f"  {folio:10s} [{data['section_name']:8s}]  "
          f"objects={data['n_objects']:3d}  text={data['n_text']:3d}  rect={data['n_rectangles']:2d}")

# Folios with most text regions
print(f"\nMost text-dense folios:")
for folio, data in sorted(all_data.items(), key=lambda x: -x[1]['n_text'])[:15]:
    print(f"  {folio:10s} [{data['section_name']:8s}]  "
          f"text={data['n_text']:3d}  objects={data['n_objects']:3d}")

# Save
with open('data/visual/segmentation_data.json', 'w') as f:
    json.dump(all_data, f, indent=2)
print(f"\nSegmentation data saved to segmentation_data.json")
print(f"Folio images in: folios/")
print(f"\nReady for object matching against label lexicon.")
