"""
Object Extraction & Label Matching Pipeline
=============================================
For each folio:
1. Crop salient illustration regions from the high-res image
2. Detect text regions and match them to nearby illustration objects
3. Map text positions to transcript labels (from our label network)
4. Save annotated crops with their associated label text
5. Build a visual catalog linking objects → labels → lexicon entries
"""

import json
import re
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

# Paths
FOLIO_DIR = Path('folios')
CROPS_DIR = Path('crops')
CROPS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR = Path('annotated')
ANNOTATED_DIR.mkdir(exist_ok=True)

# Load all data
with open('data/visual/segmentation_data.json') as f:
    seg_data = json.load(f)
with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
with open('data/lexicon/label_network.json') as f:
    label_net = json.load(f)
with open('data/lexicon/lexicon.json') as f:
    lexicon = json.load(f)

metadata = nlp['metadata']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}
TYPE_NAMES = {
    'H': 'Herbal', 'A': 'Astro', 'Z': 'Zodiac', 'S': 'Stars',
    'B': 'Bio', 'C': 'Cosmo', 'P': 'Pharma', 'T': 'Text'
}

EVA_TO_PUA = {
    'a': '\U000FF410', 'b': '\U000FF408', 'c': '\U000FF40C',
    'd': '\U000FF409', 'e': '\U000FF406', 'f': '\U000FF420',
    'g': '\U000FF40B', 'h': '\U000FF40F', 'i': '\U000FF400',
    'j': '\U000FF402', 'k': '\U000FF422', 'l': '\U000FF41A',
    'm': '\U000FF404', 'n': '\U000FF401', 'o': '\U000FF414',
    'p': '\U000FF421', 'q': '\U000FF41D', 'r': '\U000FF403',
    's': '\U000FF40A', 't': '\U000FF423', 'u': '\U000FF411',
    'v': '\U000FF41B', 'x': '\U000FF41C', 'y': '\U000FF417',
    'z': '\U000FF41E',
}
def glyph(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)

# Build lexicon lookup
lexicon_lookup = {}
for entry in lexicon.get('entries', []):
    lexicon_lookup[entry['eva']] = entry

# Build label lookup: folio → list of labels with text
folio_labels = defaultdict(list)
for folio, info in label_net.get('folios', {}).items():
    for label in info.get('labels', []):
        folio_labels[folio].append(label)

# Also build paragraph text per folio with word positions
folio_paragraphs = defaultdict(list)
for s in nlp['sentences']:
    folio_paragraphs[s['folio']].append(s)

# Cross-reference data from label network
word_xrefs = label_net.get('word_index', {})

# =====================================================================
# Vision coordinates are normalized (0-1), origin at BOTTOM-LEFT
# PIL coordinates have origin at TOP-LEFT
# We need to convert: pil_y = img_height - (vision_y + vision_h) * img_height
# =====================================================================

def vision_to_pil(bbox, img_w, img_h):
    """Convert Vision normalized bbox to PIL pixel coords."""
    x = bbox['x'] * img_w
    y = (1.0 - bbox['y'] - bbox['h']) * img_h  # flip Y
    w = bbox['w'] * img_w
    h = bbox['h'] * img_h
    return (int(x), int(y), int(x + w), int(y + h))

def bbox_center(bbox):
    """Get center of a Vision normalized bbox."""
    return (bbox['x'] + bbox['w'] / 2, bbox['y'] + bbox['h'] / 2)

def bbox_distance(b1, b2):
    """Euclidean distance between bbox centers (normalized coords)."""
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def bbox_overlap(b1, b2):
    """Check if two bboxes overlap."""
    x1 = max(b1['x'], b2['x'])
    y1 = max(b1['y'], b2['y'])
    x2 = min(b1['x'] + b1['w'], b2['x'] + b2['w'])
    y2 = min(b1['y'] + b1['h'], b2['y'] + b2['h'])
    return x1 < x2 and y1 < y2

# =====================================================================
# Process each folio
# =====================================================================

catalog = {}
total_crops = 0
total_matched = 0

print("="*80)
print("OBJECT EXTRACTION & LABEL MATCHING")
print("="*80)

for folio_id, seg in sorted(seg_data.items()):
    img_path = FOLIO_DIR / f'{folio_id}.png'
    if not img_path.exists():
        continue

    sec = seg.get('section', '?')
    sec_name = TYPE_NAMES.get(sec, sec)
    orig_w, orig_h = seg['original_size']

    # Get all detected regions
    attention = seg.get('attention_saliency', [])
    objects = seg.get('object_saliency', [])
    text_regions = seg.get('text_regions', [])
    rectangles = seg.get('rectangles', [])

    # Merge all object-like regions
    all_objects = []
    for obj in attention:
        obj['source'] = 'attention'
        all_objects.append(obj)
    for obj in objects:
        obj['source'] = 'objectness'
        # Avoid duplicates with attention
        is_dup = any(bbox_distance(obj, a) < 0.05 for a in attention)
        if not is_dup:
            all_objects.append(obj)
    for rect in rectangles:
        rect['source'] = 'rectangle'
        # Only include rectangles that don't overlap heavily with attention
        is_dup = any(bbox_overlap(rect, a) for a in attention)
        if not is_dup and rect.get('conf', 0) > 0.5:
            all_objects.append(rect)

    if not all_objects:
        continue

    # Load full-res image
    img = Image.open(img_path)

    # Create annotated copy
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    # Get labels for this folio
    labels = folio_labels.get(folio_id, [])

    # Get paragraph words for context
    para_words = []
    for s in folio_paragraphs.get(folio_id, []):
        para_words.extend(s['words'])

    folio_catalog = {
        'folio': folio_id,
        'section': sec_name,
        'original_size': [orig_w, orig_h],
        'objects': [],
        'labels': [],
        'paragraph_word_count': len(para_words),
    }

    # Process each detected object
    for obj_idx, obj in enumerate(all_objects):
        # Convert to pixel coords
        pil_box = vision_to_pil(obj, orig_w, orig_h)

        # Clamp to image bounds
        x1 = max(0, pil_box[0])
        y1 = max(0, pil_box[1])
        x2 = min(orig_w, pil_box[2])
        y2 = min(orig_h, pil_box[3])

        if x2 - x1 < 50 or y2 - y1 < 50:
            continue

        # Crop the object
        crop = img.crop((x1, y1, x2, y2))
        crop_filename = f'{folio_id}_obj{obj_idx}.png'
        crop.save(CROPS_DIR / crop_filename)
        total_crops += 1

        # Find nearest text regions to this object
        nearby_text = []
        for tr in text_regions:
            dist = bbox_distance(obj, tr)
            if dist < 0.3:  # within 30% of image diagonal
                nearby_text.append({
                    'distance': round(dist, 4),
                    'position': bbox_center(tr),
                    'bbox': tr,
                })
        nearby_text.sort(key=lambda x: x['distance'])

        # Match labels based on position
        # Labels are typically near the objects they describe
        matched_labels = []
        for label in labels:
            # We don't have exact label pixel positions from transcript,
            # but we can match by label content
            matched_labels.append({
                'text': label.get('text', ''),
                'words': label.get('words', []),
                'unit': label.get('unit', ''),
                'line': label.get('line', 0),
            })

        # Check if any label words are in our lexicon
        lexicon_matches = []
        for label in matched_labels:
            for word in label.get('words', []):
                if word in lexicon_lookup:
                    entry = lexicon_lookup[word]
                    lexicon_matches.append({
                        'word': word,
                        'meaning': entry.get('meaning', ''),
                        'confidence': entry.get('confidence', ''),
                    })
                # Check cross-references
                if word in word_xrefs:
                    xref = word_xrefs[word]
                    lexicon_matches.append({
                        'word': word,
                        'cross_ref': True,
                        'other_folios': xref.get('folios', []),
                        'sections': xref.get('sections', []),
                    })

        # Draw annotation on the full image
        color = {
            'attention': '#FF6B35',
            'objectness': '#00D4AA',
            'rectangle': '#4A90D9',
        }.get(obj['source'], '#FFFFFF')

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        # Add label text
        label_text = f"obj{obj_idx} [{obj['source'][:3]}]"
        if nearby_text:
            label_text += f" txt:{len(nearby_text)}"
        try:
            draw.text((x1 + 5, y1 + 5), label_text, fill=color)
        except:
            pass

        obj_entry = {
            'index': obj_idx,
            'source': obj['source'],
            'bbox_normalized': {k: obj[k] for k in ['x', 'y', 'w', 'h']},
            'bbox_pixels': [x1, y1, x2, y2],
            'confidence': obj.get('conf', 0),
            'crop_file': crop_filename,
            'nearby_text_regions': len(nearby_text),
            'lexicon_matches': lexicon_matches[:5],
        }
        folio_catalog['objects'].append(obj_entry)

    # Add all labels to catalog
    for label in labels:
        label_entry = {
            'text': label.get('text', ''),
            'words': label.get('words', []),
            'unit': label.get('unit', ''),
            'line': label.get('line', 0),
            'in_lexicon': any(w in lexicon_lookup for w in label.get('words', [])),
            'has_cross_ref': any(w in word_xrefs for w in label.get('words', [])),
        }

        # Add cross-ref details
        xref_details = []
        for w in label.get('words', []):
            if w in word_xrefs:
                other = [f for f in word_xrefs[w]['folios'] if f != folio_id]
                if other:
                    xref_details.append({
                        'word': w,
                        'other_folios': other,
                    })
        label_entry['cross_refs'] = xref_details

        folio_catalog['labels'].append(label_entry)
        if label_entry['has_cross_ref']:
            total_matched += 1

    # Draw text region boxes in blue
    for tr in text_regions:
        pil_box = vision_to_pil(tr, orig_w, orig_h)
        draw.rectangle(pil_box, outline='#4A90D9', width=2)

    # Save annotated image
    annotated.save(ANNOTATED_DIR / f'{folio_id}_annotated.png')

    catalog[folio_id] = folio_catalog

    if len(catalog) % 30 == 0:
        print(f"  Processed {len(catalog)} folios, {total_crops} crops...")

print(f"\nDone: {len(catalog)} folios")
print(f"  Total object crops: {total_crops}")
print(f"  Labels with cross-references: {total_matched}")

# =====================================================================
# Summary by section
# =====================================================================

print(f"\n{'='*80}")
print("EXTRACTION SUMMARY BY SECTION")
print(f"{'='*80}")

for sec_code in ['H', 'A', 'Z', 'S', 'B', 'C', 'P', 'T']:
    sec_name = TYPE_NAMES.get(sec_code, sec_code)
    sec_folios = {f: d for f, d in catalog.items() if folio_type.get(f) == sec_code}
    if not sec_folios:
        continue

    n_objects = sum(len(d['objects']) for d in sec_folios.values())
    n_labels = sum(len(d['labels']) for d in sec_folios.values())
    n_xref = sum(1 for d in sec_folios.values()
                 for l in d['labels'] if l.get('has_cross_ref'))
    n_lexicon = sum(1 for d in sec_folios.values()
                    for l in d['labels'] if l.get('in_lexicon'))

    print(f"  {sec_name:10s}: {len(sec_folios):3d} folios  "
          f"objects={n_objects:4d}  labels={n_labels:4d}  "
          f"cross-refs={n_xref:3d}  in-lexicon={n_lexicon:3d}")

# =====================================================================
# Most interesting folios (most cross-referenced labels)
# =====================================================================

print(f"\n{'='*80}")
print("MOST CROSS-REFERENCED FOLIOS")
print(f"{'='*80}")

folio_xref_count = []
for folio_id, data in catalog.items():
    n_xref = sum(len(l.get('cross_refs', [])) for l in data['labels'])
    if n_xref > 0:
        folio_xref_count.append((folio_id, data['section'], n_xref, data['labels']))

folio_xref_count.sort(key=lambda x: -x[2])

for folio_id, sec, n_xref, labels in folio_xref_count[:20]:
    print(f"\n  {folio_id} [{sec}] — {n_xref} cross-references:")
    for label in labels:
        if label.get('cross_refs'):
            for xr in label['cross_refs']:
                other = ', '.join(xr['other_folios'][:5])
                word = xr['word']
                print(f"    {label['text']:20s}  '{word}' → {other}")

# Save catalog
with open('data/visual/object_catalog.json', 'w') as f:
    json.dump(catalog, f, indent=2)
print(f"\nObject catalog saved to object_catalog.json")
print(f"Annotated images in: {ANNOTATED_DIR}/")
print(f"Object crops in: {CROPS_DIR}/")
