"""
Label Cross-Reference Network Map
====================================
Build a complete map of every label in the manuscript, grouped by folio,
with cross-references to every other folio where the same label appears.
Output as a detailed text report suitable for visual verification against
high-res manuscript images.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse ALL labels from transcript (consensus across transcribers)
# ---------------------------------------------------------------------------

with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)

metadata = nlp['metadata']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}

TYPE_NAMES = {
    'H': 'Herbal', 'A': 'Astro', 'Z': 'Zodiac', 'S': 'Stars',
    'B': 'Bio', 'C': 'Cosmo', 'P': 'Pharma', 'T': 'Text'
}

LINE_RE = re.compile(
    r'^<(?P<folio>f\d+[rv]\d?)\.(?P<line>\d+)'
    r',(?P<break_char>[+@*=])(?P<unit>[^;]+);(?P<transcriber>[A-Z])>'
    r'\s+(?P<text>.+?)\s*$'
)
COMMENT_RE = re.compile(r'\{[^}]*\}')
WEIRDO_RE = re.compile(r'<[^>]*>')

def clean(raw):
    t = COMMENT_RE.sub('', raw)
    t = WEIRDO_RE.sub('', t)
    t = t.replace('!', '').replace('%', '')
    t = t.rstrip('-=')
    return t.strip()

def tokenize(cleaned):
    tokens = re.split(r'[.,]+', cleaned)
    return [t.strip() for t in tokens if t.strip() and t.strip() != '?']

# Parse with consensus
label_raw = defaultdict(lambda: defaultdict(dict))
with open('data/transcription/transcript.txt', 'r', encoding='latin-1') as f:
    for raw_line in f:
        raw_line = raw_line.rstrip('\n')
        m = LINE_RE.match(raw_line)
        if not m:
            continue
        unit = m.group('unit')
        if not unit.startswith('L'):
            continue
        folio = m.group('folio')
        line = m.group('line')
        tr = m.group('transcriber')
        text = clean(m.group('text'))
        words = tokenize(text)
        if words:
            label_raw[folio][(line, unit)][tr] = words

PREF = ['H', 'C', 'F', 'U', 'V', 'N']
all_labels = []

for folio in sorted(label_raw.keys()):
    sec = folio_type.get(folio, '?')
    for (line, unit), transcribers in sorted(label_raw[folio].items(),
                                              key=lambda x: int(x[0][0])):
        words = None
        for tr in PREF:
            if tr in transcribers:
                words = transcribers[tr]
                break
        if not words:
            words = list(transcribers.values())[0]

        all_labels.append({
            'folio': folio,
            'unit': unit,
            'line': int(line),
            'section': sec,
            'words': words,
            'text': ' '.join(words),
        })

# ---------------------------------------------------------------------------
# Build word → location index
# ---------------------------------------------------------------------------

word_locations = defaultdict(list)
for l in all_labels:
    for w in l['words']:
        word_locations[w].append(l)

# Also index full label text
text_locations = defaultdict(list)
for l in all_labels:
    text_locations[l['text']].append(l)

# Build paragraph word index too (for label→paragraph references)
para_word_folios = defaultdict(set)
for s in nlp['sentences']:
    for w in s['words']:
        para_word_folios[w].add(s['folio'])

# ---------------------------------------------------------------------------
# EVA glyph helper
# ---------------------------------------------------------------------------

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
def eva_glyph(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)

# Figure database for context
try:
    with open('data/lexicon/figure_database.json') as f:
        figs = json.load(f)
    figures = figs.get('figures', {})
except:
    figures = {}

# ---------------------------------------------------------------------------
# Generate the full label network report
# ---------------------------------------------------------------------------

# Group labels by folio
folio_labels = defaultdict(list)
for l in all_labels:
    folio_labels[l['folio']].append(l)

# Sort folios in manuscript order
def folio_sort_key(f):
    m = re.match(r'f(\d+)([rv])(\d*)', f)
    if m:
        return (int(m.group(1)), 0 if m.group(2) == 'r' else 1, int(m.group(3) or 0))
    return (999, 0, 0)

sorted_folios = sorted(folio_labels.keys(), key=folio_sort_key)

print("=" * 100)
print("VOYNICH MANUSCRIPT — COMPLETE LABEL NETWORK MAP")
print("=" * 100)
print(f"\nTotal labels: {len(all_labels)}")
print(f"Folios with labels: {len(folio_labels)}")
print(f"Unique label words: {len(word_locations)}")
print()

# For each folio, list all labels with their cross-references
for folio in sorted_folios:
    labels = folio_labels[folio]
    sec = folio_type.get(folio, '?')
    sec_name = TYPE_NAMES.get(sec, sec)

    # Get figure description
    fig_info = figures.get(folio, {})
    fig_desc = ', '.join(fig_info.get('content', []))
    zodiac = fig_info.get('zodiac', '')
    plant_id = fig_info.get('plant_id', '')

    print(f"{'━' * 100}")
    print(f"  {folio}  [{sec_name}]", end='')
    if zodiac:
        print(f"  Zodiac: {zodiac}", end='')
    if plant_id:
        print(f"  Plant: {plant_id}", end='')
    if fig_desc:
        print(f"\n  Content: {fig_desc}", end='')
    print()
    print(f"{'━' * 100}")

    for l in sorted(labels, key=lambda x: x['line']):
        text = l['text']
        glyph = ' '.join(eva_glyph(w) for w in l['words'])
        unit = l['unit']

        # Find cross-references for each word
        xrefs = []
        for w in l['words']:
            other_folios = set()
            # Label cross-refs
            for loc in word_locations[w]:
                if loc['folio'] != folio:
                    other_sec = TYPE_NAMES.get(loc['section'], loc['section'])
                    other_folios.add(f"{loc['folio']}[{other_sec[:3]}]")
            # Paragraph cross-refs (only if word appears as label elsewhere)
            para_folios = para_word_folios.get(w, set()) - {folio}
            # Only note paragraph refs if there are few (otherwise it's a common word)
            n_para = len(para_folios)

            if other_folios or (n_para > 0 and n_para <= 20):
                label_xref = ', '.join(sorted(other_folios)[:6])
                if len(other_folios) > 6:
                    label_xref += f' +{len(other_folios)-6}'
                xref_parts = []
                if label_xref:
                    xref_parts.append(f"label→ {label_xref}")
                if 0 < n_para <= 20:
                    sample = sorted(para_folios)[:4]
                    xref_parts.append(f"text({n_para}pp)")
                if xref_parts:
                    xrefs.append((w, ' | '.join(xref_parts)))

        # Print label
        print(f"\n  line {l['line']:3d}  {unit:4s}  {text:25s}  {glyph}")

        # Print cross-references
        if xrefs:
            for w, xref in xrefs:
                print(f"           {'':4s}  └─ {w:15s} → {xref}")
        else:
            print(f"           {'':4s}  └─ (no cross-references)")

print(f"\n{'━' * 100}")
print("END OF LABEL NETWORK MAP")
print(f"{'━' * 100}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

print(f"\n\n{'=' * 100}")
print("NETWORK SUMMARY")
print(f"{'=' * 100}")

# Most connected labels (appearing on most folios)
print(f"\nMost connected label words (by # of folios):")
connected = [(w, len(set(l['folio'] for l in locs)), locs)
             for w, locs in word_locations.items()
             if len(w) > 1]  # skip single chars
connected.sort(key=lambda x: -x[1])

print(f"{'Word':18s} {'Glyph':10s} {'Folios':>6s} Sections")
for w, n_folios, locs in connected[:25]:
    secs = sorted(set(TYPE_NAMES.get(l['section'], '?') for l in locs))
    folios = sorted(set(l['folio'] for l in locs))
    print(f"  {w:16s} {eva_glyph(w):10s} {n_folios:6d}  [{', '.join(secs)}]")
    print(f"    {'':16s} {'':10s}        {', '.join(folios)}")

# Section-bridging labels
print(f"\n\nLabels bridging sections:")
bridges = defaultdict(list)
for w, locs in word_locations.items():
    if len(w) <= 1:
        continue
    secs = set(l['section'] for l in locs)
    if len(secs) >= 2:
        sec_pair = tuple(sorted(secs))
        bridges[sec_pair].append(w)

for pair, words in sorted(bridges.items(), key=lambda x: -len(x[1])):
    pair_names = ' ↔ '.join(TYPE_NAMES.get(s, s) for s in pair)
    print(f"  {pair_names}: {', '.join(sorted(words)[:10])}")

# Export the full network as JSON for potential visualization
network_export = {
    'folios': {},
    'word_index': {},
}

for folio in sorted_folios:
    labels = folio_labels[folio]
    sec = folio_type.get(folio, '?')
    network_export['folios'][folio] = {
        'section': sec,
        'section_name': TYPE_NAMES.get(sec, sec),
        'labels': [
            {
                'line': l['line'],
                'unit': l['unit'],
                'text': l['text'],
                'words': l['words'],
            }
            for l in sorted(labels, key=lambda x: x['line'])
        ],
    }

for w, locs in word_locations.items():
    if len(w) <= 1:
        continue
    folios = sorted(set(l['folio'] for l in locs))
    if len(folios) >= 2:
        network_export['word_index'][w] = {
            'folios': folios,
            'sections': sorted(set(l['section'] for l in locs)),
        }

with open('data/lexicon/label_network.json', 'w') as f:
    json.dump(network_export, f, indent=2)
print(f"\nFull network exported to label_network.json")
