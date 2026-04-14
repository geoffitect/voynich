"""
Visual Cross-Reference Analysis
=================================
Cross-reference word vectors with illustrated content to find
words/inflected forms that correlate with specific visual elements
(pools, nymphs, stars, plants, jars, zodiac animals).
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# Load everything
with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
with open('data/analysis/word_vectors.json') as f:
    vecs = json.load(f)
with open('data/lexicon/figure_database.json') as f:
    figs = json.load(f)

metadata = nlp['metadata']
sentences = nlp['sentences']
figures = figs['figures']

vocab = vecs['vocab']
vectors = np.array(vecs['vectors'])
word2idx = {w: i for i, w in enumerate(vocab)}

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

# Map folios to their visual object tags
folio_objects = {}
for folio, info in figures.items():
    folio_objects[folio] = set(info.get('objects', []))

# Build word frequency per folio
folio_words = defaultdict(Counter)
for s in sentences:
    folio = s['folio']
    folio_words[folio].update(s['words'])

# Define visual feature groups to test
VISUAL_FEATURES = {
    'pools_water': {
        'description': 'Pools, bathtubs, water features',
        'match_any': ['pool', 'bathtub', 'water', 'blue_water', 'green_water',
                      'outflow', 'blue_pool', 'green_pool', '2_green_pools',
                      '2_blue_pools', '3_sub_pools', '2_bathtubs'],
        'match_type': None,
    },
    'nymphs': {
        'description': 'Nymph/human figures (bath section)',
        'match_any': ['nymph', 'nymph_row', 'nymph_rows', 'crowned_nymph',
                      'fighting_nymphs', 'connected_nymphs', 'nymph_with_device'],
        'match_type': 'B',
    },
    'tubes_pipes': {
        'description': 'Tubes, pipes, organ-like structures',
        'match_any': ['tube', 'tubes', 'pipe', 'pipes', 'organ_pipes',
                      'connecting_tube', 'rainbow_tube', 'rainbow_tubes',
                      '4_element_openings', 'vertical_structure'],
        'match_type': None,
    },
    'stars_celestial': {
        'description': 'Stars, sun, moon (astronomical)',
        'match_any': ['sun_face', 'moon_face', 'star', 'star_labels',
                      'star_circle', '29_stars', '7_star_cluster', 'pleiades',
                      'blue_star'],
        'match_type': None,
    },
    'margin_stars': {
        'description': 'Marginal stars (recipe/stars section)',
        'match_any': ['margin_stars', '19_margin_stars', '14_margin_stars',
                      '13_tailed_stars', '10_margin_stars', '15_large_stars',
                      '15_margin_stars', '16_margin_stars', '17_margin_stars'],
        'match_type': 'S',
    },
    'zodiac_animals': {
        'description': 'Zodiac animal figures',
        'match_any': ['sheep', 'bull', 'lobster', 'lion', 'fish',
                      'balance_scales', 'couple', 'woman_with_star',
                      '2_lobsters', '2_fish'],
        'match_type': 'Z',
    },
    'nymphs_with_stars': {
        'description': 'Nymphs holding stars (zodiac pages)',
        'match_any': ['15_nymphs_with_stars', '29_nymphs_with_stars',
                      '30_nymphs_with_stars', 'nymphs_with_stars'],
        'match_type': 'Z',
    },
    'jars_containers': {
        'description': 'Pharmaceutical jars/containers',
        'match_any': ['3_jars', '4_jars', '2_jars', 'jar', 'flask',
                      'red_bucket', 'ornate_container'],
        'match_type': 'P',
    },
    'plant_fragments': {
        'description': 'Plant fragments in pharmaceutical pages',
        'match_any': ['plant_fragments', '12_plant_fragments', '10_plant_fragments',
                      '16_plant_fragments', '6_plant_fragments'],
        'match_type': 'P',
    },
    'globes_structures': {
        'description': 'Green globes, complex structures (bio section)',
        'match_any': ['green_globes', 'complex_structure', 'roofed_pool'],
        'match_type': None,
    },
}

# For each visual feature, find folios that have it
feature_folios = {}
for feat_name, feat_info in VISUAL_FEATURES.items():
    matching_folios = set()
    for folio, info in figures.items():
        objects = set(info.get('objects', []))
        content = set(info.get('content', []))
        all_tags = objects | content

        # Check object substring match
        matched = False
        for tag in all_tags:
            for pattern in feat_info['match_any']:
                if pattern in tag:
                    matched = True
                    break
            if matched:
                break

        # Also match by type if specified
        if feat_info['match_type'] and info.get('type') == feat_info['match_type']:
            matched = True

        if matched:
            matching_folios.add(folio)

    feature_folios[feat_name] = matching_folios

# =====================================================================
# For each visual feature, find words enriched on those folios
# =====================================================================

print("="*90)
print("VISUAL CROSS-REFERENCE: Words enriched near specific illustrations")
print("="*90)

# Global word frequency
global_freq = Counter()
for s in sentences:
    global_freq.update(s['words'])
total_words = sum(global_freq.values())

results = {}

for feat_name, feat_info in VISUAL_FEATURES.items():
    folios = feature_folios[feat_name]
    if not folios:
        continue

    # Word frequency on these folios
    feat_freq = Counter()
    feat_total = 0
    for folio in folios:
        for w, c in folio_words.get(folio, {}).items():
            feat_freq[w] += c
            feat_total += c

    if feat_total < 20:
        continue

    # Background = all other folios
    bg_total = total_words - feat_total
    bg_freq = {w: global_freq[w] - feat_freq.get(w, 0) for w in global_freq}

    # Log-odds ratio for enrichment
    enriched = []
    for w, count in feat_freq.items():
        if count < 3 or w not in word2idx:
            continue
        rate_feat = count / feat_total
        rate_bg = max(bg_freq.get(w, 0), 0.5) / bg_total
        log_odds = np.log2(rate_feat / rate_bg) if rate_bg > 0 else 0
        enriched.append({
            'word': w,
            'feat_count': count,
            'feat_rate': rate_feat * 1000,
            'bg_rate': rate_bg * 1000,
            'log_odds': log_odds,
            'total': global_freq[w],
            'pct_in_feat': 100 * count / global_freq[w],
        })

    enriched.sort(key=lambda x: -x['log_odds'])
    results[feat_name] = {
        'info': feat_info,
        'n_folios': len(folios),
        'n_words': feat_total,
        'enriched': enriched[:30],
        'folios': sorted(folios),
    }

    print(f"\n{'─'*90}")
    print(f"  {feat_info['description']} ({len(folios)} folios, {feat_total} words)")
    print(f"  Folios: {', '.join(sorted(folios)[:10])}{'...' if len(folios) > 10 else ''}")
    print(f"{'─'*90}")
    print(f"  {'Word':15s} {'Glyph':10s} {'Here':>5s} {'Total':>5s} {'%Here':>6s} {'LogOdds':>8s}")

    for item in enriched[:20]:
        w = item['word']
        glyph = eva_glyph(w)
        print(f"  {w:15s} {glyph:10s} {item['feat_count']:5d} {item['total']:5d} "
              f"{item['pct_in_feat']:5.0f}% {item['log_odds']:8.2f}")

# =====================================================================
# Compute feature centroids and find words closest to each
# =====================================================================

print(f"\n\n{'='*90}")
print("VECTOR-SPACE FEATURE CENTROIDS")
print("="*90)
print("(Words closest to the centroid of each visual-feature vocabulary)")

for feat_name, feat_data in results.items():
    enriched = feat_data['enriched']
    if len(enriched) < 5:
        continue

    # Build centroid from top enriched words
    vecs_list = []
    for item in enriched[:20]:
        wi = word2idx.get(item['word'])
        if wi is not None:
            vecs_list.append(vectors[wi] * item['log_odds'])  # weight by enrichment

    if not vecs_list:
        continue

    centroid = np.mean(vecs_list, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    # Find nearest words to centroid
    sims = vectors @ centroid
    top_idx = np.argsort(-sims)[:15]

    print(f"\n  {feat_data['info']['description']}:")
    for idx in top_idx:
        w = vocab[idx]
        glyph = eva_glyph(w)
        # Check if this word is enriched in this feature
        in_feat = any(item['word'] == w for item in enriched)
        marker = ' ★' if in_feat else ''
        print(f"    {w:15s} {glyph:10s}  sim={sims[idx]:.3f}{marker}")

# =====================================================================
# KEY TEST: Do "bath" words differ from "zodiac nymph" words?
# =====================================================================

# =====================================================================
# CRITICAL TEST: Pairwise section comparison
# =====================================================================

print(f"\n\n{'='*90}")
print("CRITICAL TEST: Pairwise section vocabulary comparison")
print("="*90)

# Use type codes directly for clean separation
section_types = {
    'B (Bio/Bath)': 'B',
    'H (Herbal)': 'H',
    'S (Stars)': 'S',
    'P (Pharma)': 'P',
    'C (Cosmo)': 'C',
    'T (Text)': 'T',
}

# Build per-section word counts
section_word_counts = {}
section_totals = {}
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}
for name, stype in section_types.items():
    wc = Counter()
    for s in sentences:
        if folio_type.get(s['folio']) == stype:
            wc.update(s['words'])
    section_word_counts[name] = wc
    section_totals[name] = sum(wc.values())

# Pairwise centroid similarities
print(f"\n  Section centroid cosine similarities:")
print(f"  {'':20s}", end='')
sec_names = [n for n in section_types if section_totals[n] > 100]
for n in sec_names:
    print(f" {n[:8]:>8s}", end='')
print()

sec_centroids = {}
for name in sec_names:
    wc = section_word_counts[name]
    vecs_list = []
    for w, c in wc.most_common(80):
        wi = word2idx.get(w)
        if wi is not None:
            vecs_list.append(vectors[wi] * c)
    if vecs_list:
        cent = np.mean(vecs_list, axis=0)
        cent /= np.linalg.norm(cent) + 1e-10
        sec_centroids[name] = cent

for n1 in sec_names:
    print(f"  {n1:20s}", end='')
    for n2 in sec_names:
        if n1 in sec_centroids and n2 in sec_centroids:
            sim = float(np.dot(sec_centroids[n1], sec_centroids[n2]))
            print(f" {sim:8.3f}", end='')
        else:
            print(f" {'---':>8s}", end='')
    print()

# Section-distinctive words (each section vs all others)
print(f"\n\n{'='*90}")
print("SECTION-DISTINCTIVE VOCABULARY (each section vs rest)")
print("="*90)

for name in sec_names:
    wc = section_word_counts[name]
    st = section_totals[name]
    if st < 50:
        continue

    rest_total = total_words - st
    enriched = []
    for w, count in wc.items():
        if count < 3 or w not in word2idx:
            continue
        bg = global_freq[w] - count
        rate_here = count / st
        rate_bg = max(bg, 0.5) / rest_total
        lo = np.log2(rate_here / rate_bg)
        enriched.append((w, count, global_freq[w], 100*count/global_freq[w], lo))

    enriched.sort(key=lambda x: -x[4])

    print(f"\n  {name} ({st} words):")
    print(f"  {'Word':18s} {'Glyph':12s} {'Sect':>5s} {'Tot':>5s} {'%Sect':>6s} {'LO':>6s}")
    for w, sc, tot, pct, lo in enriched[:20]:
        print(f"  {w:18s} {eva_glyph(w):12s} {sc:5d} {tot:5d} {pct:5.0f}% {lo:6.2f}")

# =====================================================================
# Bath (Bio) vs Herbal vs Stars: the three big sections head-to-head
# =====================================================================

print(f"\n\n{'='*90}")
print("HEAD-TO-HEAD: Bio(Bath) vs Herbal vs Stars — Unique vocabulary per section")
print("="*90)

for target, others in [
    ('B (Bio/Bath)', ['H (Herbal)', 'S (Stars)']),
    ('H (Herbal)', ['B (Bio/Bath)', 'S (Stars)']),
    ('S (Stars)', ['B (Bio/Bath)', 'H (Herbal)']),
]:
    if target not in sec_names:
        continue
    twc = section_word_counts[target]
    tt = section_totals[target]

    # Words that appear ONLY in this section (or almost only)
    exclusive = []
    for w, c in twc.items():
        if c < 3:
            continue
        other_count = sum(section_word_counts.get(o, {}).get(w, 0) for o in others)
        total_other = sum(section_totals.get(o, 0) for o in others)
        if global_freq[w] > 0:
            pct = 100 * c / global_freq[w]
            exclusive.append((w, c, other_count, global_freq[w], pct))

    exclusive.sort(key=lambda x: (-x[4], -x[1]))

    print(f"\n  {target} — words with highest % in this section:")
    print(f"  {'Word':18s} {'Glyph':12s} {'Here':>5s} {'Others':>6s} {'Total':>5s} {'%Here':>6s}")
    for w, c, oc, tot, pct in exclusive[:25]:
        if pct >= 50:  # at least 50% of occurrences in this section
            print(f"  {w:18s} {eva_glyph(w):12s} {c:5d} {oc:6d} {tot:5d} {pct:5.0f}%")

# =====================================================================
# Nearest neighbors in vector space for section-exclusive words
# =====================================================================

print(f"\n\n{'='*90}")
print("SEMANTIC NEIGHBORHOODS of section-exclusive words")
print("="*90)

# Pick top exclusive words from each section and show their nearest neighbors
def nearest_neighbors(word, n=8):
    wi = word2idx.get(word)
    if wi is None:
        return []
    sims = vectors @ vectors[wi]
    top_idx = np.argsort(-sims)[:n+1]
    return [(vocab[i], float(sims[i])) for i in top_idx if i != wi][:n]

spotlight_words = {
    'Bath/Bio exclusive': ['qol', 'olkain', 'oly', 'lol', 'qolchedy', 'raiin'],
    'Herbal exclusive': ['cthy', 'chor', 'chy', 'shol', 'sho', 'kchy'],
    'Stars exclusive': ['lkaiin', 'lkeey', 'chear', 'rain', 'lchedy', 'okeeo'],
    'Pharma exclusive': ['cheol', 'cheor', 'okeol', 'dol', 'sheol', 'qokol'],
}

for section, words in spotlight_words.items():
    print(f"\n  {section}:")
    for w in words:
        if w not in word2idx:
            continue
        nn = nearest_neighbors(w, n=6)
        nn_str = ', '.join(f"{nw}({ns:.2f})" for nw, ns in nn)
        glyph = eva_glyph(w)
        # What section is each neighbor from?
        print(f"    {w:15s} {glyph:10s} → {nn_str}")

# =====================================================================
# Export and HTML
# =====================================================================

export = {
    'feature_enrichment': {
        name: {
            'description': data['info']['description'],
            'n_folios': data['n_folios'],
            'n_words': data['n_words'],
            'top_enriched': data['enriched'][:20],
        }
        for name, data in results.items()
    },
}

with open('data/analysis/visual_crossref.json', 'w') as f:
    json.dump(export, f, indent=2)
print(f"\nExported to visual_crossref.json")

# Quick HTML
html_parts = ['<!DOCTYPE html><html><head><meta charset="UTF-8">',
    '<title>Voynich Visual Cross-Reference</title>',
    "<style>@font-face{font-family:'VoynichEVA';src:url('fonts/Voynich/VoynichEVA.ttf')format('truetype')}",
    "body{font-family:'Segoe UI',system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9;line-height:1.6}",
    "h1{color:#58a6ff;border-bottom:2px solid #1f6feb}h2{color:#79c0ff;margin-top:30px}",
    ".v{font-family:'VoynichEVA',serif;font-size:1.4em;color:#ffa657;letter-spacing:2px}",
    ".eva{font-family:monospace;background:#161b22;padding:2px 6px;border-radius:3px;color:#7ee787}",
    "table{border-collapse:collapse;margin:10px 0;background:#161b22}",
    "th,td{border:1px solid #30363d;padding:6px 10px;text-align:left}",
    "th{background:#21262d;color:#58a6ff}",
    "tr:hover{background:#1c2128}",
    ".note{background:#1c2128;border-left:4px solid #1f6feb;padding:12px 15px;margin:15px 0}",
    "</style></head><body>",
    "<h1>Visual Cross-Reference: Words Enriched Near Illustrations</h1>",
    "<div class='note'>For each type of illustration (pools, nymphs, stars, jars, etc.), ",
    "the words that appear disproportionately on those folios are listed. ",
    "Log-odds > 1 means the word is 2x more common near that illustration type.</div>",
]

for feat_name, data in results.items():
    html_parts.append(f"<h2>{data['info']['description']} ({data['n_folios']} folios)</h2>")
    html_parts.append("<table><tr><th>Word</th><th>Glyph</th><th>Here</th><th>Total</th><th>%Here</th><th>Log-odds</th></tr>")
    for item in data['enriched'][:15]:
        w = item['word']
        g = eva_glyph(w)
        html_parts.append(
            f"<tr><td><span class='eva'>{w}</span></td>"
            f"<td><span class='v'>{g}</span></td>"
            f"<td>{item['feat_count']}</td><td>{item['total']}</td>"
            f"<td>{item['pct_in_feat']:.0f}%</td>"
            f"<td>{item['log_odds']:.2f}</td></tr>")
    html_parts.append("</table>")

html_parts.append("</body></html>")
Path('reports/html/visual_crossref.html').write_text('\n'.join(html_parts), encoding='utf-8')
print("HTML report: visual_crossref.html")
