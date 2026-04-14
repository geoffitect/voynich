"""
Plant Feature Analysis
========================
Group herbal folios by shared visual features and find vocabulary
enriched within each feature group. Words that appear specifically
with plants sharing a visual trait are candidate "adjectives."
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
with open('data/analysis/word_vectors.json') as f:
    vecs = json.load(f)

metadata = nlp['metadata']
sentences = nlp['sentences']
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

# Build word freq per folio
folio_words = defaultdict(Counter)
for s in sentences:
    folio_words[s['folio']].update(s['words'])

global_freq = Counter()
for s in sentences:
    global_freq.update(s['words'])
total_words = sum(global_freq.values())

# =====================================================================
# Plant feature database (compiled from voynich.nu descriptions)
# =====================================================================

# Tag each herbal folio with visual features
# Features are binary tags: present or absent

PLANT_FEATURES = {
    # Quire 1
    'f1v':  {'bulbous_root': True, 'single_flower': True, 'broad_leaves': True},
    'f2r':  {'thin_roots': True, 'multiple_flowers': True, 'small_leaves': True},
    'f2v':  {'water_plant': True, 'veined_leaves': True, 'round_leaves': True},
    'f3r':  {'alternating_colors': True, 'stacked_leaves': True, 'no_flowers': True},
    'f4r':  {'small_leaves': True, 'small_flowers': True, 'berries': True},
    'f4v':  {'small_leaves': True, 'bulging_stem': True, 'multiple_flowers': True},
    'f5r':  {'green_dominant': True, 'leaves_connect_one_point': True},
    'f6r':  {'pods': True, 'veined_leaves': True, 'decay_patches': True},
    'f6v':  {'star_leaves': True, 'spiky_fruits': True, 'veined_leaves': True},
    'f7r':  {'large_flower': True, 'symmetrical': True, 'simple_roots': True, 'pinwheel': True},
    'f7v':  {'blue_dots': True, 'berries': True, 'pinwheel': True},
    'f8r':  {'single_large_leaf': True, 'veined_leaves': True, 'no_flowers': True},
    'f8v':  {'blue_flowers': True, 'red_flowers': True, 'plain_roots': True},
    # Quire 2
    'f9r':  {'hairy_roots': True, 'flat_topped_root': True, 'seeds': True},
    'f9v':  {'plain_roots': True, 'veined_leaves': True, 'blue_flowers': True},
    'f10v': {'hanging_leaves': True, 'calyx': True},
    'f11r': {'flat_topped_root': True, 'two_leaf_shapes': True, 'blue_flowers': True, 'dense_bushel': True},
    'f11v': {'dense_bushel': True, 'single_flower': True},
    'f13r': {'brown_root': True, 'round_leaves': True, 'veined_leaves': True, 'flat_topped_root': True},
    'f14r': {'arrow_leaves': True, 'single_flower': True, 'flat_topped_root': True},
    'f14v': {'large_leaves': True, 'multiple_flowers': True, 'three_stems': True},
    'f15r': {'large_complex': True, 'two_leaf_shapes': True, 'multiple_flowers': True},
    'f15v': {'pinwheel': True, 'tendrils': True, 'large_leaves': True},
    'f16r': {'star_leaves': True, 'seeds': True, 'flat_topped_root': True},
    'f16v': {'blue_flowers': True, 'red_leaves': True, 'veined_leaves': True, 'flat_topped_root': True},
    # Quire 3
    'f17r': {'thin_roots': True, 'thin_leaves': True, 'blue_flowers': True},
    'f17v': {'bulbous_root': True, 'berries': True},
    'f18r': {'plain_roots': True, 'two_leaf_shapes': True, 'calyx': True, 'multiple_flowers': True},
    'f19r': {'veined_leaves': True, 'two_leaf_shapes': True, 'flat_topped_root': True, 'blue_flowers': True},
    'f20r': {'small_leaves': True, 'veined_leaves': True, 'small_flowers': True},
    'f20v': {'three_stems': True, 'narrow_leaves': True, 'multiple_flowers': True},
    'f21r': {'tiny_leaves': True, 'pinwheel': True, 'berries': True},
    'f22r': {'reconnecting_stem': True, 'multiple_flowers': True},
    'f22v': {'brown_root': True, 'thorny_roots': True, 'multiple_flowers': True},
    'f23r': {'flat_topped_root': True, 'blue_flowers': True, 'two_plants': True},
    'f24r': {'tall_flowers': True, 'two_plants': True},
    # Quire 4
    'f25r': {'veined_leaves': True, 'beans': True},
    'f25v': {'pinwheel': True, 'veined_leaves': True, 'no_flowers': True, 'dragon_animal': True},
    'f26r': {'plain_roots': True, 'blue_flowers': True, 'two_stalks': True},
    'f27r': {'veined_leaves': True, 'two_leaf_shapes': True, 'small_flowers': True, 'two_plants': True},
    'f27v': {'flat_topped_root': True, 'thorny_roots': True, 'white_flowers': True, 'red_center': True},
    'f28r': {'simple_roots': True, 'crown_leaves': True, 'pinwheel': True},
    'f28v': {'large_flower': True, 'veined_leaves': True, 'two_leaf_shapes': True, 'white_flowers': True},
    'f29r': {'alternating_colors': True},
    'f29v': {'needle_roots': True, 'bulbs': True, 'blue_flowers': True},
    'f30r': {'long_leaves': True, 'blue_flowers': True, 'veined_leaves': True},
    'f30v': {'conspicuous_root': True, 'brown_leaves': True, 'berries': True, 'tendrils': True},
    'f31r': {'twisted_root': True, 'brown_root': True, 'calyx': True},
    'f32r': {'small_leaves': True, 'white_flowers': True, 'blue_flowers': True, 'yellow_crown': True},
    'f32v': {'large_root': True, 'star_leaves': True, 'blue_flowers': True, 'tall_plant': True},
    # Quire 5
    'f33r': {'humanoid_root': True, 'green_flowers': True},
    'f33v': {'star_leaves': True, 'huge_flowers': True, 'dead_flower': True},
    'f34r': {'large_root': True, 'parallel_leaves': True, 'calyx': True, 'multiple_flowers': True},
    'f34v': {'extensive_roots': True, 'animal_root': True, 'brown_leaves': True},
    'f35r': {'blue_flowers': True, 'red_flowers': True, 'leaves_connect_one_point': True},
    'f35v': {'strong_roots': True, 'oak_leaves': True, 'berries': True, 'currants': True},
    'f36r': {'flat_topped_root': True, 'veined_leaves': True, 'small_buds': True},
    'f36v': {'finger_leaves': True, 'seeds': True},
    'f37r': {'oblong_leaves': True, 'berries': True, 'hanging_leaves': True},
    'f37v': {'humanoid_root': True, 'alternating_colors': True, 'flat_topped_root': True},
    'f38r': {'huge_leaf': True, 'tiny_buds': True, 'white_dots': True},
    'f38v': {'three_colors': True, 'blue_flowers': True, 'bush': True},
    'f39r': {'root_platform': True, 'parallel_leaves': True, 'calyx': True},
    'f39v': {'large_root': True, 'flat_topped_root': True, 'parallel_leaves': True, 'multiple_flowers': True},
    'f40r': {'loop_stem': True, 'parallel_leaves': True, 'blue_flowers': True},
    'f40v': {'two_plants': True, 'parallel_leaves': True, 'calyx': True, 'huge_flowers': True},
    # Quire 6
    'f41v': {'brown_root': True, 'finger_leaves': True, 'seeds': True, 'veined_leaves': True},
    'f42v': {'narrow_roots': True, 'thin_stem': True, 'single_flower': True},
    'f43r': {'extensive_roots': True, 'root_platform': True, 'small_leaves': True, 'alternating_colors': True, 'calyx': True},
    'f43v': {'snake_root': True, 'bushy_root': True, 'crescent_leaves': True, 'two_plants': True},
    'f44r': {'thick_stem': True, 'thorny_roots': True, 'small_flower': True},
    'f45r': {'conspicuous_root': True, 'flat_topped_root': True, 'triangular_leaves': True, 'blue_flowers': True, 'three_stems': True},
    'f45v': {'large_root': True, 'flat_topped_root': True, 'three_stems': True, 'blue_flowers': True, 'small_leaves': True},
    'f46r': {'root_platform': True, 'three_stems': True, 'parallel_leaves': True},
    'f46v': {'eagle_root': True, 'large_leaves': True, 'spiraling_stem': True, 'calyx': True},
    'f47r': {'thin_stem': True, 'large_leaves': True, 'no_flowers': True},
    'f47v': {'plain_roots': True, 'large_leaves': True, 'blue_flowers': True},
    'f48v': {'crossing_roots': True, 'two_plants': True, 'blue_flowers': True, 'calyx': True},
}

# =====================================================================
# Group by features and find enriched vocabulary
# =====================================================================

# Get all unique features
all_features = set()
for feats in PLANT_FEATURES.values():
    all_features.update(feats.keys())

# Herbal folios only
herbal_folios = set(PLANT_FEATURES.keys())

# For each feature, compute enriched words
print("="*90)
print("PLANT FEATURE → VOCABULARY ENRICHMENT")
print("="*90)

feature_results = {}

for feature in sorted(all_features):
    # Folios WITH this feature
    with_feat = {f for f, feats in PLANT_FEATURES.items() if feats.get(feature)}
    without_feat = herbal_folios - with_feat

    if len(with_feat) < 3 or len(without_feat) < 3:
        continue

    # Word counts
    with_words = Counter()
    with_total = 0
    for f in with_feat:
        for w, c in folio_words.get(f, {}).items():
            with_words[w] += c
            with_total += c

    without_words = Counter()
    without_total = 0
    for f in without_feat:
        for w, c in folio_words.get(f, {}).items():
            without_words[w] += c
            without_total += c

    if with_total < 30 or without_total < 30:
        continue

    # Log-odds enrichment
    enriched = []
    for w, count in with_words.items():
        if count < 2:
            continue
        bg = without_words.get(w, 0)
        rate_with = count / with_total
        rate_without = max(bg, 0.5) / without_total
        lo = np.log2(rate_with / rate_without)
        if lo > 0.5:  # at least 1.4x enriched
            enriched.append({
                'word': w,
                'with_count': count,
                'without_count': bg,
                'total': global_freq.get(w, count + bg),
                'log_odds': lo,
            })

    enriched.sort(key=lambda x: -x['log_odds'])

    feature_results[feature] = {
        'n_folios_with': len(with_feat),
        'n_folios_without': len(without_feat),
        'folios': sorted(with_feat),
        'n_words': with_total,
        'enriched': enriched[:15],
    }

# Print results sorted by number of enriched words (most informative features first)
sorted_features = sorted(feature_results.items(),
                         key=lambda x: -len(x[1]['enriched']))

for feature, data in sorted_features:
    enriched = data['enriched']
    if len(enriched) < 3:
        continue

    print(f"\n{'─'*90}")
    print(f"  Feature: {feature.upper().replace('_', ' ')}")
    print(f"  {data['n_folios_with']} folios WITH vs {data['n_folios_without']} WITHOUT  "
          f"({data['n_words']} words)")
    print(f"  Folios: {', '.join(data['folios'][:8])}{'...' if len(data['folios']) > 8 else ''}")
    print(f"{'─'*90}")
    print(f"  {'Word':18s} {'Glyph':12s} {'With':>5s} {'W/O':>5s} {'LO':>6s}")

    for item in enriched[:12]:
        w = item['word']
        glyph = eva_glyph(w)
        print(f"  {w:18s} {glyph:12s} {item['with_count']:5d} {item['without_count']:5d} "
              f"{item['log_odds']:6.2f}")

# =====================================================================
# Cross-feature comparison: find "adjective" candidates
# =====================================================================

print(f"\n\n{'='*90}")
print("ADJECTIVE CANDIDATES: Words enriched for ONE feature but not others")
print("="*90)

# A true adjective should be enriched for exactly one visual feature
word_feature_enrichment = defaultdict(list)
for feature, data in feature_results.items():
    for item in data['enriched'][:10]:
        word_feature_enrichment[item['word']].append(
            (feature, item['log_odds'], item['with_count']))

# Words enriched in exactly 1 feature (most likely adjectives)
single_feature_words = []
for w, features in word_feature_enrichment.items():
    if len(features) == 1:
        feat, lo, count = features[0]
        if count >= 3 and lo > 1.0:
            single_feature_words.append((w, feat, lo, count))

single_feature_words.sort(key=lambda x: -x[2])

print(f"\nWords enriched in exactly ONE plant feature (strongest adjective candidates):")
print(f"{'Word':18s} {'Glyph':12s} {'Feature':25s} {'LO':>6s} {'Count':>5s}")
print("-"*70)
for w, feat, lo, count in single_feature_words[:30]:
    print(f"  {w:16s} {eva_glyph(w):12s} {feat:25s} {lo:6.2f} {count:5d}")

# Words enriched in 2+ features (possibly more general descriptors)
multi_feature_words = []
for w, features in word_feature_enrichment.items():
    if len(features) >= 2:
        feats_str = ', '.join(f"{f}({lo:.1f})" for f, lo, _ in features)
        total_count = sum(c for _, _, c in features)
        multi_feature_words.append((w, features, feats_str, total_count))

multi_feature_words.sort(key=lambda x: -len(x[1]))

print(f"\n\nWords enriched in MULTIPLE features (general descriptors):")
print(f"{'Word':18s} {'Glyph':12s} {'#Feat':>5s} Features")
print("-"*90)
for w, features, feats_str, total in multi_feature_words[:20]:
    print(f"  {w:16s} {eva_glyph(w):12s} {len(features):5d}   {feats_str}")

# =====================================================================
# Root type comparison
# =====================================================================

print(f"\n\n{'='*90}")
print("ROOT TYPE VOCABULARY COMPARISON")
print("="*90)

root_types = {
    'bulbous/large root': {'bulbous_root', 'large_root', 'conspicuous_root'},
    'flat-topped root': {'flat_topped_root'},
    'thin/plain root': {'thin_roots', 'plain_roots', 'simple_roots', 'narrow_roots'},
    'elaborate root': {'humanoid_root', 'animal_root', 'eagle_root', 'snake_root',
                       'twisted_root', 'thorny_roots', 'hairy_roots', 'needle_roots'},
    'root platform': {'root_platform', 'extensive_roots'},
}

for root_name, root_features in root_types.items():
    folios = {f for f, feats in PLANT_FEATURES.items()
              if any(feats.get(rf) for rf in root_features)}
    if len(folios) < 3:
        continue

    # Word counts on these folios vs rest of herbal
    with_words = Counter()
    with_total = 0
    for f in folios:
        for w, c in folio_words.get(f, {}).items():
            with_words[w] += c
            with_total += c

    other_herbal = herbal_folios - folios
    without_words = Counter()
    without_total = 0
    for f in other_herbal:
        for w, c in folio_words.get(f, {}).items():
            without_words[w] += c
            without_total += c

    enriched = []
    for w, count in with_words.items():
        if count < 2:
            continue
        bg = without_words.get(w, 0)
        rate_with = count / max(with_total, 1)
        rate_without = max(bg, 0.5) / max(without_total, 1)
        lo = np.log2(rate_with / rate_without)
        if lo > 0.8:
            enriched.append((w, count, bg, lo))

    enriched.sort(key=lambda x: -x[3])

    print(f"\n  {root_name} ({len(folios)} folios: {', '.join(sorted(folios)[:6])}{'...' if len(folios) > 6 else ''})")
    print(f"  {'Word':18s} {'Glyph':12s} {'Here':>5s} {'Other':>5s} {'LO':>6s}")
    for w, c, bg, lo in enriched[:10]:
        print(f"  {w:18s} {eva_glyph(w):12s} {c:5d} {bg:5d} {lo:6.2f}")

# =====================================================================
# Flower/color comparison
# =====================================================================

print(f"\n\n{'='*90}")
print("FLOWER & COLOR VOCABULARY COMPARISON")
print("="*90)

color_features = {
    'blue flowers': {'blue_flowers'},
    'red flowers/leaves': {'red_flowers', 'red_leaves', 'red_center'},
    'white flowers': {'white_flowers'},
    'no flowers': {'no_flowers'},
    'multiple flowers': {'multiple_flowers'},
    'single flower': {'single_flower'},
    'berries/seeds': {'berries', 'seeds', 'currants'},
}

for color_name, color_feats in color_features.items():
    folios = {f for f, feats in PLANT_FEATURES.items()
              if any(feats.get(cf) for cf in color_feats)}
    if len(folios) < 3:
        continue

    with_words = Counter()
    with_total = 0
    for f in folios:
        for w, c in folio_words.get(f, {}).items():
            with_words[w] += c
            with_total += c

    other = herbal_folios - folios
    without_words = Counter()
    without_total = 0
    for f in other:
        for w, c in folio_words.get(f, {}).items():
            without_words[w] += c
            without_total += c

    enriched = []
    for w, count in with_words.items():
        if count < 2:
            continue
        bg = without_words.get(w, 0)
        rate_with = count / max(with_total, 1)
        rate_without = max(bg, 0.5) / max(without_total, 1)
        lo = np.log2(rate_with / rate_without)
        if lo > 0.8:
            enriched.append((w, count, bg, lo))

    enriched.sort(key=lambda x: -x[3])

    print(f"\n  {color_name} ({len(folios)} folios)")
    print(f"  {'Word':18s} {'Glyph':12s} {'Here':>5s} {'Other':>5s} {'LO':>6s}")
    for w, c, bg, lo in enriched[:10]:
        print(f"  {w:18s} {eva_glyph(w):12s} {c:5d} {bg:5d} {lo:6.2f}")

# Export
export = {
    'plant_features': {f: list(feats.keys()) for f, feats in PLANT_FEATURES.items()},
    'feature_enrichment': {
        feat: {
            'folios': data['folios'],
            'enriched': data['enriched'][:15],
        }
        for feat, data in feature_results.items()
    },
    'adjective_candidates': [
        {'word': w, 'feature': f, 'log_odds': lo, 'count': c}
        for w, f, lo, c in single_feature_words[:50]
    ],
}

with open('data/analysis/plant_analysis.json', 'w') as f:
    json.dump(export, f, indent=2)
print(f"\nExported to plant_analysis.json")
