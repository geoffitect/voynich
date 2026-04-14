"""
Color Cross-Reference Analysis
=================================
Tag every herbal folio with the colors actually painted on the illustration
(from voynich.nu descriptions + manuscript observation). Then find words
enriched on pages with specific colors.

If a word appears ONLY on pages where red paint is used, it's a strong
candidate for "red." Same for blue, brown, green, etc.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)

metadata = nlp['metadata']
sentences = nlp['sentences']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}

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
def g(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)

# Build word freq per folio (herbal only)
folio_words = defaultdict(Counter)
for s in sentences:
    if folio_type.get(s['folio']) == 'H':
        folio_words[s['folio']].update(s['words'])

global_herbal_freq = Counter()
for wc in folio_words.values():
    global_herbal_freq.update(wc)
total_herbal = sum(global_herbal_freq.values())

# =====================================================================
# Color tags per herbal folio
# =====================================================================
# Colors: R=red, B=blue, G=green, Br=brown, W=white, Y=yellow, U=unpainted
# Tagged from voynich.nu descriptions and standard manuscript observations
# NOTE: Almost all plants have GREEN leaves (painted). We tag green only when
# it's the DOMINANT or NOTABLE color, not ubiquitous leaf color.

COLOR_TAGS = {
    # Quire 1
    'f1v':   {'Br': 'brown tuber/root'},
    'f2r':   {'B': 'blue dotted bulbs'},
    'f2v':   {'W': 'white water flower', 'G': 'green leaves'},
    'f3r':   {'R': 'red leaves alternating', 'W': 'white leaves', 'G': 'green leaves'},
    'f4r':   {'G': 'green leaves/berries'},
    'f4v':   {'G': 'green'},
    'f5r':   {'G': 'primarily green'},
    'f6r':   {'G': 'green pods', 'Br': 'brown decay patches'},
    'f6v':   {'G': 'green star leaves', 'B': 'blue details'},
    'f7r':   {'G': 'green', 'W': 'white/light flower'},
    'f7v':   {'B': 'blue dots/berries', 'G': 'green'},
    'f8r':   {'G': 'green leaf'},
    'f8v':   {'B': 'blue flowers', 'R': 'red flowers'},
    # Quire 2
    'f9r':   {'Br': 'brown/hairy roots'},
    'f9v':   {'B': 'blue flowers'},
    'f10r':  {'G': 'green'},
    'f10v':  {'G': 'green hanging leaves'},
    'f11r':  {'B': 'dark blue flowers', 'G': 'green bushel'},
    'f11v':  {'G': 'green dense bushel'},
    'f13r':  {'Br': 'brown root', 'G': 'green large round leaves'},
    'f13v':  {'G': 'green'},
    'f14r':  {'G': 'green arrow leaves'},
    'f14v':  {'G': 'green'},
    'f15r':  {'G': 'green complex'},
    'f15v':  {'G': 'green pinwheel leaves'},
    'f16r':  {'G': 'green starry leaves'},
    'f16v':  {'B': 'blue flower', 'R': 'red leaves'},
    # Quire 3
    'f17r':  {'B': 'dark blue flowers'},
    'f17v':  {'Br': 'brown/hairy bulbs', 'R': 'red berries/currants'},
    'f18r':  {'G': 'green'},
    'f18v':  {'G': 'green'},
    'f19r':  {'B': 'dark blue lily flower'},
    'f19v':  {'G': 'green'},
    'f20r':  {'G': 'green small leaves'},
    'f20v':  {'G': 'green'},
    'f21r':  {'G': 'green tiny leaves'},
    'f21v':  {'B': 'blue flowers', 'R': 'red flowers', 'G': 'alternating green/white'},
    'f22r':  {'G': 'green'},
    'f22v':  {'Br': 'brown thorny roots'},
    'f23r':  {'B': 'small blue flowers'},
    'f23v':  {'G': 'green'},
    'f24r':  {'G': 'green'},
    'f24v':  {'W': 'white flowers', 'B': 'blue flowers'},
    # Quire 4
    'f25r':  {'Br': 'small brown beans'},
    'f25v':  {'G': 'green'},
    'f26r':  {'B': 'blue flowers', 'Br': 'brown/plain roots'},
    'f26v':  {'B': 'many small blue flowers'},
    'f27r':  {'G': 'green'},
    'f27v':  {'W': 'white flowers', 'R': 'red hearts/centers', 'Br': 'brown root nails'},
    'f28r':  {'G': 'green crown leaves'},
    'f28v':  {'W': 'large white flower', 'G': 'green flower behind'},
    'f29r':  {'B': 'blue leaves alternating', 'G': 'green leaves alternating'},
    'f29v':  {'B': 'one blue flower', 'G': 'green bulbs', 'Br': 'needle roots'},
    'f30r':  {'B': 'small blue flowers', 'G': 'long green leaves'},
    'f30v':  {'Br': 'brown leaves', 'R': 'red/conspicuous root'},
    'f31r':  {'Br': 'long twisted brown root', 'U': 'unpainted flowers'},
    'f31v':  {'Br': 'brown leaf', 'G': 'green leaves'},
    'f32r':  {'W': 'white flowers', 'B': 'blue flowers', 'Y': 'yellow crown'},
    'f32v':  {'B': 'three blue flowers', 'G': 'green starry leaves'},
    # Quire 5
    'f33r':  {'G': 'green flowers', 'Br': 'brown humanoid root faces'},
    'f33v':  {'G': 'green', 'B': 'blue', 'R': 'red'},
    'f34r':  {'G': 'green'},
    'f34v':  {'Br': 'brown leaves', 'U': 'unpainted flowers'},
    'f35r':  {'B': 'blue bud', 'R': 'red bud', 'G': 'green'},
    'f35v':  {'Br': 'brown currants/berries', 'G': 'green oak leaves'},
    'f36r':  {'G': 'green leaves', 'Y': 'yellow leaves'},
    'f36v':  {'G': 'green finger leaves'},
    'f37r':  {'G': 'green', 'Br': 'brown berries'},
    'f37v':  {'G': 'green', 'W': 'white alternating', 'Br': 'brown humanoid root'},
    'f38r':  {'G': 'huge dark green leaf', 'W': 'white commas with dots'},
    'f38v':  {'B': 'one blue flower', 'G': 'green', 'Br': 'brown stacked'},
    'f39r':  {'G': 'green bed of flowers'},
    'f39v':  {'G': 'green dominant'},
    'f40r':  {'B': 'one blue flower', 'G': 'green'},
    'f40v':  {'G': 'green'},
    # Quire 6
    'f41r':  {'G': 'light green'},
    'f41v':  {'Br': 'bright brown roots', 'G': 'green frilled leaves'},
    'f42r':  {'G': 'green leaf', 'Br': 'bright brown small leaves'},
    'f42v':  {'G': 'green leaves'},
    'f43r':  {'G': 'green/white alternating leaves'},
    'f43v':  {'G': 'green', 'Br': 'brown snake root'},
    'f44r':  {'Br': 'brown thorny/hairy root'},
    'f44v':  {'R': 'red striped flower buds'},
    'f45r':  {'B': 'tiny blue flowers', 'G': 'green triangular'},
    'f45v':  {'B': 'small blue flowers', 'G': 'green'},
    'f46r':  {'G': 'green'},
    'f46v':  {'G': 'green large leaves'},
    'f47r':  {'G': 'green'},
    'f47v':  {'B': 'three small blue flowers', 'G': 'green large leaves'},
    'f48r':  {'G': 'green only leaves painted'},
    'f48v':  {'B': 'blue flowers', 'Y': 'yellow flowers', 'G': 'green'},
    # Additional herbal quires
    'f49r':  {'G': 'green'},
    'f49v':  {'G': 'green'},
    'f50r':  {'G': 'green'},
    'f50v':  {'G': 'green'},
    'f51r':  {'G': 'green'},
    'f51v':  {'G': 'green'},
    'f52r':  {'G': 'green', 'R': 'red'},
    'f52v':  {'G': 'green'},
    'f53r':  {'G': 'green'},
    'f53v':  {'G': 'green'},
    'f54r':  {'G': 'green'},
    'f54v':  {'G': 'green'},
    'f55r':  {'G': 'green'},
    'f55v':  {'G': 'green'},
    'f56r':  {'G': 'green'},
    'f56v':  {'G': 'green', 'B': 'blue'},
    'f65r':  {'G': 'green'},
    'f65v':  {'G': 'green'},
    'f66v':  {'G': 'green', 'Br': 'brown'},
    'f87r':  {'G': 'green'},
    'f87v':  {'G': 'green'},
    'f90r1': {'G': 'green'},
    'f90r2': {'G': 'green'},
    'f90v1': {'G': 'green'},
    'f90v2': {'G': 'green'},
    'f93r':  {'Br': 'brown pod', 'G': 'green'},
    'f93v':  {'R': 'red currants', 'G': 'green', 'Br': 'brown bulbous roots'},
    'f94r':  {'B': 'blue buds', 'G': 'green', 'Br': 'brown thorny roots'},
    'f94v':  {'G': 'green bulb/flower'},
    'f95r1': {'G': 'green/white alternating'},
    'f95r2': {'Br': 'brown roots', 'G': 'green'},
    'f95v1': {'B': 'blue flowers', 'Br': 'brown bulbous roots'},
    'f95v2': {'G': 'green'},
    'f96r':  {'G': 'green'},
    'f96v':  {'G': 'green', 'Br': 'brown bulbous root'},
}

# =====================================================================
# For each color, find enriched vocabulary
# =====================================================================

color_names = {
    'R': 'RED',
    'B': 'BLUE',
    'Br': 'BROWN',
    'W': 'WHITE',
    'Y': 'YELLOW',
    'G': 'GREEN',
    'U': 'UNPAINTED',
}

print("=" * 100)
print("COLOR CROSS-REFERENCE: Words enriched on pages with specific paint colors")
print("=" * 100)

# Summary of color distribution
for color_code, color_name in color_names.items():
    folios = [f for f, colors in COLOR_TAGS.items() if color_code in colors]
    print(f"  {color_name:10s}: {len(folios)} herbal folios")

results = {}

for color_code in ['R', 'B', 'Br', 'W', 'Y']:
    color_name = color_names[color_code]

    # Folios WITH this color
    with_color = {f for f, colors in COLOR_TAGS.items() if color_code in colors}
    # Folios WITHOUT (but still herbal with color tags)
    without_color = {f for f in COLOR_TAGS if color_code not in COLOR_TAGS.get(f, {})}

    # Must have word data
    with_color = {f for f in with_color if f in folio_words}
    without_color = {f for f in without_color if f in folio_words}

    if len(with_color) < 3 or len(without_color) < 3:
        continue

    # Word frequencies
    with_words = Counter()
    with_total = 0
    for f in with_color:
        with_words.update(folio_words[f])
        with_total += sum(folio_words[f].values())

    without_words = Counter()
    without_total = 0
    for f in without_color:
        without_words.update(folio_words[f])
        without_total += sum(folio_words[f].values())

    # Enrichment
    enriched = []
    for w, count in with_words.items():
        if count < 2:
            continue
        bg = without_words.get(w, 0)
        rate_with = count / max(with_total, 1)
        rate_without = max(bg, 0.5) / max(without_total, 1)
        lo = np.log2(rate_with / rate_without)
        if lo > 0.5:
            # Also check: what OTHER colors does this word appear with?
            word_colors = set()
            for f in with_color:
                if w in folio_words.get(f, {}):
                    for cc in COLOR_TAGS.get(f, {}):
                        word_colors.add(cc)
            for f in without_color:
                if w in folio_words.get(f, {}):
                    for cc in COLOR_TAGS.get(f, {}):
                        word_colors.add(cc)

            enriched.append({
                'word': w,
                'with_count': count,
                'without_count': bg,
                'log_odds': lo,
                'total': global_herbal_freq.get(w, 0),
                'pct_with': 100 * count / max(global_herbal_freq.get(w, 1), 1),
                'also_colors': sorted(word_colors - {color_code, 'G'}),  # exclude green (ubiquitous)
            })

    enriched.sort(key=lambda x: -x['log_odds'])
    results[color_code] = {
        'name': color_name,
        'n_folios': len(with_color),
        'n_words': with_total,
        'enriched': enriched,
    }

    print(f"\n{'━' * 100}")
    print(f"  {color_name} paint ({len(with_color)} folios, {with_total} words)")
    print(f"  Folios: {', '.join(sorted(with_color)[:12])}{'...' if len(with_color) > 12 else ''}")
    print(f"{'━' * 100}")
    print(f"  {'Word':18s} {'Glyph':12s} {'Here':>5s} {'Other':>5s} {'%Here':>6s} "
          f"{'LO':>6s} Also with colors")

    for item in enriched[:20]:
        w = item['word']
        also = ', '.join(color_names.get(c, c) for c in item['also_colors']) if item['also_colors'] else '(this color only!)'
        print(f"  {w:18s} {g(w):12s} {item['with_count']:5d} {item['without_count']:5d} "
              f"{item['pct_with']:5.0f}% {item['log_odds']:6.2f}  {also}")

# =====================================================================
# EXCLUSIVE color words: appear ONLY with one color
# =====================================================================

print(f"\n\n{'=' * 100}")
print("COLOR-EXCLUSIVE WORDS (appear only on pages with ONE non-green color)")
print("=" * 100)

for color_code in ['R', 'B', 'Br', 'W']:
    color_name = color_names[color_code]
    data = results.get(color_code)
    if not data:
        continue

    exclusives = [item for item in data['enriched']
                  if not item['also_colors'] or item['also_colors'] == []
                  and item['with_count'] >= 2]

    # Better: check which words appear ONLY on pages with this color
    # (excluding green which is everywhere)
    with_color_folios = {f for f, colors in COLOR_TAGS.items() if color_code in colors}
    true_exclusives = []
    for item in data['enriched']:
        if item['with_count'] < 2:
            continue
        w = item['word']
        # Check every folio this word appears on
        word_folios = {f for f in folio_words if w in folio_words[f]}
        # Does it ONLY appear on folios with this color?
        non_color_folios = word_folios - with_color_folios
        if not non_color_folios:
            true_exclusives.append(item)
        elif len(non_color_folios) <= 1 and item['pct_with'] >= 70:
            true_exclusives.append(item)

    if not true_exclusives:
        continue

    print(f"\n  {color_name} exclusives ({len(true_exclusives)} words):")
    print(f"  {'Word':18s} {'Glyph':12s} {'OnColor':>7s} {'Total':>5s} {'%Color':>6s} {'LO':>6s}")
    for item in true_exclusives[:15]:
        w = item['word']
        print(f"  {w:18s} {g(w):12s} {item['with_count']:7d} "
              f"{item['total']:5d} {item['pct_with']:5.0f}% {item['log_odds']:6.2f}")

# =====================================================================
# Cross-color comparison: same word, different colors
# =====================================================================

print(f"\n\n{'=' * 100}")
print("CROSS-COLOR COMPARISON: Do any words correlate with multiple specific colors?")
print("=" * 100)

# For each word, build its color profile
word_color_profile = defaultdict(lambda: defaultdict(int))
word_total_herbal = Counter()

for f, colors in COLOR_TAGS.items():
    if f not in folio_words:
        continue
    for w, c in folio_words[f].items():
        word_total_herbal[w] += c
        for color_code in colors:
            word_color_profile[w][color_code] += c

# Find words with strong single-color associations
print(f"\nWords with strongest single-color bias (freq >= 5):")
print(f"{'Word':18s} {'Glyph':12s} {'Total':>5s} {'R':>4s} {'B':>4s} {'Br':>4s} "
      f"{'W':>4s} {'Y':>3s} {'G':>4s} Best match")

color_candidates = []
for w in sorted(word_total_herbal, key=lambda x: -word_total_herbal[x]):
    total = word_total_herbal[w]
    if total < 5:
        continue
    profile = word_color_profile[w]
    # Normalize by how many folios have each color
    color_rates = {}
    for cc in ['R', 'B', 'Br', 'W', 'Y']:
        n_folios_with_color = len([f for f in COLOR_TAGS if cc in COLOR_TAGS.get(f, {})])
        if n_folios_with_color == 0:
            continue
        # What fraction of this word's occurrences are on pages with this color?
        on_color = profile.get(cc, 0)
        rate = on_color / total
        # Normalize by base rate of color
        base_rate = n_folios_with_color / len(COLOR_TAGS)
        if base_rate > 0:
            enrichment = rate / base_rate
            color_rates[cc] = (rate, enrichment, on_color)

    if not color_rates:
        continue

    # Find the best color match (highest enrichment)
    best_color = max(color_rates, key=lambda x: color_rates[x][1])
    best_rate, best_enrich, best_count = color_rates[best_color]

    if best_enrich > 1.5 and best_count >= 3:
        color_candidates.append({
            'word': w,
            'total': total,
            'profile': profile,
            'best_color': best_color,
            'best_enrichment': best_enrich,
            'best_count': best_count,
        })

color_candidates.sort(key=lambda x: -x['best_enrichment'])

for item in color_candidates[:40]:
    w = item['word']
    p = item['profile']
    best = item['best_color']
    best_name = color_names.get(best, best)
    enrich = item['best_enrichment']
    marker = ' ★★★' if enrich > 3 else ' ★★' if enrich > 2 else ' ★' if enrich > 1.5 else ''
    print(f"  {w:18s} {g(w):12s} {item['total']:5d}  "
          f"{p.get('R',0):4d} {p.get('B',0):4d} {p.get('Br',0):4d} "
          f"{p.get('W',0):4d} {p.get('Y',0):3d} {p.get('G',0):4d}  "
          f"{best_name} ({enrich:.1f}x){marker}")

# Export
export = {
    'color_tags': {f: {cc: desc for cc, desc in colors.items()} for f, colors in COLOR_TAGS.items()},
    'color_enrichment': {
        code: {
            'name': data['name'],
            'n_folios': data['n_folios'],
            'top_enriched': [
                {'word': i['word'], 'count': i['with_count'],
                 'log_odds': i['log_odds'], 'pct': i['pct_with']}
                for i in data['enriched'][:20]
            ],
        }
        for code, data in results.items()
    },
    'color_candidates': [
        {'word': c['word'], 'best_color': color_names[c['best_color']],
         'enrichment': c['best_enrichment'], 'count': c['best_count']}
        for c in color_candidates[:30]
    ],
}

with open('data/analysis/color_crossref.json', 'w') as f:
    json.dump(export, f, indent=2)
print(f"\nExported to color_crossref.json")
