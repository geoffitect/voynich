"""
Hapax Legomena Analysis
========================
Catalog all words appearing exactly once (hapax legomena) and words
appearing exactly twice (dis legomena). In a natural language text,
hapax words are often:
  - Proper nouns (plant names, place names, star names)
  - Technical terms specific to one description
  - Compound forms or rare inflections

If the ms is a reference work, many hapax words should be unique names
for individual plants/stars/procedures — one name per illustration.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
with open('data/transcription/tokenized_corpus.json') as f:
    corpus = json.load(f)

metadata = nlp['metadata']
sentences = nlp['sentences']
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
def g(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)

# Token decomposition for structural analysis
SEGMENT_RULES = [
    ('aiii', 'V_AIII'), ('cth', 'C_CTH'), ('ckh', 'C_CKH'),
    ('cph', 'C_CPH'), ('cfh', 'C_CFH'), ('eee', 'V_EEE'),
    ('aii', 'V_AII'), ('ch', 'C_CH'), ('sh', 'C_SH'),
    ('qo', 'C_QO'), ('ee', 'V_EE'), ('ai', 'V_AI'),
]
SINGLE_TOKENS = {
    'o': 'V_O', 'e': 'V_E', 'y': 'V_Y', 'a': 'V_A', 'i': 'V_I',
    'd': 'C_D', 'l': 'C_L', 'k': 'C_K', 'r': 'C_R', 'n': 'C_N',
    't': 'C_T', 's': 'C_S', 'p': 'C_P', 'f': 'C_F', 'm': 'C_M',
    'g': 'C_G', 'q': 'C_Q', 'h': 'X_H', 'c': 'X_C', '?': 'X_UNK',
}

def tokenize(word):
    tokens = []
    i = 0
    while i < len(word):
        matched = False
        for pattern, tok in SEGMENT_RULES:
            if tok and word[i:i+len(pattern)] == pattern:
                tokens.append(tok)
                i += len(pattern)
                matched = True
                break
        if not matched:
            tokens.append(SINGLE_TOKENS.get(word[i], f'X_{word[i]}'))
            i += 1
    return tokens

# ---------------------------------------------------------------------------
# Build frequency and location data
# ---------------------------------------------------------------------------

freq = Counter()
word_folios = defaultdict(list)  # word -> [(folio, section, position_in_sentence)]
word_contexts = defaultdict(list)  # word -> [(prev_word, next_word)]

for s in sentences:
    folio = s['folio']
    sec = folio_type.get(folio, '?')
    words = s['words']
    for i, w in enumerate(words):
        freq[w] += 1
        word_folios[w].append((folio, sec))
        prev_w = words[i-1] if i > 0 else '<START>'
        next_w = words[i+1] if i < len(words)-1 else '<END>'
        word_contexts[w].append((prev_w, next_w))

total_words = sum(freq.values())
unique_types = len(freq)
hapax = {w for w, c in freq.items() if c == 1}
dis = {w for w, c in freq.items() if c == 2}

print("=" * 100)
print("HAPAX LEGOMENA ANALYSIS")
print("=" * 100)
print(f"\nTotal tokens: {total_words}")
print(f"Unique types: {unique_types}")
print(f"Hapax legomena (freq=1): {len(hapax)} ({100*len(hapax)/unique_types:.1f}% of types)")
print(f"Dis legomena (freq=2): {len(dis)} ({100*len(dis)/unique_types:.1f}% of types)")
print(f"Hapax + dis: {len(hapax)+len(dis)} ({100*(len(hapax)+len(dis))/unique_types:.1f}% of types)")

# For comparison: Zipf's law predicts ~50% hapax in natural language
# Cipher/random: much lower hapax rate
# Glossolalia: also lower (repetitive)

# ---------------------------------------------------------------------------
# Hapax by section
# ---------------------------------------------------------------------------

print(f"\n{'=' * 100}")
print("HAPAX DISTRIBUTION BY SECTION")
print(f"{'=' * 100}")

hapax_by_section = Counter()
dis_by_section = Counter()
words_by_section = Counter()

for w in hapax:
    for folio, sec in word_folios[w]:
        hapax_by_section[sec] += 1
for w in dis:
    for folio, sec in word_folios[w]:
        dis_by_section[sec] += 1
for s in sentences:
    sec = folio_type.get(s['folio'], '?')
    words_by_section[sec] += len(s['words'])

print(f"\n{'Section':10s} {'Words':>6s} {'Hapax':>6s} {'%Hapax':>7s} {'Dis':>5s} "
      f"{'Hapax/page':>10s}")
print("-" * 55)

section_folio_count = Counter()
for f, m in metadata.items():
    section_folio_count[m.get('illustration', '?')] += 1

for sec in sorted(words_by_section.keys()):
    n_words = words_by_section[sec]
    n_hapax = hapax_by_section[sec]
    n_dis = dis_by_section[sec]
    n_folios = section_folio_count.get(sec, 1)
    name = TYPE_NAMES.get(sec, sec)
    print(f"  {name:8s} {n_words:6d} {n_hapax:6d} {100*n_hapax/max(n_words,1):6.1f}% "
          f"{n_dis:5d} {n_hapax/max(n_folios,1):10.1f}")

# ---------------------------------------------------------------------------
# Structural analysis of hapax words
# ---------------------------------------------------------------------------

print(f"\n{'=' * 100}")
print("HAPAX STRUCTURAL ANALYSIS")
print(f"{'=' * 100}")

# Length distribution
hapax_lengths = Counter(len(w) for w in hapax)
common_lengths = Counter(len(w) for w, c in freq.items() if c >= 5)

print(f"\nWord length distribution (hapax vs common):")
print(f"{'Length':>6s} {'Hapax':>6s} {'%':>5s} {'Common':>7s} {'%':>5s}")
for length in range(2, 16):
    h = hapax_lengths.get(length, 0)
    c = common_lengths.get(length, 0)
    hp = 100 * h / max(len(hapax), 1)
    cp = 100 * c / max(sum(common_lengths.values()), 1)
    bar_h = '█' * int(hp)
    bar_c = '░' * int(cp)
    print(f"  {length:4d}   {h:6d} {hp:4.1f}% {bar_h}")
    print(f"  {'':4s}   {c:6d} {cp:4.1f}% {bar_c}")

# Do hapax words decompose into known stems + known endings?
print(f"\nHapax decomposition: known stem + known ending?")

known_stems = set()
known_endings = set()
for w, c in freq.items():
    if c >= 10 and len(w) >= 3:
        for split in range(2, min(5, len(w))):
            stem = w[:split]
            ending = w[split:]
            if len(ending) >= 1:
                known_stems.add(stem)
                known_endings.add(ending)

decomposable = 0
novel_stem = 0
novel_ending = 0
fully_novel = 0

hapax_decomposed = []

for w in hapax:
    if len(w) < 3:
        continue
    found = False
    for split in range(2, min(5, len(w))):
        stem = w[:split]
        ending = w[split:]
        if stem in known_stems and ending in known_endings:
            decomposable += 1
            found = True
            break
        elif stem in known_stems:
            novel_ending += 1
            found = True
            break
        elif ending in known_endings:
            novel_stem += 1
            hapax_decomposed.append((w, '?', stem, ending))
            found = True
            break
    if not found:
        fully_novel += 1
        hapax_decomposed.append((w, 'novel', w[:3], w[3:]))

total_analyzed = decomposable + novel_stem + novel_ending + fully_novel
print(f"  Known stem + known ending: {decomposable} ({100*decomposable/max(total_analyzed,1):.0f}%)")
print(f"  Known stem + novel ending: {novel_ending} ({100*novel_ending/max(total_analyzed,1):.0f}%)")
print(f"  Novel stem + known ending: {novel_stem} ({100*novel_stem/max(total_analyzed,1):.0f}%)")
print(f"  Fully novel: {fully_novel} ({100*fully_novel/max(total_analyzed,1):.0f}%)")

# ---------------------------------------------------------------------------
# Hapax as potential proper nouns / plant names
# ---------------------------------------------------------------------------

print(f"\n{'=' * 100}")
print("HAPAX AS PLANT/STAR/PROCEDURE NAMES")
print("(One unique word per folio = likely a proper noun)")
print(f"{'=' * 100}")

# For herbal pages: list hapax words that are the ONLY hapax on their folio
# These are strongest candidates for plant names

folio_hapax = defaultdict(list)
for w in hapax:
    for folio, sec in word_folios[w]:
        folio_hapax[folio].append(w)

# Also collect dis legomena per folio
folio_dis = defaultdict(list)
for w in dis:
    for folio, sec in word_folios[w]:
        if folio == word_folios[w][0][0]:  # only count on first folio
            folio_dis[folio].append(w)

print(f"\n--- HERBAL PAGES: Unique words per plant ---")
print(f"{'Folio':10s} {'#Hapax':>6s} {'#Dis':>5s} Top unique words (with context)")
print("-" * 100)

for folio in sorted(folio_hapax.keys(), key=lambda f: (
    {'H':0,'A':1,'S':2,'Z':3,'B':4,'P':5,'C':6,'T':7}.get(folio_type.get(f,'?'), 9),
    f)):
    sec = folio_type.get(folio, '?')
    if sec != 'H':
        continue

    hapax_words = folio_hapax[folio]
    dis_words = folio_dis.get(folio, [])

    if not hapax_words:
        continue

    # Sort by word length (longer = more likely proper noun)
    hapax_words.sort(key=lambda w: -len(w))

    # Get context for each
    details = []
    for w in hapax_words[:6]:
        ctx = word_contexts[w][0]
        prev, nxt = ctx
        tokens = tokenize(w)
        n_tokens = len(tokens)
        details.append(f"{w}({g(w)}) [{prev}→_→{nxt}]")

    details_str = '; '.join(details)
    more = f" +{len(hapax_words)-6}" if len(hapax_words) > 6 else ""
    print(f"  {folio:8s} {len(hapax_words):6d} {len(dis_words):5d}  {details_str}{more}")

# Same for other sections
for sec_code, sec_name in [('A', 'ASTRO'), ('S', 'STARS'), ('B', 'BIO/BATH'), ('P', 'PHARMA')]:
    print(f"\n--- {sec_name} PAGES: Unique words ---")
    print(f"{'Folio':10s} {'#Hapax':>6s} Top unique words")
    print("-" * 100)

    for folio in sorted(folio_hapax.keys()):
        sec = folio_type.get(folio, '?')
        if sec != sec_code:
            continue

        hapax_words = sorted(folio_hapax[folio], key=lambda w: -len(w))
        if not hapax_words:
            continue

        words_str = ', '.join(f"{w}({g(w)})" for w in hapax_words[:5])
        more = f" +{len(hapax_words)-5}" if len(hapax_words) > 5 else ""
        print(f"  {folio:8s} {len(hapax_words):6d}  {words_str}{more}")

# ---------------------------------------------------------------------------
# Hapax that look like compound words (contain known words as substrings)
# ---------------------------------------------------------------------------

print(f"\n{'=' * 100}")
print("COMPOUND HAPAX: Hapax containing known words as substrings")
print("(These may be compound terms or phrases written as one word)")
print(f"{'=' * 100}")

common_words = {w for w, c in freq.items() if c >= 15 and len(w) >= 3}

compounds = []
for w in hapax:
    if len(w) < 6:
        continue
    # Check if this hapax contains any common word as a substring
    parts_found = []
    for cw in common_words:
        if cw in w and cw != w:
            parts_found.append(cw)
    if len(parts_found) >= 2:
        # Sort parts by position in the word
        parts_found.sort(key=lambda p: w.index(p))
        folio = word_folios[w][0][0]
        sec = folio_type.get(folio, '?')
        compounds.append((w, parts_found, folio, sec))

compounds.sort(key=lambda x: -len(x[0]))

print(f"\n{len(compounds)} compound hapax found")
print(f"\n{'Hapax':25s} {'Glyph':15s} {'Parts':35s} {'Folio':10s} Section")
print("-" * 100)
for w, parts, folio, sec in compounds[:40]:
    parts_str = ' + '.join(parts)
    sec_name = TYPE_NAMES.get(sec, sec)
    print(f"  {w:23s} {g(w):15s} {parts_str:35s} {folio:10s} {sec_name}")

# ---------------------------------------------------------------------------
# Folio-level hapax density as "uniqueness score"
# ---------------------------------------------------------------------------

print(f"\n{'=' * 100}")
print("FOLIO UNIQUENESS SCORE")
print("(Folios with highest proportion of unique vocabulary)")
print(f"{'=' * 100}")

folio_word_counts = Counter()
for s in sentences:
    folio_word_counts[s['folio']] += len(s['words'])

folio_scores = []
for folio in sorted(folio_hapax.keys()):
    n_hapax = len(folio_hapax[folio])
    n_words = folio_word_counts.get(folio, 0)
    if n_words < 10:
        continue
    score = n_hapax / n_words
    sec = folio_type.get(folio, '?')
    folio_scores.append((folio, sec, n_hapax, n_words, score))

folio_scores.sort(key=lambda x: -x[4])

print(f"\n{'Folio':10s} {'Section':8s} {'Hapax':>6s} {'Words':>6s} {'Score':>6s} Top hapax")
print("-" * 100)
for folio, sec, nh, nw, score in folio_scores[:30]:
    sec_name = TYPE_NAMES.get(sec, sec)[:6]
    top = ', '.join(sorted(folio_hapax[folio], key=lambda w: -len(w))[:3])
    print(f"  {folio:8s} {sec_name:8s} {nh:6d} {nw:6d} {score:5.1%}  {top}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

export = {
    'statistics': {
        'total_tokens': total_words,
        'unique_types': unique_types,
        'hapax_count': len(hapax),
        'dis_count': len(dis),
        'hapax_pct': round(100 * len(hapax) / unique_types, 1),
    },
    'hapax_by_folio': {
        folio: {
            'section': folio_type.get(folio, '?'),
            'words': sorted(words, key=lambda w: -len(w)),
        }
        for folio, words in folio_hapax.items()
    },
    'compound_hapax': [
        {'word': w, 'parts': parts, 'folio': folio, 'section': sec}
        for w, parts, folio, sec in compounds[:100]
    ],
    'folio_uniqueness': [
        {'folio': f, 'section': s, 'hapax': nh, 'words': nw, 'score': round(sc, 3)}
        for f, s, nh, nw, sc in folio_scores
    ],
}

with open('data/analysis/hapax_analysis.json', 'w') as f:
    json.dump(export, f, indent=2)
print(f"\nExported to hapax_analysis.json")
