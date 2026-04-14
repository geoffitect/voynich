"""
Voynich Distributional Vectors
================================
Build word vectors from the tokenized corpus using PMI-weighted
co-occurrence matrices + SVD dimensionality reduction.

With ~35k tokens the corpus is small, so we use:
  - Word-word co-occurrence in a ±3 window (sentence-bounded)
  - Positive PMI weighting (PPMI)
  - Truncated SVD to 50 dimensions
  - Cosine similarity for nearest neighbors

Then test whether the vector space encodes genuine semantic structure
by checking:
  1. Do section-specific words cluster? (herbal vs astro vs pharma)
  2. Do declined forms of the same stem cluster?
  3. Do zodiac/star labels form coherent groups?
  4. Can we find analogies (a:b :: c:d)?
  5. Do word clusters correlate with illustrated content?
"""

import json
import math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Load tokenized corpus
# ---------------------------------------------------------------------------

with open('data/transcription/tokenized_corpus.json') as f:
    corpus = json.load(f)

with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)

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

def eva_glyph(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)

# Build flat sentence list with section labels
sentences = []
for s in corpus['tokenized_sentences']:
    folio = s['folio']
    sec = folio_type.get(folio, '?')
    eva_words = s['eva_words']
    if len(eva_words) >= 2:
        sentences.append({
            'folio': folio,
            'section': sec,
            'words': eva_words,
        })

# Global vocabulary (words with freq >= 3)
word_freq = Counter()
for s in sentences:
    word_freq.update(s['words'])

MIN_FREQ = 3
vocab = {w for w, c in word_freq.items() if c >= MIN_FREQ}
vocab_list = sorted(vocab, key=lambda w: -word_freq[w])
word2idx = {w: i for i, w in enumerate(vocab_list)}
V = len(vocab_list)

print(f"Vocabulary: {V} words (freq >= {MIN_FREQ})")
print(f"Sentences: {len(sentences)}")

# ---------------------------------------------------------------------------
# Co-occurrence matrix (±3 window, sentence-bounded)
# ---------------------------------------------------------------------------

WINDOW = 3
print(f"\nBuilding co-occurrence matrix (window=±{WINDOW})...")

cooccur = defaultdict(Counter)
word_count = Counter()

for s in sentences:
    words = s['words']
    for i, w in enumerate(words):
        if w not in vocab:
            continue
        word_count[w] += 1
        for j in range(max(0, i - WINDOW), min(len(words), i + WINDOW + 1)):
            if j == i:
                continue
            ctx = words[j]
            if ctx not in vocab:
                continue
            cooccur[w][ctx] += 1

total_cooccur = sum(sum(c.values()) for c in cooccur.values())
print(f"  Total co-occurrence pairs: {total_cooccur}")

# ---------------------------------------------------------------------------
# PPMI matrix
# ---------------------------------------------------------------------------

print("Computing PPMI matrix...")

# Build sparse PPMI
# PMI(w, c) = log2(P(w,c) / (P(w) * P(c)))
# PPMI = max(0, PMI)

total_word_count = sum(word_count.values())

# For efficiency, build as dense numpy array (V is manageable ~2-3k)
ppmi = np.zeros((V, V), dtype=np.float32)

for w, ctxs in cooccur.items():
    wi = word2idx.get(w)
    if wi is None:
        continue
    p_w = word_count[w] / total_word_count
    for c, count in ctxs.items():
        ci = word2idx.get(c)
        if ci is None:
            continue
        p_c = word_count[c] / total_word_count
        p_wc = count / total_cooccur
        pmi = math.log2(p_wc / (p_w * p_c)) if p_w * p_c > 0 else 0
        ppmi[wi, ci] = max(0, pmi)

print(f"  PPMI matrix shape: {ppmi.shape}")
print(f"  Non-zero entries: {np.count_nonzero(ppmi)}")

# ---------------------------------------------------------------------------
# SVD dimensionality reduction
# ---------------------------------------------------------------------------

DIM = 50
print(f"\nSVD reduction to {DIM} dimensions...")

from numpy.linalg import svd

# Truncated SVD
U, S, Vt = svd(ppmi, full_matrices=False)
# Word vectors = U[:, :DIM] * sqrt(S[:DIM])
word_vectors = U[:, :DIM] * np.sqrt(S[:DIM])

# Normalize to unit length
norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
norms[norms == 0] = 1
word_vectors = word_vectors / norms

print(f"  Word vectors shape: {word_vectors.shape}")

# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def cosine_sim(w1, w2):
    i1, i2 = word2idx.get(w1), word2idx.get(w2)
    if i1 is None or i2 is None:
        return None
    return float(np.dot(word_vectors[i1], word_vectors[i2]))

def nearest_neighbors(word, n=10):
    wi = word2idx.get(word)
    if wi is None:
        return []
    sims = word_vectors @ word_vectors[wi]
    top_idx = np.argsort(-sims)[:n+1]
    return [(vocab_list[i], float(sims[i])) for i in top_idx if i != wi][:n]

def analogy(a, b, c, n=5):
    """a is to b as c is to ?"""
    ai, bi, ci = word2idx.get(a), word2idx.get(b), word2idx.get(c)
    if any(x is None for x in [ai, bi, ci]):
        return []
    vec = word_vectors[bi] - word_vectors[ai] + word_vectors[ci]
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    sims = word_vectors @ vec
    # Exclude the input words
    exclude = {ai, bi, ci}
    top_idx = np.argsort(-sims)
    results = []
    for i in top_idx:
        if i not in exclude:
            results.append((vocab_list[i], float(sims[i])))
            if len(results) >= n:
                break
    return results

# ---------------------------------------------------------------------------
# TEST 1: Section-specific clustering
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TEST 1: SECTION-SPECIFIC WORD CLUSTERING")
print(f"{'='*80}")

# For each section, find the centroid of its words and the words closest to that centroid
section_words = defaultdict(Counter)
for s in sentences:
    sec = s['section']
    for w in s['words']:
        if w in vocab:
            section_words[sec][w] += 1

for sec in ['H', 'S', 'P', 'B']:
    name = TYPE_NAMES.get(sec, sec)
    # Get section centroid
    sec_vecs = []
    for w, c in section_words[sec].most_common(100):
        wi = word2idx.get(w)
        if wi is not None:
            sec_vecs.append(word_vectors[wi] * c)
    if not sec_vecs:
        continue
    centroid = np.mean(sec_vecs, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    # Find words closest to this centroid
    sims = word_vectors @ centroid
    top_idx = np.argsort(-sims)[:15]

    print(f"\n  {name} section centroid — nearest words:")
    for i in top_idx:
        w = vocab_list[i]
        # What % of this word's occurrences are in this section?
        in_sec = section_words[sec].get(w, 0)
        total = word_freq[w]
        pct = 100 * in_sec / total if total > 0 else 0
        glyph = eva_glyph(w)
        print(f"    {w:15s} {glyph:10s}  sim={sims[i]:.3f}  "
              f"sect={in_sec}/{total} ({pct:.0f}%)")

# ---------------------------------------------------------------------------
# TEST 2: Declined forms cluster together
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TEST 2: DECLINED FORMS — DO INFLECTIONS OF THE SAME STEM CLUSTER?")
print(f"{'='*80}")

# Test pairs where we know the stem is shared (from declension analysis)
test_families = [
    ('dain/daiin', ['dain', 'daiin', 'daiiin', 'dair', 'dar', 'dal', 'dam']),
    ('ain/aiin', ['ain', 'aiin', 'aiiin', 'air', 'ar', 'al', 'am']),
    ('chedy/cheedy', ['chedy', 'cheedy', 'chey', 'cheey', 'cheol', 'cheor']),
    ('shedy/sheedy', ['shedy', 'sheedy', 'shey', 'sheey', 'sheol', 'sheor']),
    ('okain/okaiin', ['okain', 'okaiin', 'okal', 'okar', 'okam']),
    ('qokedy/qokeedy', ['qokedy', 'qokeedy', 'qokeey', 'qokey', 'qokain', 'qokaiin']),
    ('otedy/oteedy', ['otedy', 'oteedy', 'otaiin', 'otain', 'otal', 'otar']),
]

for family_name, members in test_families:
    present = [w for w in members if w in word2idx]
    if len(present) < 3:
        continue

    # Compute average pairwise similarity within the family
    sims_within = []
    for i in range(len(present)):
        for j in range(i+1, len(present)):
            s = cosine_sim(present[i], present[j])
            if s is not None:
                sims_within.append(s)

    # Compare to random baseline
    import random
    random.seed(42)
    random_words = [w for w in vocab_list[:500] if w not in present]
    sims_random = []
    for _ in range(len(sims_within) * 5):
        w1, w2 = random.sample(random_words, 2)
        s = cosine_sim(w1, w2)
        if s is not None:
            sims_random.append(s)

    avg_within = sum(sims_within) / len(sims_within) if sims_within else 0
    avg_random = sum(sims_random) / len(sims_random) if sims_random else 0

    print(f"\n  {family_name}:")
    print(f"    Members in vocab: {present}")
    print(f"    Avg within-family sim: {avg_within:.3f}")
    print(f"    Avg random baseline:   {avg_random:.3f}")
    print(f"    Ratio: {avg_within / max(avg_random, 0.001):.2f}x")

    # Show pairwise
    for i in range(len(present)):
        for j in range(i+1, len(present)):
            s = cosine_sim(present[i], present[j])
            if s is not None and abs(s) > 0.1:
                print(f"      {present[i]:12s} ↔ {present[j]:12s}  sim={s:.3f}")

# ---------------------------------------------------------------------------
# TEST 3: Nearest neighbors for key astronomical words
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TEST 3: NEAREST NEIGHBORS — ASTRONOMICAL VOCABULARY")
print(f"{'='*80}")

astro_words = ['lkaiin', 'lkeey', 'lchedy', 'otol', 'otor', 'chocphy',
               'otalchy', 'otalalg', 'oteos', 'ofaralar', 'olkalaiin']

for w in astro_words:
    if w not in word2idx:
        continue
    nn = nearest_neighbors(w, n=8)
    glyph = eva_glyph(w)
    print(f"\n  {w:15s} {glyph}")
    for neighbor, sim in nn:
        n_glyph = eva_glyph(neighbor)
        sec_dist = []
        for sec in 'HSPCB':
            c = section_words.get(sec, {}).get(neighbor, 0)
            if c > 0:
                sec_dist.append(f"{TYPE_NAMES.get(sec, sec)[:3]}:{c}")
        sec_str = ', '.join(sec_dist) if sec_dist else ''
        print(f"    {neighbor:15s} {n_glyph:10s}  sim={sim:.3f}  [{sec_str}]")

# ---------------------------------------------------------------------------
# TEST 4: Analogies
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TEST 4: VECTOR ANALOGIES")
print(f"{'='*80}")

# Test if the vowel-length inflection is consistent in vector space
# daiin : dain :: okaiin : ?  (should give okain)
# chedy : cheedy :: shedy : ?  (should give sheedy)

analogy_tests = [
    ('daiin', 'dain', 'okaiin', 'okain?'),
    ('daiin', 'dain', 'aiin', 'ain?'),
    ('daiin', 'dain', 'saiin', 'sain?'),
    ('chedy', 'cheedy', 'shedy', 'sheedy?'),
    ('chey', 'cheey', 'shey', 'sheey?'),
    ('chol', 'chor', 'shol', 'shor?'),
    ('ol', 'or', 'al', 'ar?'),
    ('qokedy', 'qokeedy', 'okedy', 'okeedy?'),
    ('daiin', 'dar', 'okaiin', 'okar?'),
    ('daiin', 'dal', 'aiin', 'al?'),
]

print(f"\n  {'A':>12s} : {'B':>12s} :: {'C':>12s} : {'Expected':>12s}  → Top results")
print("-"*90)

for a, b, c, expected in analogy_tests:
    if any(w not in word2idx for w in [a, b, c]):
        continue
    results = analogy(a, b, c, n=5)
    result_str = ', '.join(f"{w}({s:.2f})" for w, s in results[:5])
    # Check if expected (minus ?) is in top results
    exp_clean = expected.rstrip('?')
    hit = '✓' if any(w == exp_clean for w, _ in results[:3]) else ' '
    print(f"  {a:>12s} : {b:>12s} :: {c:>12s} : {expected:>12s}  → {result_str} {hit}")

# ---------------------------------------------------------------------------
# TEST 5: Word clusters correlated with illustration type
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TEST 5: SEMANTIC CLUSTERS VIA K-MEANS")
print(f"{'='*80}")

# Cluster the top-500 words into groups and check section correlation
from scipy.cluster.hierarchy import linkage, fcluster

TOP_N = 500
top_words = vocab_list[:TOP_N]
top_vecs = word_vectors[:TOP_N]

# Hierarchical clustering
Z = linkage(top_vecs, method='ward', metric='euclidean')
n_clusters = 12
labels = fcluster(Z, n_clusters, criterion='maxclust')

print(f"\nClustered top {TOP_N} words into {n_clusters} groups:")

for cluster_id in range(1, n_clusters + 1):
    members = [top_words[i] for i in range(TOP_N) if labels[i] == cluster_id]
    if not members:
        continue

    # Section distribution for this cluster
    sec_counts = Counter()
    for w in members:
        for sec, words in section_words.items():
            sec_counts[sec] += words.get(w, 0)

    total = sum(sec_counts.values())
    sec_str = ', '.join(
        f"{TYPE_NAMES.get(s, s)[:3]}:{100*c/total:.0f}%"
        for s, c in sec_counts.most_common(4)
        if c > 0
    )

    # Show first few members with glyphs
    member_str = ' '.join(f"{eva_glyph(w)}" for w in members[:8])
    member_eva = ', '.join(members[:8])
    more = f" +{len(members)-8}" if len(members) > 8 else ""

    dominant = sec_counts.most_common(1)[0] if sec_counts else ('?', 0)
    dom_pct = 100 * dominant[1] / max(total, 1)
    flag = ' ←←' if dom_pct > 50 else ' ←' if dom_pct > 35 else ''

    print(f"\n  Cluster {cluster_id:2d} ({len(members):3d} words): [{sec_str}]{flag}")
    print(f"    {member_eva}{more}")
    print(f"    {member_str}")

# ---------------------------------------------------------------------------
# Export vectors
# ---------------------------------------------------------------------------

vector_export = {
    'vocab': vocab_list,
    'vectors': word_vectors.tolist(),
    'word2idx': word2idx,
    'dimensions': DIM,
}

with open('data/analysis/word_vectors.json', 'w') as f:
    json.dump(vector_export, f)
print(f"\n\nWord vectors exported to word_vectors.json ({V} words × {DIM} dims)")

# HTML report
def gen_html():
    header = ('<!DOCTYPE html><html><head><meta charset="UTF-8">'
              '<title>Voynich Distributional Word Vectors</title><style>'
              "@font-face { font-family: 'VoynichEVA'; src: url('fonts/Voynich/VoynichEVA.ttf') format('truetype'); }"
              "body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1400px; margin: 0 auto;"
              "padding: 20px; background: #0d1117; color: #c9d1d9; line-height: 1.6; }"
              "h1 { color: #58a6ff; border-bottom: 2px solid #1f6feb; }"
              "h2 { color: #79c0ff; margin-top: 40px; }"
              ".v { font-family: 'VoynichEVA', serif; font-size: 1.4em; color: #ffa657; letter-spacing: 2px; }"
              ".eva { font-family: monospace; background: #161b22; padding: 2px 6px; border-radius: 3px; color: #7ee787; }"
              "table { border-collapse: collapse; margin: 15px 0; background: #161b22; }"
              "th, td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }"
              "th { background: #21262d; color: #58a6ff; }"
              "tr:hover { background: #1c2128; }"
              ".note { background: #1c2128; border-left: 4px solid #1f6feb; padding: 12px 15px; margin: 15px 0; }"
              ".highlight { background: #2d1b00; border-left: 4px solid #d29922; padding: 12px 15px; margin: 15px 0; }"
              ".good { color: #7ee787; } .bad { color: #f78166; }"
              "</style></head><body>"
              "<h1>Voynich Distributional Word Vectors</h1>"
              "<div class='note'>PPMI co-occurrence (window 3) reduced to 50 dims via SVD. "
              f"Vocabulary: {V} words (freq >= 3).</div>"
              )

    parts = [header]
    parts.append('<h2>Vector Analogies</h2>')
    parts.append('<div class="highlight">If the inflectional system is real, '
                 'vector arithmetic should capture it: '
                 '<strong>long - short + other_long = other_short</strong></div>')
    parts.append('<table><tr><th>A</th><th>:</th><th>B</th><th>::</th><th>C</th>'
                '<th>:</th><th>Expected</th><th>Top Results</th></tr>')

    for a, b, c, expected in analogy_tests:
        if any(w not in word2idx for w in [a, b, c]):
            continue
        results = analogy(a, b, c, n=5)
        exp_clean = expected.rstrip('?')
        hit = any(w == exp_clean for w, _ in results[:3])
        res_parts = []
        for w, s in results[:5]:
            css = 'good' if w == exp_clean else ''
            g = eva_glyph(w)
            res_parts.append(f'<span class="{css}">{w} <span class="v">{g}</span> ({s:.2f})</span>')

        parts.append(
            f'<tr><td><span class="eva">{a}</span></td><td>:</td>'
            f'<td><span class="eva">{b}</span></td><td>::</td>'
            f'<td><span class="eva">{c}</span></td><td>:</td>'
            f'<td><span class="eva">{expected}</span></td>'
            f'<td>{", ".join(res_parts)}</td></tr>')

    parts.append('</table>')
    parts.append('</body></html>')
    return '\n'.join(parts)

html = gen_html()
Path('reports/html/vectors_report.html').write_text(html, encoding='utf-8')
print(f"HTML report: vectors_report.html")
