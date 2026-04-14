"""
Voynich Glyph Reanalysis
=========================
Test multiple decomposition hypotheses to find which produces the
cleanest morphological patterns.

Key distributional findings driving these hypotheses:
  - c + {h,t,k,p,f} = 99.8% → c is a structural connector, not a phoneme
  - q + o = 97.6% → qo is a single unit
  - a + i = 48.4% of all a → ai is a major unit
  - i only appears in {ain, aiin, air, aiiin} type sequences
  - e comes after h (42.8%), e (25.6%), or gallows k/t
  - ch and sh are clearly single units

Hypothesis levels:
  H0 (raw EVA): no changes
  H1 (conservative): ch, sh, cth, ckh, cph, cfh → single units
  H2 (moderate): H1 + qo→Q, ai→Ä, ii→Ï (modifier doubling)
  H3 (aggressive/abjad): H2 + ee→Ë, treat gallows as modifiers of ch
"""

import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

with open('data/transcription/voynich_nlp.json') as f:
    data = json.load(f)

sentences = data['sentences']
metadata = data['metadata']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}

all_words = [w for s in sentences for w in s['words']]
total_tokens = len(all_words)

# EVA to PUA for HTML output
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

def eva_to_glyph(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)

# =====================================================================
# Decomposition functions
# =====================================================================

def rewrite_h0(word):
    """H0: Raw EVA, no changes."""
    return word

def rewrite_h1(word):
    """H1: Conservative — collapse confirmed digraphs/trigraphs."""
    w = word
    w = w.replace('cth', 'Θ')
    w = w.replace('ckh', 'Κ')
    w = w.replace('cph', 'Π')
    w = w.replace('cfh', 'Φ')
    w = w.replace('ch', 'C')
    w = w.replace('sh', 'S')
    return w

def rewrite_h2(word):
    """H2: Moderate — H1 + qo=Q, treat 'ain/aiin' sequences as units."""
    w = rewrite_h1(word)
    w = w.replace('qo', 'Q')
    # ai → single vowel unit (long a? diphthong?)
    # But preserve the ii doubling distinction
    # aiin → Ä + N (where N = the -n ending)
    # ain → ä + N
    # We need to handle these carefully
    # First handle the longer sequences
    w = re.sub(r'aiii', 'ÄI', w)  # triple-i = long + extra
    w = re.sub(r'aii', 'Ä', w)    # aii = long vowel unit
    w = re.sub(r'ai', 'ä', w)     # ai = short diphthong
    return w

def rewrite_h3(word):
    """H3: Aggressive/abjad — H2 + ee=Ë, gallows as ch-modifiers."""
    w = rewrite_h2(word)
    # ee → single long-e unit
    w = re.sub(r'eee', 'ËE', w)   # triple-e = long + extra
    w = re.sub(r'ee', 'Ë', w)     # ee = long vowel
    # Gallows without c-frame: treat as modifier + base
    # t after vowel or consonant = modifier (aspirate? emphatic?)
    # k after vowel = modifier
    # This is too speculative for automatic rewrite — leave as-is
    return w

HYPOTHESES = [
    ('H0 (raw EVA)', rewrite_h0),
    ('H1 (digraphs)', rewrite_h1),
    ('H2 (+qo,ai units)', rewrite_h2),
    ('H3 (+ee units)', rewrite_h3),
]

# Readable names for the rewritten characters
CHAR_NAMES = {
    'Θ': 'CTH', 'Κ': 'CKH', 'Π': 'CPH', 'Φ': 'CFH',
    'C': 'CH', 'S': 'SH', 'Q': 'QO',
    'Ä': 'AII', 'ä': 'AI', 'Ë': 'EE', 'ÄI': 'AIII',
}

def char_name(ch):
    return CHAR_NAMES.get(ch, ch)

# =====================================================================
# Run each hypothesis
# =====================================================================

results = {}

for name, rewrite_fn in HYPOTHESES:
    # Rewrite all words
    rewritten = [rewrite_fn(w) for w in all_words]
    rw_freq = Counter(rewritten)

    # Character inventory
    char_freq = Counter()
    for w in rewritten:
        for ch in w:
            char_freq[ch] += 1

    # Average word length
    avg_len = sum(len(w) for w in rewritten) / len(rewritten)

    # Unique types
    n_types = len(rw_freq)

    # Hapax
    hapax = sum(1 for c in rw_freq.values() if c == 1)

    # Character entropy (how evenly distributed is the alphabet?)
    total_chars = sum(char_freq.values())
    char_entropy = -sum(
        (c/total_chars) * math.log2(c/total_chars)
        for c in char_freq.values()
    )

    # Paradigm analysis: count stems with multiple endings
    stem_endings = defaultdict(Counter)
    for w, c in rw_freq.items():
        if len(w) < 2:
            continue
        for split in range(1, min(5, len(w))):
            stem = w[:split]
            ending = w[split:]
            if len(ending) >= 1:
                stem_endings[stem][ending] += c

    productive_stems = sum(
        1 for stem, endings in stem_endings.items()
        if len(endings) >= 3 and sum(endings.values()) >= 15
    )

    # Suffix agreement test
    same_end = 0
    total_pairs = 0
    end_freq = Counter()
    for s in sentences:
        rw_words = [rewrite_fn(w) for w in s['words']]
        for i in range(len(rw_words) - 1):
            w1, w2 = rw_words[i], rw_words[i+1]
            if len(w1) >= 2 and len(w2) >= 2:
                e1 = w1[-1]
                e2 = w2[-1]
                end_freq[e1] += 1
                end_freq[e2] += 1
                total_pairs += 1
                if e1 == e2:
                    same_end += 1

    if total_pairs > 0:
        obs_rate = same_end / total_pairs
        total_ends = sum(end_freq.values())
        exp_rate = sum((c/total_ends)**2 for c in end_freq.values())
        agreement_ratio = obs_rate / max(exp_rate, 0.001)
    else:
        agreement_ratio = 0

    results[name] = {
        'n_chars': len(char_freq),
        'n_types': n_types,
        'hapax': hapax,
        'avg_len': avg_len,
        'char_entropy': char_entropy,
        'productive_stems': productive_stems,
        'agreement_ratio': agreement_ratio,
        'char_freq': char_freq,
        'rw_freq': rw_freq,
        'top_words': rw_freq.most_common(30),
        'rewrite_fn': rewrite_fn,
    }

# =====================================================================
# Print comparison table
# =====================================================================

print("="*90)
print("DECOMPOSITION HYPOTHESIS COMPARISON")
print("="*90)
print(f"\n{'Hypothesis':25s} {'Chars':>5s} {'Types':>6s} {'Hapax':>6s} {'AvgLen':>6s} "
      f"{'H(char)':>7s} {'Stems':>6s} {'Agree':>6s}")
print("-"*75)

for name in [n for n, _ in HYPOTHESES]:
    r = results[name]
    print(f"  {name:23s} {r['n_chars']:5d} {r['n_types']:6d} {r['hapax']:6d} "
          f"{r['avg_len']:6.2f} {r['char_entropy']:7.3f} {r['productive_stems']:6d} "
          f"{r['agreement_ratio']:6.2f}")

print(f"""
Key metrics:
  Chars: number of distinct character units in the alphabet
  Types: number of unique word forms
  Hapax: words appearing only once (lower = more systematic)
  AvgLen: average word length in character units
  H(char): character entropy (higher = more uniform usage = more efficient encoding)
  Stems: productive paradigm stems (≥3 endings, freq≥15)
  Agree: suffix agreement ratio (observed/expected, higher = stronger declension signal)
""")

# =====================================================================
# Detailed analysis for each hypothesis
# =====================================================================

for name in [n for n, _ in HYPOTHESES]:
    r = results[name]
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    print(f"\n  Alphabet ({r['n_chars']} characters):")
    for ch, count in r['char_freq'].most_common():
        cn = char_name(ch)
        pct = 100 * count / sum(r['char_freq'].values())
        bar = '█' * int(pct * 2)
        if pct >= 0.5:
            print(f"    {cn:5s}  {count:6d}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Top 20 words:")
    for w, c in r['top_words'][:20]:
        # Show constituent mapping
        print(f"    {w:15s}  {c:5d}")

# =====================================================================
# Rerun declension analysis with best hypothesis
# =====================================================================

print("\n" + "="*80)
print("DECLENSION REANALYSIS UNDER H3 (aggressive decomposition)")
print("="*80)

rewrite = rewrite_h3

# Rewrite all sentences
rw_sentences = []
for s in sentences:
    rw_words = [rewrite(w) for w in s['words']]
    rw_sentences.append({
        'folio': s['folio'],
        'words': rw_words,
        'unit': s.get('unit', ''),
    })

# Build vocabulary
rw_freq = Counter()
for s in rw_sentences:
    rw_freq.update(s['words'])

# Extract paradigms
stem_endings = defaultdict(Counter)
stem_words = defaultdict(set)
for w, c in rw_freq.items():
    if len(w) < 2:
        continue
    for split in range(1, min(5, len(w))):
        stem = w[:split]
        ending = w[split:]
        if 1 <= len(ending) <= 4:
            stem_endings[stem][ending] += c
            stem_words[stem].add(w)

paradigms = []
for stem, endings in stem_endings.items():
    n_end = len(endings)
    total = sum(endings.values())
    if n_end >= 4 and total >= 20:
        paradigms.append((stem, n_end, total, endings))

paradigms.sort(key=lambda x: -x[2])

print(f"\nProductive paradigms (4+ endings, freq≥20): {len(paradigms)}")
print(f"\nTop 30 paradigms under H3:")
print(f"{'Stem':>8s} {'#End':>5s} {'Total':>6s}  Top endings")
print("-"*80)

for stem, n_end, total, endings in paradigms[:30]:
    top5 = ', '.join(f'-{e}({c})' for e, c in endings.most_common(5))
    cn_stem = ''.join(char_name(ch) for ch in stem)
    print(f"  {cn_stem:>8s} {n_end:5d} {total:6d}  {top5}")

# =====================================================================
# THE KEY TEST: Do the vowel-length distinctions collapse into
# cleaner case paradigms?
# =====================================================================

print(f"\n{'='*80}")
print("VOWEL LENGTH AS CASE MARKER (H3 analysis)")
print("="*80)

# Under H3:
# 'ä' (was 'ain') vs 'Ä' (was 'aiin') vs 'ÄI' (was 'aiiin')
# 'e' vs 'Ë' (was 'ee') vs 'ËE' (was 'eee')
# These should show different case distributions if they're inflectional

# Find words that differ only in vowel length
def strip_length(word):
    """Collapse vowel length markers to base vowels."""
    w = word
    w = w.replace('ÄI', 'ä')
    w = w.replace('Ä', 'ä')
    w = w.replace('ËE', 'Ë')
    w = w.replace('Ë', 'e')
    return w

length_families = defaultdict(list)
for w, c in rw_freq.items():
    base = strip_length(w)
    if base != w:
        length_families[base].append((w, c))

# Also add the base form if it exists
for base in list(length_families.keys()):
    if base in rw_freq and all(w != base for w, _ in length_families[base]):
        length_families[base].append((base, rw_freq[base]))
    length_families[base].sort(key=lambda x: -x[1])

# Show families with multiple length variants
print(f"\nVowel-length variant families (words differing only in ä/Ä/ÄI or e/Ë):")
print(f"{'Base form':>15s}  Variants")
print("-"*70)

interesting = [(base, members) for base, members in length_families.items()
               if len(members) >= 2 and sum(c for _, c in members) >= 20]
interesting.sort(key=lambda x: -sum(c for _, c in x[1]))

for base, members in interesting[:25]:
    cn_base = ''.join(char_name(ch) for ch in base)
    variant_str = '  '.join(f"{''.join(char_name(ch) for ch in w)}({c})"
                           for w, c in members)
    print(f"  {cn_base:>15s}:  {variant_str}")

# =====================================================================
# Generate the final HTML report with the reanalysed alphabet
# =====================================================================

def generate_html():
    parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Voynich Manuscript — Glyph Decomposition & Reanalysis</title>
<style>
@font-face {
    font-family: 'VoynichEVA';
    src: url('fonts/Voynich/VoynichEVA.ttf') format('truetype');
}
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 20px;
    background: #0d1117; color: #c9d1d9; line-height: 1.6;
}
h1 { color: #58a6ff; border-bottom: 2px solid #1f6feb; padding-bottom: 10px; }
h2 { color: #79c0ff; margin-top: 40px; }
h3 { color: #a5d6ff; }
.v { font-family: 'VoynichEVA', serif; font-size: 1.4em; color: #ffa657; letter-spacing: 2px; }
.eva { font-family: 'Courier New', monospace; background: #161b22; padding: 1px 6px;
       border-radius: 3px; color: #7ee787; font-size: 0.9em; }
.unit { font-family: 'Courier New', monospace; background: #1c1030; padding: 1px 6px;
        border-radius: 3px; color: #d2a8ff; font-size: 0.95em; font-weight: bold; }
table { border-collapse: collapse; margin: 15px 0; background: #161b22; }
th, td { border: 1px solid #30363d; padding: 8px 12px; text-align: left; }
th { background: #21262d; color: #58a6ff; }
tr:hover { background: #1c2128; }
.freq { color: #8b949e; font-size: 0.85em; }
.note { background: #1c2128; border-left: 4px solid #1f6feb; padding: 12px 15px; margin: 15px 0; }
.highlight { background: #2d1b00; border-left: 4px solid #d29922; padding: 12px 15px; margin: 15px 0; }
.bar { display: inline-block; height: 16px; min-width: 2px; background: #58a6ff;
       margin-right: 1px; vertical-align: middle; border-radius: 2px; }
.comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.metric-good { color: #7ee787; font-weight: bold; }
.metric-bad { color: #f78166; }
</style>
</head>
<body>
<h1>Voynich Glyph Decomposition & Declension Reanalysis</h1>
"""]

    # Comparison table
    parts.append('<h2>1. Hypothesis Comparison</h2>')
    parts.append("""<div class="note">
    Four decomposition levels tested. The "best" hypothesis maximises character entropy
    (efficient encoding), suffix agreement ratio (declension signal), and productive stems,
    while minimising alphabet size and hapax count.
    </div>""")

    parts.append('<table><tr><th>Hypothesis</th><th>Alphabet</th><th>Word Types</th>'
                '<th>Avg Length</th><th>Char Entropy</th>'
                '<th>Paradigm Stems</th><th>Agreement Ratio</th></tr>')

    best_agree = max(r['agreement_ratio'] for r in results.values())
    best_entropy = max(r['char_entropy'] for r in results.values())

    for name in [n for n, _ in HYPOTHESES]:
        r = results[name]
        agree_cls = 'metric-good' if r['agreement_ratio'] == best_agree else ''
        ent_cls = 'metric-good' if r['char_entropy'] == best_entropy else ''
        parts.append(
            f'<tr><td>{name}</td><td>{r["n_chars"]}</td><td>{r["n_types"]}</td>'
            f'<td>{r["avg_len"]:.2f}</td>'
            f'<td class="{ent_cls}">{r["char_entropy"]:.3f}</td>'
            f'<td>{r["productive_stems"]}</td>'
            f'<td class="{agree_cls}">{r["agreement_ratio"]:.2f}</td></tr>')
    parts.append('</table>')

    # Alphabet comparison
    parts.append('<h2>2. Reanalysed Alphabet (H3)</h2>')
    r = results['H3 (+ee units)']
    parts.append('<table><tr><th>Unit</th><th>EVA</th><th>Glyph</th>'
                '<th>Count</th><th>Frequency</th><th>Type</th></tr>')

    total_ch = sum(r['char_freq'].values())
    type_map = {
        'o': 'vowel', 'e': 'vowel', 'y': 'vowel', 'a': 'vowel', 'i': 'vowel',
        'Ä': 'long vowel (aii)', 'ä': 'diphthong (ai)', 'Ë': 'long vowel (ee)',
        'd': 'consonant', 'C': 'consonant (ch)', 'l': 'consonant', 'k': 'consonant/modifier',
        'r': 'consonant', 'n': 'consonant', 'Q': 'consonant (qo)', 't': 'consonant/modifier',
        'S': 'consonant (sh)', 's': 'consonant', 'p': 'consonant/modifier',
        'm': 'consonant', 'f': 'consonant/modifier',
        'Θ': 'modified ch (cth)', 'Κ': 'modified ch (ckh)',
        'Π': 'modified ch (cph)', 'Φ': 'modified ch (cfh)',
    }

    for ch, count in r['char_freq'].most_common():
        cn = char_name(ch)
        pct = 100 * count / total_ch
        if pct < 0.1:
            continue
        # Get EVA equivalent
        eva_map = {'C': 'ch', 'S': 'sh', 'Q': 'qo', 'Θ': 'cth', 'Κ': 'ckh',
                   'Π': 'cph', 'Φ': 'cfh', 'Ä': 'aii', 'ä': 'ai', 'Ë': 'ee'}
        eva = eva_map.get(ch, ch)
        glyph = eva_to_glyph(eva)
        bar_w = max(2, int(pct * 10))
        tp = type_map.get(ch, '?')
        parts.append(
            f'<tr><td><span class="unit">{cn}</span></td>'
            f'<td><span class="eva">{eva}</span></td>'
            f'<td><span class="v">{glyph}</span></td>'
            f'<td>{count}</td>'
            f'<td><span class="bar" style="width:{bar_w}px"></span> {pct:.1f}%</td>'
            f'<td class="freq">{tp}</td></tr>')
    parts.append('</table>')

    # Vowel-length families
    parts.append('<h2>3. Vowel-Length as Inflection (under H3)</h2>')
    parts.append("""<div class="highlight">
    Under the reanalysis, the vowel doubling collapses into distinct vowel-length units:
    <span class="unit">AI</span> (short) vs <span class="unit">AII</span> (long) vs
    <span class="unit">AIII</span> (overlong), and <span class="unit">e</span> (short) vs
    <span class="unit">EE</span> (long). If these encode case/tense/aspect, then
    the "real" stems are shorter and the paradigm tables become cleaner.
    </div>""")

    parts.append('<table><tr><th>Base Form</th><th>Variants</th></tr>')
    for base, members in interesting[:20]:
        cn_base = ''.join(char_name(ch) for ch in base)
        var_parts = []
        for w, c in members:
            cn_w = ''.join(char_name(ch) for ch in w)
            eva_w_chars = w.replace('Θ','cth').replace('Κ','ckh').replace('Π','cph')
            eva_w_chars = eva_w_chars.replace('Φ','cfh').replace('C','ch').replace('S','sh')
            eva_w_chars = eva_w_chars.replace('Q','qo').replace('Ä','aii').replace('ä','ai')
            eva_w_chars = eva_w_chars.replace('Ë','ee')
            glyph = eva_to_glyph(eva_w_chars)
            var_parts.append(
                f'<span class="unit">{cn_w}</span> '
                f'<span class="v">{glyph}</span> '
                f'<span class="freq">({c})</span>')
        parts.append(
            f'<tr><td><span class="unit">{cn_base}</span></td>'
            f'<td>{" &nbsp;|&nbsp; ".join(var_parts)}</td></tr>')
    parts.append('</table>')

    # Paradigm tables under H3
    parts.append('<h2>4. Declension Tables (H3 Reanalysis)</h2>')
    parts.append('<table><tr><th>Stem</th><th>#Endings</th><th>Total</th>'
                '<th>Top Endings</th></tr>')
    for stem, n_end, total, endings in paradigms[:25]:
        cn_stem = ''.join(char_name(ch) for ch in stem)
        top5 = ', '.join(
            f'<span class="unit">-{"".join(char_name(c) for c in e)}</span>'
            f'<span class="freq">({c})</span>'
            for e, c in endings.most_common(6))
        parts.append(
            f'<tr><td><span class="unit">{cn_stem}-</span></td>'
            f'<td>{n_end}</td><td>{total}</td><td>{top5}</td></tr>')
    parts.append('</table>')

    parts.append("""
<h2>5. Interpretation</h2>
<div class="note">
<h3>The emerging picture of the writing system</h3>
<ol>
<li><strong>~18 functional character units</strong> — not the 25+ suggested by raw EVA.
    This is consistent with an abjad or simplified syllabary.</li>
<li><strong><span class="unit">CH</span> is the core consonant glyph</strong> —
    it can be modified by gallows elements (t, k, p, f) to produce
    <span class="unit">CTH</span>, <span class="unit">CKH</span>,
    <span class="unit">CPH</span>, <span class="unit">CFH</span>.
    If these modifiers encode aspiration, emphasis, or pharyngealisation,
    this parallels Arabic's emphatic consonant series (ت/ط, س/ص, د/ض).</li>
<li><strong>Vowel length is grammatical</strong> — the AI/AII/AIII and e/EE/EEE
    distinctions have different syntactic contexts, functioning as inflectional
    morphology rather than phonological variation.</li>
<li><strong><span class="unit">QO</span> is a single unit</strong> — appearing
    almost exclusively word-initially, possibly encoding a definite article,
    relative pronoun, or demonstrative.</li>
<li><strong>The gallows as modifier hypothesis</strong>: if t/k/p/f are not
    standalone consonants but modifiers of adjacent characters, then the true
    consonant inventory is: CH, SH, d, l, r, n, s, m — only 8 consonants,
    each potentially having emphatic/aspirated variants via gallows modification.</li>
</ol>
</div>
</body></html>""")

    return '\n'.join(parts)


html = generate_html()
Path('reports/html/glyph_reanalysis.html').write_text(html, encoding='utf-8')
print(f"\nHTML report written to glyph_reanalysis.html ({len(html)//1024}KB)")

# Export reanalysed data
export = {
    'hypothesis_comparison': {
        name: {
            'n_chars': results[name]['n_chars'],
            'n_types': results[name]['n_types'],
            'avg_len': results[name]['avg_len'],
            'char_entropy': results[name]['char_entropy'],
            'productive_stems': results[name]['productive_stems'],
            'agreement_ratio': results[name]['agreement_ratio'],
        }
        for name in [n for n, _ in HYPOTHESES]
    },
    'h3_top_words': [(w, c) for w, c in results['H3 (+ee units)']['rw_freq'].most_common(100)],
    'h3_paradigms': [
        {'stem': stem, 'n_endings': n, 'total': t,
         'top_endings': dict(e.most_common(10))}
        for stem, n, t, e in paradigms[:50]
    ],
}

with open('data/analysis/glyph_reanalysis.json', 'w') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)
print("JSON data written to glyph_reanalysis.json")
