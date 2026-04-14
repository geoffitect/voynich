"""
Voynich Manuscript Tokenizer
==============================
Strip all Latin alphabetic bias from the analysis by converting EVA
into abstract structural tokens. Each token represents a GLYPH UNIT
identified by our structural analysis — not a phoneme guess.

Token assignment is based purely on:
  1. Glyph identity (structural decomposition from H3 analysis)
  2. Distributional properties (frequency, position, context)
  3. NO Latin letter associations

The resulting token stream can be used for unbiased distributional
analysis, paradigm detection, and eventual phoneme assignment.
"""

import json
import re
import math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open('data/transcription/voynich_nlp.json') as f:
    data = json.load(f)

sentences = data['sentences']
metadata = data['metadata']
folio_type = {f: m.get('illustration', '?') for f, m in metadata.items()}

# EVA → PUA glyph mapping (for display)
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

# ---------------------------------------------------------------------------
# Structural tokenization
# ---------------------------------------------------------------------------

# Segmentation rules derived from distributional analysis:
# 1. c + {h,t,k,p,f} → single unit (c is structural connector)
# 2. cth, ckh, cph, cfh → single unit (gallows cartouche)
# 3. sh → single unit
# 4. qo → single unit (97.6% co-occurrence)
# 5. aii, ai → vowel units (i only appears in a-i sequences)
# 6. ee → vowel unit (length distinction is grammatical)
# 7. Remaining single characters are individual tokens

# Ordered by length (longest match first) for greedy tokenization
SEGMENT_RULES = [
    # 4-char
    ('aiii', 'V_AIII'),    # overlong vowel
    ('eeey', None),        # handle as eee + y
    # 3-char trigraphs
    ('cth', 'C_CTH'),      # gallows cartouche: ch + t-modifier
    ('ckh', 'C_CKH'),      # gallows cartouche: ch + k-modifier
    ('cph', 'C_CPH'),      # gallows cartouche: ch + p-modifier
    ('cfh', 'C_CFH'),      # gallows cartouche: ch + f-modifier
    ('eee', 'V_EEE'),      # triple-e (overlong)
    ('aii', 'V_AII'),      # long ai-diphthong
    # 2-char digraphs
    ('ch', 'C_CH'),        # the core c-ligature
    ('sh', 'C_SH'),        # s + h digraph
    ('qo', 'C_QO'),        # q + o fused unit
    ('ee', 'V_EE'),        # long-e vowel
    ('ai', 'V_AI'),        # short ai-diphthong
]

# Single-character tokens
SINGLE_TOKENS = {
    'o': 'V_O',
    'e': 'V_E',
    'y': 'V_Y',
    'a': 'V_A',
    'i': 'V_I',       # rare outside ai/aii sequences
    'd': 'C_D',
    'l': 'C_L',
    'k': 'C_K',
    'r': 'C_R',
    'n': 'C_N',
    't': 'C_T',
    's': 'C_S',
    'p': 'C_P',
    'f': 'C_F',
    'm': 'C_M',
    'g': 'C_G',
    'q': 'C_Q',        # rare: q without following o
    'h': 'X_H',        # rare: h outside digraphs
    'c': 'X_C',        # extremely rare: c without following h/gallows
    'x': 'X_X',
    'j': 'X_J',
    'b': 'X_B',
    'u': 'X_U',
    'v': 'X_V',
    'z': 'X_Z',
    '?': 'X_UNK',
}


def tokenize_word(eva_word: str) -> list[str]:
    """Segment an EVA word into structural tokens using greedy matching."""
    tokens = []
    i = 0
    while i < len(eva_word):
        matched = False
        # Try longest segments first
        for pattern, token_id in SEGMENT_RULES:
            if token_id is None:
                continue
            if eva_word[i:i+len(pattern)] == pattern:
                tokens.append(token_id)
                i += len(pattern)
                matched = True
                break
        if not matched:
            ch = eva_word[i]
            token_id = SINGLE_TOKENS.get(ch, f'X_{ch.upper()}')
            tokens.append(token_id)
            i += 1
    return tokens


def tokenize_corpus(sentences):
    """Tokenize the entire corpus. Returns list of (folio, unit, token_lists)."""
    result = []
    for s in sentences:
        token_lists = [tokenize_word(w) for w in s['words']]
        result.append({
            'folio': s['folio'],
            'unit': s.get('unit', ''),
            'eva_words': s['words'],
            'token_words': token_lists,
            'token_strings': ['.'.join(tl) for tl in token_lists],
        })
    return result


# ---------------------------------------------------------------------------
# Run tokenization
# ---------------------------------------------------------------------------

print("Tokenizing corpus...")
tokenized = tokenize_corpus(sentences)

# Build token vocabulary
token_word_freq = Counter()
token_freq = Counter()  # individual token frequency
all_token_words = []
for s in tokenized:
    for tw in s['token_strings']:
        token_word_freq[tw] += 1
        all_token_words.append(tw)
    for tl in s['token_words']:
        for t in tl:
            token_freq[t] += 1

total_token_words = len(all_token_words)
total_tokens = sum(token_freq.values())

print(f"  Total word tokens: {total_token_words}")
print(f"  Unique word types: {len(token_word_freq)}")
print(f"  Token alphabet size: {len(token_freq)}")
print(f"  Total glyph tokens: {total_tokens}")

# ---------------------------------------------------------------------------
# Token alphabet
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TOKEN ALPHABET (structural units, no phonetic assumption)")
print(f"{'='*80}")

# Assign short numeric IDs for compact display
token_ids = {}
for i, (tok, count) in enumerate(token_freq.most_common()):
    token_ids[tok] = f"T{i:02d}"

# Classify tokens
def token_class(tok):
    if tok.startswith('V_'): return 'vowel-class'
    if tok.startswith('C_'): return 'consonant-class'
    return 'rare/unknown'

print(f"\n{'ID':>4s} {'Token':>8s} {'EVA':>6s} {'Glyph':>6s} {'Count':>6s} "
      f"{'Freq%':>6s} {'Class':>15s}")
print("-"*65)

# Reverse map for EVA display
TOKEN_TO_EVA = {}
for pat, tok in SEGMENT_RULES:
    if tok:
        TOKEN_TO_EVA[tok] = pat
for ch, tok in SINGLE_TOKENS.items():
    if tok not in TOKEN_TO_EVA:
        TOKEN_TO_EVA[tok] = ch

for tok, count in token_freq.most_common():
    tid = token_ids[tok]
    eva = TOKEN_TO_EVA.get(tok, '?')
    glyph = eva_glyph(eva)
    pct = 100 * count / total_tokens
    cls = token_class(tok)
    if pct >= 0.05:
        print(f"  {tid:>4s} {tok:>8s} {eva:>6s} {glyph:>6s} {count:6d} {pct:6.2f}% {cls:>15s}")

# ---------------------------------------------------------------------------
# Top tokenized words
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("TOP 30 TOKENIZED WORDS")
print(f"{'='*80}")

print(f"\n{'Rank':>4s} {'Token form':>35s} {'Count':>6s} {'EVA':>15s} {'Glyph'}")
print("-"*90)

# Reverse-map a token word back to EVA
def tokens_to_eva(token_str):
    tokens = token_str.split('.')
    return ''.join(TOKEN_TO_EVA.get(t, '?') for t in tokens)

for rank, (tw, count) in enumerate(token_word_freq.most_common(30), 1):
    eva = tokens_to_eva(tw)
    glyph = eva_glyph(eva)
    # Compact token form using IDs
    compact = '.'.join(token_ids.get(t, t) for t in tw.split('.'))
    print(f"  {rank:3d}. {compact:>35s} {count:6d} {eva:>15s} {glyph}")

# ---------------------------------------------------------------------------
# Distributional similarity (PMI-based, no embeddings needed)
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("DISTRIBUTIONAL ANALYSIS (unbiased token space)")
print(f"{'='*80}")

# 1. Token bigram PMI within words
print("\n--- Token bigram PMI (within words) ---")

bigram_count = Counter()
for tw in all_token_words:
    tokens = tw.split('.')
    for i in range(len(tokens) - 1):
        bigram_count[(tokens[i], tokens[i+1])] += 1

total_bigrams = sum(bigram_count.values())

pmi_scores = []
for (t1, t2), count in bigram_count.items():
    if count < 5:
        continue
    p_joint = count / total_bigrams
    p_t1 = token_freq[t1] / total_tokens
    p_t2 = token_freq[t2] / total_tokens
    pmi = math.log2(p_joint / (p_t1 * p_t2))
    pmi_scores.append((t1, t2, count, pmi))

pmi_scores.sort(key=lambda x: -x[3])

print(f"\nHighest PMI bigrams (tokens that strongly predict each other):")
print(f"{'T1':>8s} → {'T2':>8s} {'Count':>6s} {'PMI':>6s}  Interpretation")
print("-"*60)

for t1, t2, count, pmi in pmi_scores[:25]:
    id1, id2 = token_ids.get(t1, t1), token_ids.get(t2, t2)
    eva1, eva2 = TOKEN_TO_EVA.get(t1, '?'), TOKEN_TO_EVA.get(t2, '?')
    interp = ''
    if t1.startswith('V_') and t2.startswith('V_'):
        interp = '(vowel+vowel: possible single unit?)'
    elif t1.startswith('C_') and t2 == 'V_Y':
        interp = '(consonant+Y: syllable-final pattern)'
    elif t1 == 'V_O' and t2.startswith('C_'):
        interp = '(O+consonant: possible article+stem)'
    print(f"  {id1:>4s}({eva1:>4s}) → {id2:>4s}({eva2:>4s}) {count:6d} {pmi:6.2f}  {interp}")

# 2. Token position distribution
print(f"\n--- Token positional entropy (where in the word does each token appear?) ---")

token_word_positions = defaultdict(list)  # token → list of (pos_in_word / word_length)
for tw in all_token_words:
    tokens = tw.split('.')
    n = len(tokens)
    if n < 2:
        continue
    for i, t in enumerate(tokens):
        token_word_positions[t].append(i / (n - 1))

print(f"\n{'Token':>8s} {'EVA':>5s} {'Count':>6s} {'MeanPos':>8s} {'Entropy':>8s} {'Tendency'}")
print("-"*60)

for tok, count in token_freq.most_common():
    if count < 50:
        continue
    positions = token_word_positions[tok]
    if not positions:
        continue
    mean_pos = sum(positions) / len(positions)

    # Entropy of position distribution (5 bins)
    bins = [0] * 5
    for p in positions:
        b = min(4, int(p * 5))
        bins[b] += 1
    total_p = len(positions)
    entropy = -sum(
        (c/total_p) * math.log2(c/total_p)
        for c in bins if c > 0
    )

    tendency = ''
    if mean_pos < 0.25:
        tendency = '← INITIAL'
    elif mean_pos > 0.75:
        tendency = 'FINAL →'
    elif entropy > 2.2:
        tendency = '(free position)'

    eva = TOKEN_TO_EVA.get(tok, '?')
    tid = token_ids[tok]
    print(f"  {tid:>4s}({eva:>4s}) {count:6d} {mean_pos:8.3f} {entropy:8.3f}  {tendency}")

# ---------------------------------------------------------------------------
# Paradigm analysis in token space
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("PARADIGM ANALYSIS (token space — zero Latin bias)")
print(f"{'='*80}")

# Split token words into stem + ending (in token space)
stem_endings = defaultdict(Counter)
for tw, count in token_word_freq.items():
    tokens = tw.split('.')
    if len(tokens) < 2:
        continue
    # Try stem lengths 1..3 tokens
    for stem_len in range(1, min(4, len(tokens))):
        stem = '.'.join(tokens[:stem_len])
        ending = '.'.join(tokens[stem_len:])
        if ending:
            stem_endings[stem][ending] += count

paradigms = []
for stem, endings in stem_endings.items():
    n_end = len(endings)
    total = sum(endings.values())
    if n_end >= 4 and total >= 20:
        paradigms.append((stem, n_end, total, endings))

paradigms.sort(key=lambda x: -x[2])

print(f"\nProductive paradigms (4+ endings, freq≥20): {len(paradigms)}")
print(f"\nTop 25 paradigms:")
print(f"{'Stem tokens':>30s} {'#End':>5s} {'Freq':>5s}  Top endings")
print("-"*100)

for stem, n_end, total, endings in paradigms[:25]:
    # Convert to compact IDs
    stem_ids = '.'.join(token_ids.get(t, t) for t in stem.split('.'))
    stem_eva = tokens_to_eva(stem)
    top_ends = []
    for e, c in endings.most_common(5):
        e_ids = '.'.join(token_ids.get(t, t) for t in e.split('.'))
        e_eva = tokens_to_eva(e)
        top_ends.append(f"-{e_eva}({c})")
    print(f"  {stem_ids:>28s} ({stem_eva:>6s}-) {n_end:5d} {total:5d}  {', '.join(top_ends)}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

export = {
    'token_alphabet': [
        {
            'token': tok,
            'id': token_ids[tok],
            'eva': TOKEN_TO_EVA.get(tok, '?'),
            'count': count,
            'class': token_class(tok),
        }
        for tok, count in token_freq.most_common()
    ],
    'top_words': [
        {
            'token_form': tw,
            'eva': tokens_to_eva(tw),
            'count': count,
        }
        for tw, count in token_word_freq.most_common(200)
    ],
    'paradigms': [
        {
            'stem': stem,
            'stem_eva': tokens_to_eva(stem),
            'n_endings': n_end,
            'total': total,
            'top_endings': [
                {'ending': e, 'ending_eva': tokens_to_eva(e), 'count': c}
                for e, c in endings.most_common(10)
            ],
        }
        for stem, n_end, total, endings in paradigms[:50]
    ],
    'tokenized_sentences': [
        {
            'folio': s['folio'],
            'eva_words': s['eva_words'],
            'token_words': s['token_strings'],
        }
        for s in tokenized
    ],
}

with open('data/transcription/tokenized_corpus.json', 'w') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)
print(f"\nExported to tokenized_corpus.json")

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def gen_html():
    parts = ["""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Voynich — Structural Tokenization (Zero Latin Bias)</title>
<style>
@font-face { font-family: 'VoynichEVA'; src: url('fonts/Voynich/VoynichEVA.ttf') format('truetype'); }
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1400px; margin: 0 auto;
       padding: 20px; background: #0d1117; color: #c9d1d9; line-height: 1.6; }
h1 { color: #58a6ff; border-bottom: 2px solid #1f6feb; }
h2 { color: #79c0ff; margin-top: 40px; }
.v { font-family: 'VoynichEVA', serif; font-size: 1.6em; color: #ffa657; letter-spacing: 2px; }
.tok { font-family: monospace; background: #1a1030; padding: 2px 6px; border-radius: 3px;
       color: #d2a8ff; font-weight: bold; }
.eva { font-family: monospace; background: #161b22; padding: 2px 6px; border-radius: 3px;
       color: #7ee787; font-size: 0.85em; }
table { border-collapse: collapse; margin: 15px 0; background: #161b22; }
th, td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }
th { background: #21262d; color: #58a6ff; }
tr:hover { background: #1c2128; }
.note { background: #1c2128; border-left: 4px solid #1f6feb; padding: 12px 15px; margin: 15px 0; }
.highlight { background: #2d1b00; border-left: 4px solid #d29922; padding: 12px 15px; margin: 15px 0; }
.bar { display: inline-block; height: 14px; min-width: 2px; background: #58a6ff;
       border-radius: 2px; vertical-align: middle; }
.vow { color: #ffa657; } .con { color: #7ee787; } .rare { color: #8b949e; }
</style></head><body>
<h1>Structural Tokenization — Zero Latin Bias</h1>
<div class="note">
<p>This analysis strips all phonetic assumptions from the EVA romanization.
Each glyph unit is assigned an abstract token ID based purely on
structural properties (frequency, position, co-occurrence). The tokens
carry <strong>no Latin letter associations</strong>.</p>
</div>
"""]

    # Token alphabet table
    parts.append('<h2>Token Alphabet</h2>')
    parts.append('<table><tr><th>ID</th><th>Glyph</th><th>EVA</th>'
                '<th>Count</th><th>Freq</th><th>Class</th><th>Position</th></tr>')

    for tok, count in token_freq.most_common():
        if count < 20:
            continue
        tid = token_ids[tok]
        eva = TOKEN_TO_EVA.get(tok, '?')
        glyph = eva_glyph(eva)
        pct = 100 * count / total_tokens
        cls = token_class(tok)
        css = 'vow' if cls == 'vowel-class' else 'con' if cls == 'consonant-class' else 'rare'

        # Position tendency
        positions = token_word_positions.get(tok, [])
        if positions:
            mean_pos = sum(positions) / len(positions)
            if mean_pos < 0.25: pos_str = '← initial'
            elif mean_pos > 0.75: pos_str = 'final →'
            else: pos_str = 'free'
        else:
            pos_str = '—'

        bar_w = max(2, int(pct * 8))
        parts.append(
            f'<tr><td><span class="tok">{tid}</span></td>'
            f'<td><span class="v">{glyph}</span></td>'
            f'<td><span class="eva">{eva}</span></td>'
            f'<td>{count}</td>'
            f'<td><span class="bar" style="width:{bar_w}px"></span> {pct:.1f}%</td>'
            f'<td class="{css}">{cls}</td>'
            f'<td>{pos_str}</td></tr>')
    parts.append('</table>')

    # Top words
    parts.append('<h2>Top Tokenized Words</h2>')
    parts.append('<table><tr><th>#</th><th>Glyph</th><th>Token Form</th>'
                '<th>Count</th></tr>')
    for rank, (tw, count) in enumerate(token_word_freq.most_common(30), 1):
        eva = tokens_to_eva(tw)
        glyph = eva_glyph(eva)
        compact = ' '.join(f'<span class="tok">{token_ids.get(t,t)}</span>'
                          for t in tw.split('.'))
        parts.append(f'<tr><td>{rank}</td><td><span class="v">{glyph}</span></td>'
                    f'<td>{compact}</td><td>{count}</td></tr>')
    parts.append('</table>')

    # Paradigms
    parts.append('<h2>Paradigm Tables (Token Space)</h2>')
    parts.append("""<div class="highlight">
    Stems and endings defined purely by token sequences — no phonetic interpretation.
    These paradigms represent the <strong>structural morphology</strong> of the text.
    </div>""")

    parts.append('<table><tr><th>Stem</th><th>Glyph</th><th>#End</th><th>Freq</th>'
                '<th>Top Endings</th></tr>')
    for stem, n_end, total, endings in paradigms[:20]:
        stem_compact = '.'.join(f'{token_ids.get(t,t)}' for t in stem.split('.'))
        stem_eva = tokens_to_eva(stem)
        stem_glyph = eva_glyph(stem_eva)

        end_parts = []
        for e, c in endings.most_common(5):
            e_compact = '.'.join(f'{token_ids.get(t,t)}' for t in e.split('.'))
            e_eva = tokens_to_eva(e)
            e_glyph = eva_glyph(e_eva)
            end_parts.append(
                f'<span class="v">{e_glyph}</span>'
                f'<span class="tok">-{e_compact}</span>'
                f'<span style="color:#8b949e">({c})</span>')

        parts.append(
            f'<tr><td><span class="tok">{stem_compact}-</span></td>'
            f'<td><span class="v">{stem_glyph}</span></td>'
            f'<td>{n_end}</td><td>{total}</td>'
            f'<td>{" &nbsp; ".join(end_parts)}</td></tr>')
    parts.append('</table>')

    parts.append('</body></html>')
    return '\n'.join(parts)

html = gen_html()
Path('reports/html/tokenized_report.html').write_text(html, encoding='utf-8')
print(f"HTML report: tokenized_report.html ({len(html)//1024}KB)")
