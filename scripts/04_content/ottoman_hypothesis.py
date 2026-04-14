"""
Ottoman Turkic Phoneme Mapping Hypothesis
==========================================
Attempt to map the reanalysed Voynich alphabet to Ottoman/Chagatai Turkic
phonemes, tested against astronomical section labels where we have
ground truth (known star names, zodiac signs).

Strategy:
  1. Anchor on structural constraints (frequency matching, position matching)
  2. Use zodiac labels as test cases (we know which sign each page depicts)
  3. Use star labels against known Arabic/Persian/Turkic star names
  4. Try multiple mapping variants and score each

The zodiac pages have a roughly established order:
  f70v1 → Aries/Pisces    f72r2 → Virgo/Leo
  f70v2 → Taurus/Aries    f72r3 → Libra/Virgo
  f71r  → Gemini/Taurus   f72v1 → Scorpio/Libra
  f71v  → Cancer/Gemini   f72v2 → Sagittarius/Scorpio
  f72r1 → Leo/Cancer      f72v3 → Capricorn/Sagittarius
                           f73r  → Aquarius/Capricorn
                           f73v  → Pisces/Aquarius
"""

import json
import re
from collections import Counter, defaultdict
from itertools import product

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with open('data/transcription/voynich_nlp.json') as f:
    data = json.load(f)
with open('data/analysis/astro_vocab.json') as f:
    astro = json.load(f)

metadata = data['metadata']

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
# Zodiac labels with their identified signs
# ---------------------------------------------------------------------------

# Two candidate orderings are debated. We test both.
# "Standard" = Koen Gheuens / Zandbergen identification
# "Shifted" = one position shifted (common alternative)

ZODIAC_LABELS = {
    'f70v1': {
        'labels': [('otalchy tar am dy', 'main'), ('okoly', 'secondary')],
        'standard': 'Aries', 'shifted': 'Pisces',
    },
    'f70v2': {
        'labels': [('otalalg', 'main'), ('sar am', 'mid'), ('otylal', 'secondary')],
        'standard': 'Taurus', 'shifted': 'Aries',
    },
    'f71r': {
        'labels': [('oteos arar', 'main'), ('otol chdy', 'secondary')],
        'standard': 'Gemini', 'shifted': 'Taurus',
    },
    'f71v': {
        'labels': [('char orom', 'main'), ('ofacfom', 'secondary')],
        'standard': 'Cancer', 'shifted': 'Gemini',
    },
    'f72r1': {
        'labels': [('oshodody', 'main'), ('ofaralar', 'secondary')],
        'standard': 'Leo', 'shifted': 'Cancer',
    },
    'f72r2': {
        'labels': [('ofchdady', 'main'), ('otaraldy', 'mid'), ('otal', 'secondary')],
        'standard': 'Virgo', 'shifted': 'Leo',
    },
    'f72r3': {
        'labels': [('olkalaiin', 'main'), ('or alkam', 'mid'), ('opalal', 'secondary')],
        'standard': 'Libra', 'shifted': 'Virgo',
    },
    'f72v1': {
        'labels': [('oeeoty', 'main'), ('oeeoly', 'secondary')],
        'standard': 'Scorpio', 'shifted': 'Libra',
    },
    'f72v2': {
        'labels': [('oeedey', 'main'), ('okaiin', 'secondary')],
        'standard': 'Sagittarius', 'shifted': 'Scorpio',
    },
    'f72v3': {
        'labels': [('ogeom', 'main'), ('sholeey', 'secondary')],
        'standard': 'Capricorn', 'shifted': 'Sagittarius',
    },
    'f73r': {
        'labels': [('otoly', 'main'), ('otaly', 'mid'), ('okary', 'secondary')],
        'standard': 'Aquarius', 'shifted': 'Capricorn',
    },
    'f73v': {
        'labels': [('okol', 'main'), ('otedy', 'mid'), ('okeody', 'secondary')],
        'standard': 'Pisces', 'shifted': 'Aquarius',
    },
}

# Near Eastern zodiac terms
ZODIAC_TERMS = {
    'Aries':       {'ar': 'al-Ḥamal', 'pe': 'bara', 'tu': 'qoç/koç'},
    'Taurus':      {'ar': 'al-Thawr/al-Sevr', 'pe': 'gāv', 'tu': 'öküz/boğa'},
    'Gemini':      {'ar': 'al-Jawzāʾ/Cevzā', 'pe': 'du-paykar', 'tu': 'ikizler'},
    'Cancer':      {'ar': 'al-Saraṭān/Seretān', 'pe': 'kharchang/harçeng', 'tu': 'yengeç'},
    'Leo':         {'ar': 'al-Asad/Esed', 'pe': 'shīr', 'tu': 'arslan'},
    'Virgo':       {'ar': 'al-Sunbula/Sünbüle', 'pe': 'khūsha', 'tu': 'başak'},
    'Libra':       {'ar': 'al-Mīzān/Mīzān', 'pe': 'tarāzū/terazi', 'tu': 'terazi'},
    'Scorpio':     {'ar': 'al-ʿAqrab/Akreb', 'pe': 'kazhdum', 'tu': 'akrep'},
    'Sagittarius': {'ar': 'al-Qaws/Kavs', 'pe': 'kamān/kemān', 'tu': 'yay'},
    'Capricorn':   {'ar': 'al-Jady/Cedī', 'pe': 'buzghāla', 'tu': 'oğlak'},
    'Aquarius':    {'ar': 'al-Dalw/Delv', 'pe': 'dalv', 'tu': 'kova'},
    'Pisces':      {'ar': 'al-Ḥūt/Hūt', 'pe': 'māhī', 'tu': 'balık'},
}

# Star names (Arabic forms as used in Ottoman astronomy)
STAR_NAMES = {
    'al-Dabarān': 'Aldebaran (the Follower)',
    'al-Shiʿrā': 'Sirius',
    'al-Nasr': 'Eagle (Vega/Altair stem)',
    'al-Rijl': 'Rigel (the Foot)',
    'al-Simāk': 'Arcturus/Spica',
    'Suhayl': 'Canopus',
    'al-ʿAyyūq': 'Capella',
    'al-Thurayyā': 'Pleiades',
    'al-Jady': 'Polaris (the Kid)',
    'al-Qalb': 'Antares (Heart)',
    'Ülker': 'Pleiades (Turkic)',
    'Çolpan': 'Venus (Turkic)',
    'Temür Qazuq': 'Polaris (Turkic)',
    'Yulduz': 'star (Turkic generic)',
}

# ---------------------------------------------------------------------------
# Phoneme mapping hypotheses
# ---------------------------------------------------------------------------

# Build mappings from EVA characters to Ottoman Turkish phonemes.
# We use the reanalysed H3 alphabet.
#
# ANCHORING CONSTRAINTS:
# 1. `o` is the most common char (12.7%) → likely /a/ (most common in Turkish ~11.9%)
# 2. `y` is very common (11.9%) → likely a common vowel, /ı/ or /i/
# 3. `e` after collapse (6.7%) → /e/ or /ε/
# 4. `EE` = long-e → /ö/ or /ü/ (front rounded, or long /ē/)
# 5. `AI` / `AII` = diphthong or long vowel → /-ān/ or /-in/
# 6. `d` (8.8%) too common for just /d/ → might encode /d/+/t/ (dental pair)
# 7. `CH` (7.4%) = single consonant → /ç/ (Ottoman ç) or /c/ (Ottoman c/dʒ)
# 8. `SH` (3.1%) → /ş/ (Ottoman ş)
# 9. `l` (7.1%) → /l/
# 10. `k` (6.7%) → /k/ or /g/ or both
# 11. `r` (4.9%) → /r/
# 12. `n` (4.3%) → /n/
# 13. `QO` (3.8%) → /q/ (Arabic qāf) or /ḳ/ + inherent /a/
# 14. `t` (3.8%) → /t/ or modifier (emphatic)
# 15. `s` (1.7%) → /s/ or /z/
# 16. `m` (0.8%) → /m/
# 17. `p` (0.9%) → /p/ or /b/
# 18. `f` (0.2%) → /f/ or /v/

# Mapping A: "Standard Ottoman" — most straightforward frequency match
MAPPING_A = {
    'o': 'a',       # most common → most common vowel
    'y': 'ı/i',     # second most common → Turkish high vowels
    'e': 'e',       # mid vowel
    'a': 'u/o',     # back rounded vowels (less common)
    'i': 'ī',       # appears in ai/aii sequences → long vowel marker
    'n': 'n',       # stable
    'd': 'd',       # dental stop
    'l': 'l',       # liquid
    'k': 'k/g',     # velar stops
    'r': 'r',       # rhotic
    't': 't/ṭ',     # dental/emphatic
    's': 's/z',     # sibilant
    'p': 'p/b',     # labial stop
    'f': 'f/v',     # labial fricative
    'm': 'm',       # nasal
    'q': 'ḳ',       # qāf
    'ch': 'ç',      # Ottoman ç
    'sh': 'ş',      # Ottoman ş
    'cth': 'c',     # Ottoman c (dʒ) — ch modified by t
    'ckh': 'ḫ/h',   # Ottoman ḫ (Arabic khā) — ch modified by k
    'cph': 'č̣',     # emphatic ç?
    'cfh': 'ğ',     # Ottoman ğ (soft g)?
    'qo': 'ḳa',    # qāf + inherent /a/
    'ee': 'ö/ü',    # front rounded vowels
    'ai': 'ay',     # diphthong
    'aii': 'ān',    # long vowel + n (nasalized?)
}

# Mapping B: "Abjad" — treating some EVA chars as consonant+vowel syllables
MAPPING_B = {
    'o': 'al-',     # definite article (always word-initial)
    'o_mid': 'a',   # when not word-initial, just /a/
    'y': 'ı',       # high back unrounded
    'e': 'e',
    'a': 'u',
    'i': 'ī',       # vowel length marker
    'n': 'n',
    'd': 'de/da',   # syllabic: /da/ or /de/ (explains high frequency)
    'l': 'l',
    'k': 'k',
    'r': 'r',
    't': 't',
    's': 's',
    'p': 'b',       # Ottoman b more common than p
    'f': 'f',
    'm': 'm',
    'q': 'q',
    'ch': 'ç',
    'sh': 'ş',
    'cth': 'c',
    'ckh': 'h',     # just /h/
    'cph': 'č',
    'cfh': 'ğ',
    'qo': 'qa',
    'ee': 'ö',
    'ai': 'ay',
    'aii': 'ān',
}

# ---------------------------------------------------------------------------
# Apply mapping to labels
# ---------------------------------------------------------------------------

def apply_mapping(eva_word, mapping, word_initial=True):
    """Apply a phoneme mapping to an EVA word.
    Returns the transliterated form.
    """
    w = eva_word

    # Pre-process: apply H3 rewriting
    # Trigraphs first
    w = w.replace('cth', '⟨cth⟩')
    w = w.replace('ckh', '⟨ckh⟩')
    w = w.replace('cph', '⟨cph⟩')
    w = w.replace('cfh', '⟨cfh⟩')
    w = w.replace('ch', '⟨ch⟩')
    w = w.replace('sh', '⟨sh⟩')

    # Handle qo
    w = w.replace('qo', '⟨qo⟩')

    # Handle vowel sequences: aiii, aii, ai, eee, ee
    w = re.sub(r'aiii', '⟨aiii⟩', w)
    w = re.sub(r'aii', '⟨aii⟩', w)
    w = re.sub(r'ai(?!⟩)', '⟨ai⟩', w)
    w = re.sub(r'eee', '⟨eee⟩', w)
    w = re.sub(r'ee', '⟨ee⟩', w)

    # Now map each unit
    result = []
    i = 0
    first_char = True
    while i < len(w):
        if w[i] == '⟨':
            end = w.index('⟩', i)
            unit = w[i+1:end]
            phoneme = mapping.get(unit, f'?{unit}?')
            result.append(phoneme)
            i = end + 1
            first_char = False
        else:
            ch = w[i]
            if ch == 'o' and first_char and word_initial and 'o_initial' in mapping:
                result.append(mapping['o_initial'])
            elif ch == 'o' and not first_char and 'o_mid' in mapping:
                result.append(mapping['o_mid'])
            else:
                result.append(mapping.get(ch, ch))
            i += 1
            first_char = False

    return ''.join(result)


# ---------------------------------------------------------------------------
# Build and test mappings
# ---------------------------------------------------------------------------

# Mapping C: "Ottoman astronomical" — tuned for known star/zodiac names
# Start from Mapping A but refine based on zodiac test
MAPPING_C = {
    'o': 'a',       # but word-initial o- could be al- (article)
    'o_initial': 'al-',  # word-initial = article
    'o_mid': 'a',
    'y': 'ı',       # most common vowel after /a/
    'e': 'e',
    'a': 'u',       # back rounded
    'i': 'ī',       # length marker
    'n': 'n',
    'd': 'd',
    'l': 'l',
    'k': 'k',
    'r': 'r',
    't': 't',
    's': 's',
    'p': 'b',
    'f': 'f',
    'm': 'm',
    'q': 'q',
    'ch': 'ç',
    'sh': 'ş',
    'cth': 'c',      # dʒ
    'ckh': 'h',      # h/ḫ
    'cph': 'č',
    'cfh': 'ğ',
    'qo': 'qa',
    'ee': 'ö',
    'ai': 'ay',
    'aii': 'ān',
    'aiii': 'ānī',
    'eee': 'öe',
}

# Mapping D: "Reversed vowels" — what if o=e and e=a? (test alternative)
MAPPING_D = dict(MAPPING_C)
MAPPING_D.update({
    'o': 'e',
    'o_initial': 'el-',
    'o_mid': 'e',
    'e': 'a',
    'y': 'i',
    'a': 'o',
    'ee': 'ā',
})

# Mapping E: "Semitic-style" — no separate initial article, o is just a vowel
MAPPING_E = dict(MAPPING_C)
MAPPING_E.update({
    'o_initial': 'a',  # just /a/, no article
    'ee': 'ü',
})

MAPPINGS = {
    'C (Ottoman+article)': MAPPING_C,
    'D (reversed vowels)': MAPPING_D,
    'E (no article)': MAPPING_E,
}

# ---------------------------------------------------------------------------
# Test each mapping against zodiac labels
# ---------------------------------------------------------------------------

print("="*100)
print("OTTOMAN TURKIC PHONEME MAPPING — ZODIAC LABEL TEST")
print("="*100)

for map_name, mapping in MAPPINGS.items():
    print(f"\n{'─'*100}")
    print(f"MAPPING: {map_name}")
    print(f"{'─'*100}")

    for folio, info in ZODIAC_LABELS.items():
        sign_std = info['standard']
        sign_sft = info['shifted']
        terms = ZODIAC_TERMS.get(sign_std, {})
        terms_sft = ZODIAC_TERMS.get(sign_sft, {})

        print(f"\n  {folio} — {sign_std} (or {sign_sft})")

        for label_text, label_type in info['labels']:
            # Split multi-word labels
            words = label_text.split()
            translits = []
            for w in words:
                tr = apply_mapping(w, mapping, word_initial=True)
                translits.append(tr)
            full_translit = ' '.join(translits)

            glyph = eva_glyph(label_text.replace(' ', '.'))

            print(f"    {label_type:9s}  EVA: {label_text:20s}  "
                  f"→  {full_translit:25s}  glyph: {glyph}")

        # Show target terms
        print(f"    Target ({sign_std}): Ar={terms.get('ar','?')}  "
              f"Pe={terms.get('pe','?')}  Tu={terms.get('tu','?')}")
        if sign_sft != sign_std:
            print(f"    Target ({sign_sft}): Ar={terms_sft.get('ar','?')}  "
                  f"Pe={terms_sft.get('pe','?')}  Tu={terms_sft.get('tu','?')}")

# ---------------------------------------------------------------------------
# Test against star labels
# ---------------------------------------------------------------------------

print("\n\n" + "="*100)
print("STAR LABEL TRANSLITERATION (Mapping C — Ottoman+article)")
print("="*100)

star_labels = [e for e in astro['label_entries'] if e['unit'] == 'Ls']

mapping = MAPPING_C

print(f"\n{'EVA':20s} {'Glyph':15s} {'Transliteration':25s} {'Candidate':30s}")
print("-"*95)

for entry in star_labels[:50]:
    label = ' '.join(entry['words'])
    words = label.split()
    translit_parts = []
    for w in words:
        tr = apply_mapping(w, mapping)
        translit_parts.append(tr)
    translit = ' '.join(translit_parts)

    glyph = eva_glyph(label.replace(' ', '.'))

    # Try to match against known star names
    candidate = ''
    tr_lower = translit.lower().replace('al-', '')

    for star, desc in STAR_NAMES.items():
        star_clean = star.lower().replace('al-', '').replace('ü', 'u').replace('-', '')
        if len(tr_lower) >= 3 and len(star_clean) >= 3:
            # Check for substring matches
            if tr_lower[:3] == star_clean[:3]:
                candidate = f'{star} ({desc})'
            elif tr_lower[:4] == star_clean[:4]:
                candidate = f'** {star} ({desc})'

    print(f"  {label:18s} {glyph:12s}  {translit:25s} {candidate}")

# ---------------------------------------------------------------------------
# Frequency-based phoneme validation
# ---------------------------------------------------------------------------

print("\n\n" + "="*100)
print("PHONEME FREQUENCY VALIDATION")
print("="*100)

# Apply mapping to ALL words and check if resulting phoneme frequencies
# match Ottoman Turkish letter frequencies

all_words = [w for s in data['sentences'] for w in s['words']]

ottoman_freqs = {
    'a': 11.9, 'e': 8.9, 'i': 8.6, 'ı': 5.1, 'n': 7.2, 'r': 6.7,
    'l': 5.9, 'k': 4.7, 'd': 4.7, 'ü': 1.9, 'ö': 0.8, 'b': 2.8,
    't': 3.1, 'm': 3.6, 's': 3.0, 'y': 3.4, 'ş': 1.8, 'ç': 1.1,
    'u': 3.4, 'o': 2.5, 'h': 1.2, 'z': 1.5, 'g': 1.3, 'p': 0.9,
    'v': 1.0, 'f': 0.5, 'c': 1.0,
}

mapping = MAPPING_C
phoneme_counts = Counter()
total_phonemes = 0

for w in all_words:
    translit = apply_mapping(w, mapping)
    for ch in translit:
        if ch.isalpha():
            phoneme_counts[ch.lower()] += 1
            total_phonemes += 1

print(f"\n{'Phoneme':>8s} {'Voynich%':>8s} {'Ottoman%':>8s} {'Diff':>6s}")
print("-"*35)

for phoneme, voynich_pct in sorted(
    [(p, 100*c/total_phonemes) for p, c in phoneme_counts.items()],
    key=lambda x: -x[1]
)[:20]:
    ottoman_pct = ottoman_freqs.get(phoneme, 0)
    diff = voynich_pct - ottoman_pct
    marker = '!!' if abs(diff) > 5 else '!' if abs(diff) > 3 else ''
    print(f"  {phoneme:>6s} {voynich_pct:8.1f} {ottoman_pct:8.1f} {diff:+6.1f} {marker}")

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def gen_html():
    import html as html_mod
    parts = ["""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Voynich — Ottoman Turkic Hypothesis: Astronomical Test</title>
<style>
@font-face { font-family: 'VoynichEVA'; src: url('fonts/Voynich/VoynichEVA.ttf') format('truetype'); }
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1400px; margin: 0 auto;
       padding: 20px; background: #0d1117; color: #c9d1d9; line-height: 1.6; }
h1 { color: #58a6ff; border-bottom: 2px solid #1f6feb; }
h2 { color: #79c0ff; margin-top: 40px; }
.v { font-family: 'VoynichEVA', serif; font-size: 1.5em; color: #ffa657; letter-spacing: 2px; }
.eva { font-family: monospace; background: #161b22; padding: 2px 6px; border-radius: 3px;
       color: #7ee787; }
.translit { font-family: monospace; background: #1a1030; padding: 2px 6px; border-radius: 3px;
            color: #d2a8ff; font-weight: bold; font-size: 1.1em; }
.target { color: #f78166; font-style: italic; }
table { border-collapse: collapse; margin: 15px 0; background: #161b22; width: 100%; }
th, td { border: 1px solid #30363d; padding: 8px 12px; text-align: left; }
th { background: #21262d; color: #58a6ff; }
tr:hover { background: #1c2128; }
.note { background: #1c2128; border-left: 4px solid #1f6feb; padding: 12px 15px; margin: 15px 0; }
.highlight { background: #2d1b00; border-left: 4px solid #d29922; padding: 12px 15px; margin: 15px 0; }
.zodiac-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               padding: 15px; margin: 10px 0; }
.zodiac-card h3 { margin: 0 0 10px 0; color: #ffa657; }
.mapping-table td:first-child { font-weight: bold; color: #7ee787; }
</style></head><body>
<h1>Ottoman Turkic Hypothesis — Astronomical Section Test</h1>
<div class="note">
<p>Testing a phoneme mapping derived from distributional analysis of the reanalysed
Voynich alphabet against known astronomical terminology. The mapping assumes
early Ottoman / Chagatai Turkic with Arabic-Persian astronomical vocabulary.</p>
</div>
"""]

    # Mapping table
    parts.append('<h2>Phoneme Mapping (Hypothesis C — Ottoman + Article)</h2>')
    parts.append('<table><tr><th>EVA Unit</th><th>Glyph</th><th>Phoneme</th><th>Rationale</th></tr>')

    rationales = {
        'o': 'Most common (12.7%) → most common Turkish vowel /a/ (11.9%)',
        'o_initial': 'Word-initial → Arabic definite article al-',
        'y': 'Very common (11.9%) → Turkish /ı/ high vowel',
        'e': 'Common (6.7%) → /e/',
        'a': 'Moderate (4.9%) → back rounded /u/',
        'ch': 'Confirmed ligature → Ottoman /ç/',
        'sh': 'Confirmed digraph → Ottoman /ş/',
        'cth': 'ch + t modifier → Ottoman /c/ (dʒ)',
        'ckh': 'ch + k modifier → /h/ or /ḫ/',
        'ee': 'Vowel length variant → /ö/ or /ü/',
        'ai': 'Diphthong unit → /ay/',
        'aii': 'Long diphthong → /ān/ (long a + nasal)',
        'qo': '97.6% co-occurrence → single unit /qa/',
        'd': 'Common consonant → /d/',
        'l': '→ /l/', 'k': '→ /k/', 'r': '→ /r/', 'n': '→ /n/',
        't': '→ /t/', 's': '→ /s/', 'm': '→ /m/',
        'p': '→ /b/ (Ottoman b more common)', 'f': '→ /f/',
    }

    for unit in ['o', 'o_initial', 'y', 'e', 'ee', 'a', 'ai', 'aii',
                 'ch', 'sh', 'cth', 'ckh', 'd', 'l', 'k', 'r', 'n',
                 'qo', 't', 's', 'p', 'f', 'm']:
        phoneme = MAPPING_C.get(unit, '?')
        eva = unit if unit not in ('o_initial', 'o_mid') else 'o (initial)'
        glyph = eva_glyph(eva.replace('_initial', '').replace('_mid', '')
                          .replace('ee','ee').replace('ai','ai'))
        rat = rationales.get(unit, '')
        parts.append(f'<tr><td><span class="eva">{eva}</span></td>'
                    f'<td><span class="v">{glyph}</span></td>'
                    f'<td><span class="translit">{phoneme}</span></td>'
                    f'<td>{rat}</td></tr>')
    parts.append('</table>')

    # Zodiac transliterations
    parts.append('<h2>Zodiac Label Transliterations</h2>')

    for folio, info in ZODIAC_LABELS.items():
        sign = info['standard']
        terms = ZODIAC_TERMS.get(sign, {})

        parts.append(f'<div class="zodiac-card">')
        parts.append(f'<h3>{folio} — {sign}</h3>')

        for label_text, lt in info['labels']:
            words = label_text.split()
            translit = ' '.join(apply_mapping(w, MAPPING_C) for w in words)
            glyph = ' '.join(eva_glyph(w) for w in words)

            parts.append(f'<div><span class="v">{glyph}</span> &nbsp; '
                        f'<span class="eva">{label_text}</span> &nbsp;→&nbsp; '
                        f'<span class="translit">{translit}</span></div>')

        parts.append(f'<div class="target">Target: Ar: {terms.get("ar","?")} '
                    f'| Pe: {terms.get("pe","?")} | Tu: {terms.get("tu","?")}</div>')
        parts.append('</div>')

    # Star labels
    parts.append('<h2>Star Label Transliterations</h2>')
    parts.append('<table><tr><th>EVA</th><th>Glyph</th><th>Transliteration</th>'
                '<th>Notes</th></tr>')

    for entry in star_labels[:40]:
        label = ' '.join(entry['words'])
        translit = ' '.join(apply_mapping(w, MAPPING_C) for w in label.split())
        glyph = ' '.join(eva_glyph(w) for w in label.split())

        parts.append(f'<tr><td><span class="eva">{label}</span></td>'
                    f'<td><span class="v">{glyph}</span></td>'
                    f'<td><span class="translit">{translit}</span></td>'
                    f'<td></td></tr>')

    parts.append('</table>')
    parts.append('</body></html>')
    return '\n'.join(parts)

html = gen_html()
from pathlib import Path
Path('reports/html/ottoman_hypothesis.html').write_text(html, encoding='utf-8')
print(f"\nHTML report: ottoman_hypothesis.html ({len(html)//1024}KB)")
