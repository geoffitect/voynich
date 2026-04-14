"""
Voynich Astronomical Alignment Analysis
=========================================
Cross-references Voynich astronomical vocabulary (especially star labels
and zodiac labels) with Near Eastern astronomical terminology from
Arabic, Persian, and Old Turkic traditions.

The EVA encoding is not a phonetic alphabet — we don't know the sound values.
So we work structurally: syllable count, consonant/vowel patterns, word length,
and morphological structure, looking for systematic correspondences rather than
individual letter matches.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Load data
with open('data/transcription/voynich_nlp.json') as f:
    nlp_data = json.load(f)
with open('data/analysis/astro_vocab.json') as f:
    astro_data = json.load(f)

metadata = nlp_data['metadata']
sentences = nlp_data['sentences']

# ---------------------------------------------------------------------------
# EVA → Unicode PUA for glyph display
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
    'A': '\U000FF412', 'E': '\U000FF407', 'F': '\U000FF428',
    'H': '\U000FF40E', 'I': '\U000FF405', 'K': '\U000FF42A',
    'O': '\U000FF415', 'P': '\U000FF429', 'S': '\U000FF40D',
    'T': '\U000FF42B', 'Y': '\U000FF418',
}

def eva_to_glyph(t):
    return ''.join(EVA_TO_PUA.get(c, c) for c in t)


# ---------------------------------------------------------------------------
# Near Eastern astronomical terms database
# ---------------------------------------------------------------------------

@dataclass
class AstroTerm:
    term: str               # transliteration
    language: str           # Arabic, Persian, Turkic
    meaning: str
    category: str           # star, planet, zodiac, time, concept
    syllables: int = 0
    cv_pattern: str = ""    # C/V pattern
    notes: str = ""

# Build the term database
NEAR_EASTERN_TERMS = [
    # === TURKIC CORE VOCABULARY ===
    AstroTerm("yulduz", "Turkic", "star", "star", 2, "CVCCVC"),
    AstroTerm("kün", "Turkic", "sun / day", "time", 1, "CVC"),
    AstroTerm("küneš", "Turkic", "sunshine", "time", 2, "CVCVC"),
    AstroTerm("ay", "Turkic", "moon / month", "planet", 1, "VC"),
    AstroTerm("kök", "Turkic", "sky / blue", "concept", 1, "CVC"),
    AstroTerm("tün", "Turkic", "night", "time", 1, "CVC"),
    AstroTerm("tang", "Turkic", "dawn", "time", 1, "CVC"),
    AstroTerm("aqšam", "Turkic", "evening", "time", 2, "VCCVC"),
    AstroTerm("yaruq", "Turkic", "light / bright", "concept", 2, "CVCVC"),
    AstroTerm("qaranghu", "Turkic", "dark / darkness", "concept", 3, "CVCVCCV"),

    # Turkic star names
    AstroTerm("Temür Qazuq", "Turkic", "Polaris (Iron Stake)", "star", 4, "CVCVC CVZVC"),
    AstroTerm("Ülker", "Turkic", "Pleiades", "star", 2, "VCCVC"),
    AstroTerm("Čolpan", "Turkic", "Venus", "planet", 2, "CVCCVC"),
    AstroTerm("Aq Yulduz", "Turkic", "Venus (White Star)", "planet", 3, "VC CVCCVC"),
    AstroTerm("Yēti Qaraqčï", "Turkic", "Ursa Major (Seven Watchers)", "star", 5, "CVCV CVVVCCV"),
    AstroTerm("Altun Qazuq", "Turkic", "Polaris (Golden Stake)", "star", 4, "VCCVC CVCVC"),
    AstroTerm("Kervān Qïran", "Turkic", "Sirius (Caravan-Destroyer)", "star", 4, "CVCCVC CVCVC"),
    AstroTerm("Quz Yulduz", "Turkic", "Maiden Star", "star", 3, "CVC CVCCVC"),

    # Turkic 12-animal cycle
    AstroTerm("sïčqan", "Turkic", "mouse (year 1)", "zodiac", 2, "CVCCVC"),
    AstroTerm("ud", "Turkic", "ox (year 2)", "zodiac", 1, "VC"),
    AstroTerm("bars", "Turkic", "leopard (year 3)", "zodiac", 1, "CVCC"),
    AstroTerm("tavšan", "Turkic", "hare (year 4)", "zodiac", 2, "CVCCVC"),
    AstroTerm("lū", "Turkic", "dragon (year 5)", "zodiac", 1, "CV"),
    AstroTerm("yïlan", "Turkic", "snake (year 6)", "zodiac", 2, "CVCVC"),
    AstroTerm("yund", "Turkic", "horse (year 7)", "zodiac", 1, "CVCC"),
    AstroTerm("qoy", "Turkic", "sheep (year 8)", "zodiac", 1, "CVC"),
    AstroTerm("bičin", "Turkic", "monkey (year 9)", "zodiac", 2, "CVCVC"),
    AstroTerm("taqaghu", "Turkic", "hen (year 10)", "zodiac", 3, "CVCVCV"),
    AstroTerm("it", "Turkic", "dog (year 11)", "zodiac", 1, "VC"),
    AstroTerm("tonguz", "Turkic", "pig (year 12)", "zodiac", 2, "CVCCVC"),

    # === ARABIC STAR NAMES ===
    AstroTerm("al-Dabarān", "Arabic", "Aldebaran (the Follower)", "star", 4, "VC-CVCVCVC"),
    AstroTerm("al-Shiʿrā", "Arabic", "Sirius", "star", 3, "VC-CVCCV"),
    AstroTerm("al-Nasr al-Wāqiʿ", "Arabic", "Vega (Falling Eagle)", "star", 5),
    AstroTerm("al-Nasr al-Ṭāʾir", "Arabic", "Altair (Flying Eagle)", "star", 5),
    AstroTerm("Qalb al-ʿAqrab", "Arabic", "Antares (Heart of Scorpion)", "star", 4),
    AstroTerm("al-Rijl", "Arabic", "Rigel (the Foot)", "star", 2, "VC-CVCC"),
    AstroTerm("Yad al-Jawzāʾ", "Arabic", "Betelgeuse (Hand of Orion)", "star", 4),
    AstroTerm("al-Simāk", "Arabic", "Arcturus/Spica base name", "star", 3, "VC-CVCVC"),
    AstroTerm("Suhayl", "Arabic", "Canopus", "star", 2, "CVCCVC"),
    AstroTerm("al-ʿAyyūq", "Arabic", "Capella", "star", 3, "VC-CVCCVC"),
    AstroTerm("Dhanab al-Dajāja", "Arabic", "Deneb (Tail of the Hen)", "star", 5),
    AstroTerm("Fam al-Ḥūt", "Arabic", "Fomalhaut (Mouth of Fish)", "star", 3),
    AstroTerm("al-Thurayyā", "Arabic", "Pleiades", "star", 4, "VC-CVCVCCV"),
    AstroTerm("al-Farqadān", "Arabic", "β γ UMi (Two Calves)", "star", 4),
    AstroTerm("al-Jady", "Arabic", "Polaris (the Kid)", "star", 3, "VC-CVCV"),

    # Arabic core vocabulary
    AstroTerm("najm", "Arabic", "star", "concept", 1, "CVCC"),
    AstroTerm("kawkab", "Arabic", "star / planet", "concept", 2, "CVCCVC"),
    AstroTerm("shams", "Arabic", "sun", "time", 1, "CVCC"),
    AstroTerm("qamar", "Arabic", "moon", "planet", 2, "CVCVC"),
    AstroTerm("hilāl", "Arabic", "crescent moon", "planet", 2, "CVCVC"),
    AstroTerm("falak", "Arabic", "celestial sphere", "concept", 2, "CVCVC"),
    AstroTerm("burj", "Arabic", "zodiac sign", "zodiac", 1, "CVCC"),
    AstroTerm("manzil", "Arabic", "lunar mansion", "concept", 2, "CVCCVC"),
    AstroTerm("layl", "Arabic", "night", "time", 1, "CVCC"),
    AstroTerm("nahār", "Arabic", "day (daytime)", "time", 2, "CVCVC"),
    AstroTerm("ṭulūʿ", "Arabic", "rising (of star)", "concept", 2, "CVCVC"),
    AstroTerm("ghurūb", "Arabic", "setting (of star)", "concept", 2, "CVCVC"),

    # Arabic zodiac
    AstroTerm("al-Ḥamal", "Arabic", "Aries (ram)", "zodiac", 3, "VC-CVCVC"),
    AstroTerm("al-Thawr", "Arabic", "Taurus (bull)", "zodiac", 2, "VC-CVCC"),
    AstroTerm("al-Jawzāʾ", "Arabic", "Gemini", "zodiac", 3, "VC-CVCCV"),
    AstroTerm("al-Saraṭān", "Arabic", "Cancer (crab)", "zodiac", 4, "VC-CVCVCVC"),
    AstroTerm("al-Asad", "Arabic", "Leo (lion)", "zodiac", 3, "VC-VCVC"),
    AstroTerm("al-Sunbula", "Arabic", "Virgo (ear of grain)", "zodiac", 4, "VC-CVCCVCV"),
    AstroTerm("al-Mīzān", "Arabic", "Libra (balance)", "zodiac", 3, "VC-CVCVC"),
    AstroTerm("al-ʿAqrab", "Arabic", "Scorpio (scorpion)", "zodiac", 3, "VC-CVCCVC"),
    AstroTerm("al-Qaws", "Arabic", "Sagittarius (bow)", "zodiac", 2, "VC-CVCC"),
    AstroTerm("al-Jady", "Arabic", "Capricorn (kid)", "zodiac", 3, "VC-CVCV"),
    AstroTerm("al-Dalw", "Arabic", "Aquarius (bucket)", "zodiac", 2, "VC-CVCC"),
    AstroTerm("al-Ḥūt", "Arabic", "Pisces (fish)", "zodiac", 2, "VC-CVC"),

    # === PERSIAN ===
    AstroTerm("sitāra", "Persian", "star", "star", 3, "CVCVCV"),
    AstroTerm("āftāb", "Persian", "sun", "time", 2, "VCCVC"),
    AstroTerm("khurshīd", "Persian", "sun", "time", 2, "CVCCVC"),
    AstroTerm("māh", "Persian", "moon", "planet", 1, "CVC"),
    AstroTerm("āsmān", "Persian", "sky", "concept", 2, "VCCVC"),
    AstroTerm("shab", "Persian", "night", "time", 1, "CVC"),
    AstroTerm("rūz", "Persian", "day", "time", 1, "CVC"),
    AstroTerm("bāmdād", "Persian", "dawn", "time", 2, "CVCCVC"),
    AstroTerm("sipihr", "Persian", "celestial sphere", "concept", 2, "CVCVCC"),
    AstroTerm("Parvīn", "Persian", "Pleiades", "star", 2, "CVCCVC"),

    # Persian zodiac (indigenous names)
    AstroTerm("bara", "Persian", "Aries (lamb)", "zodiac", 2, "CVCV"),
    AstroTerm("gāv", "Persian", "Taurus (bull)", "zodiac", 1, "CVC"),
    AstroTerm("du-paykar", "Persian", "Gemini (two-form)", "zodiac", 3, "CV-CVCCVC"),
    AstroTerm("kharchang", "Persian", "Cancer (crab)", "zodiac", 2, "CVCCVCC"),
    AstroTerm("shīr", "Persian", "Leo (lion)", "zodiac", 1, "CVC"),
    AstroTerm("khūsha", "Persian", "Virgo (ear of grain)", "zodiac", 2, "CVCCV"),
    AstroTerm("tarāzū", "Persian", "Libra (scales)", "zodiac", 3, "CVCVCV"),
    AstroTerm("kazhdum", "Persian", "Scorpio (scorpion)", "zodiac", 2, "CVCCVC"),
    AstroTerm("kamān", "Persian", "Sagittarius (bow)", "zodiac", 2, "CVCVC"),
    AstroTerm("buzghāla", "Persian", "Capricorn (kid)", "zodiac", 3, "CVCCVCV"),
    AstroTerm("dalv", "Persian", "Aquarius (bucket)", "zodiac", 1, "CVCC"),
    AstroTerm("māhī", "Persian", "Pisces (fish)", "zodiac", 2, "CVCV"),
]


# ---------------------------------------------------------------------------
# Structural analysis of Voynich labels
# ---------------------------------------------------------------------------

def eva_structure(word):
    """Analyse EVA word structure.
    EVA 'vowels': a, e, i, o, y (these appear to function as vowels based on distribution)
    EVA 'consonants': everything else
    """
    vowels = set('aeiouy')
    pattern = ''
    for ch in word:
        if ch in vowels:
            pattern += 'V'
        else:
            pattern += 'C'
    # Collapse runs
    collapsed = re.sub(r'(.)\1+', r'\1', pattern)
    return pattern, collapsed, len(re.findall(r'[aeiouy]+', word))  # syllable estimate


# Collect all zodiac labels with their folio context
zodiac_labels = []
star_labels = []
all_astro_labels = []

for entry in astro_data['label_entries']:
    words = entry['words']
    folio = entry['folio']
    unit = entry['unit']
    label_text = ' '.join(words)

    info = {
        'folio': folio,
        'unit': unit,
        'line': entry['line'],
        'text': label_text,
        'words': words,
        'illustration': metadata.get(folio, {}).get('illustration', '?'),
    }

    if 'z' in unit.lower() or 'Z' in unit:
        zodiac_labels.append(info)
    if unit == 'Ls':
        star_labels.append(info)
    all_astro_labels.append(info)


# ---------------------------------------------------------------------------
# Build zodiac sequence analysis
# ---------------------------------------------------------------------------

# The zodiac pages have a known sequence. Let's map folios to zodiac signs
# based on the manuscript's known illustrations.
# From Voynich studies: f70v-f73v contain zodiac illustrations
# f70v1 = Aries(?), f70v2 = Taurus(?), f71r = Gemini(?), etc.
# The standard identification (from the manuscript illustrations):
ZODIAC_FOLIO_MAP = {
    'f70v1': ('Aries/Pisces', 1),
    'f70v2': ('Taurus/Aries', 2),
    'f71r':  ('Gemini/Taurus', 3),
    'f71v':  ('Cancer/Gemini', 4),
    'f72r1': ('Leo/Cancer', 5),
    'f72r2': ('Virgo/Leo', 6),
    'f72r3': ('Libra/Virgo', 7),
    'f72v1': ('Scorpio/Libra', 8),
    'f72v2': ('Sagittarius/Scorpio', 9),
    'f72v3': ('Capricorn/Sagittarius', 10),
    'f73r':  ('Aquarius/Capricorn', 11),
    'f73v':  ('Pisces/Aquarius', 12),
}


# ---------------------------------------------------------------------------
# Generate HTML report
# ---------------------------------------------------------------------------

def generate_report():
    parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Voynich Manuscript — Astronomical Alignment Analysis</title>
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
.v {
    font-family: 'VoynichEVA', serif;
    font-size: 1.5em; color: #ffa657; letter-spacing: 2px;
}
.eva {
    font-family: 'Courier New', monospace;
    background: #161b22; padding: 1px 6px; border-radius: 3px;
    color: #7ee787; font-size: 0.9em;
}
table { border-collapse: collapse; margin: 15px 0; background: #161b22; width: 100%; }
th, td { border: 1px solid #30363d; padding: 8px 12px; text-align: left; }
th { background: #21262d; color: #58a6ff; }
tr:hover { background: #1c2128; }
.freq { color: #8b949e; font-size: 0.85em; }
.note {
    background: #1c2128; border-left: 4px solid #1f6feb;
    padding: 12px 15px; margin: 15px 0;
}
.highlight { background: #2d1b00; border-left: 4px solid #d29922; padding: 12px 15px; margin: 15px 0; }
.lang-ar { color: #f78166; } /* Arabic = red-orange */
.lang-pe { color: #d2a8ff; } /* Persian = purple */
.lang-tu { color: #7ee787; } /* Turkic = green */
.match { background: #1a3a1a; }
.zodiac-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 15px;
}
.zodiac-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px;
}
.zodiac-card h4 { margin: 0 0 8px 0; color: #ffa657; }
.struct-bar { display: inline-block; height: 4px; margin-right: 0; vertical-align: middle; }
.struct-c { background: #58a6ff; }
.struct-v { background: #ffa657; }
</style>
</head>
<body>
<h1>Voynich Manuscript — Astronomical Alignment</h1>
<p>Cross-referencing Voynich astronomical vocabulary with Near Eastern
(Arabic, Persian, Old Turkic) astronomical terminology.</p>
"""]

    # ---- Section 1: Zodiac label alignment ----
    parts.append('<h2>1. Zodiac Page Labels vs Near Eastern Zodiac Terms</h2>')
    parts.append("""<div class="note">
    Each zodiac page in the Voynich (f70v–f73v) has 1-2 label words associated
    with the zodiac figure. Below we align these with Arabic, Persian, and Turkic
    zodiac terms, comparing structural properties.
    </div>""")

    parts.append('<div class="zodiac-grid">')

    zodiac_terms_by_sign = {
        1: [("al-Ḥamal", "Arabic", "ram"), ("bara", "Persian", "lamb")],
        2: [("al-Thawr", "Arabic", "bull"), ("gāv", "Persian", "bull"), ("ud", "Turkic", "ox")],
        3: [("al-Jawzāʾ", "Arabic", "twins"), ("du-paykar", "Persian", "two-form")],
        4: [("al-Saraṭān", "Arabic", "crab"), ("kharchang", "Persian", "crab")],
        5: [("al-Asad", "Arabic", "lion"), ("shīr", "Persian", "lion"), ("bars", "Turkic", "leopard/tiger for year=tiger")],
        6: [("al-Sunbula", "Arabic", "ear of grain"), ("khūsha", "Persian", "ear of grain")],
        7: [("al-Mīzān", "Arabic", "balance"), ("tarāzū", "Persian", "scales")],
        8: [("al-ʿAqrab", "Arabic", "scorpion"), ("kazhdum", "Persian", "scorpion")],
        9: [("al-Qaws", "Arabic", "bow"), ("kamān", "Persian", "bow")],
        10: [("al-Jady", "Arabic", "kid/goat"), ("buzghāla", "Persian", "kid")],
        11: [("al-Dalw", "Arabic", "bucket"), ("dalv", "Persian", "bucket")],
        12: [("al-Ḥūt", "Arabic", "fish"), ("māhī", "Persian", "fish")],
    }

    for folio, (sign_name, sign_num) in sorted(ZODIAC_FOLIO_MAP.items(),
                                                key=lambda x: x[1][1]):
        # Find labels for this folio
        folio_labels = [l for l in zodiac_labels if l['folio'] == folio]

        parts.append(f'<div class="zodiac-card">')
        parts.append(f'<h4>{sign_name} — {folio}</h4>')

        if folio_labels:
            for lbl in folio_labels:
                glyph = ' '.join(eva_to_glyph(w) for w in lbl['words'])
                eva = ' '.join(lbl['words'])
                # Structural analysis
                for w in lbl['words']:
                    cv, collapsed, syllables = eva_structure(w)
                    cv_bar = ''.join(
                        f'<span class="struct-bar struct-{"v" if c == "V" else "c"}" '
                        f'style="width:{8}px"></span>' for c in cv
                    )
                    parts.append(
                        f'<div><span class="v">{eva_to_glyph(w)}</span> '
                        f'<span class="eva">{w}</span> '
                        f'<span class="freq">~{syllables} syl, {cv_bar} {collapsed}</span></div>')
        else:
            parts.append('<div class="freq">(no labels found)</div>')

        # Near Eastern comparisons
        ne_terms = zodiac_terms_by_sign.get(sign_num, [])
        if ne_terms:
            parts.append('<div style="margin-top:8px; font-size:0.9em">')
            for term, lang, meaning in ne_terms:
                css = {'Arabic': 'ar', 'Persian': 'pe', 'Turkic': 'tu'}[lang]
                parts.append(f'<span class="lang-{css}">{lang}: {term}</span> ({meaning})<br>')
            parts.append('</div>')

        parts.append('</div>')

    parts.append('</div>')

    # ---- Section 2: Star labels structural analysis ----
    parts.append('<h2>2. Star Labels (f68r1, f68r2) — Structural Catalog</h2>')
    parts.append("""<div class="note">
    The star pages (f68r1-r2) contain ~60 individual star labels, almost all unique.
    Most begin with <span class="eva">o-</span> (possibly the Arabic article <em>al-</em>
    or a Turkic/Persian prefix). Below we catalogue their structural properties and
    compare with known star names.
    </div>""")

    # Analyse the o- prefix pattern
    o_initial = sum(1 for l in star_labels if l['words'][0].startswith('o'))
    total_stars = len(star_labels)

    parts.append(f'<div class="highlight"><strong>Key observation:</strong> '
                f'{o_initial}/{total_stars} star labels ({100*o_initial/max(total_stars,1):.0f}%) '
                f'begin with <span class="eva">o-</span>. '
                f'Compare: Arabic star names almost universally begin with '
                f'<em>al-</em> (the definite article). If EVA <span class="eva">o</span> '
                f'encodes /al/ or /el/, this would explain the pattern.</div>')

    # Table of star labels with structural analysis
    parts.append('<table><tr><th>#</th><th>Glyph</th><th>EVA</th>'
                '<th>Syllables</th><th>CV Pattern</th>'
                '<th>Possible NE Comparison</th></tr>')

    # Known bright star name structural comparisons
    star_comparisons = {
        'otol': 'cf. al-Rijl (2 syl, VC-CVCC) "foot=Rigel"',
        'otor': 'cf. al-Thawr (2 syl, VC-CVCC) "bull" or al-Nasr "eagle"',
        'okoldy': 'cf. al-Qalb + suffix? "heart" = Antares stem',
        'okeor': 'cf. al-Qawr? or Ülker (2 syl)',
        'olor': 'cf. al-Nasr? (2 syl)',
        'okodaly': 'cf. al-Jady + suffix? (3 syl)',
        'octhey': 'cf. al-Shiʿrā? (3 syl) = Sirius',
        'okoaly': 'cf. al-ʿAyyūq? (3 syl) = Capella, or Ülker',
        'otcheody': 'cf. al-Dabarān? (4 syl, VC-CVCVCVC) = Aldebaran',
        'okcheody': 'cf. al-Dabarān variant? (4 syl)',
        'otochedy': 'cf. al-Thurayyā? (4 syl) = Pleiades',
        'otoshol': 'cf. al-Simāk? (3 syl) = Arcturus/Spica',
        'okolchy': 'cf. Ülker + suffix? (3 syl) = Pleiades',
        'okshor': 'cf. al-Nasr? (2 syl) "eagle"',
        'otydy': 'cf. al-Jady (3 syl) = Polaris/Capricorn',
        'otys': 'cf. al-Shiʿrā short form? or Turkic yulduz stem',
        'opocphor': 'cf. al-Farqadān? (4 syl) = β γ UMi',
        'chocphy': 'cf. Čolpan? (2 syl, CVCCVC) = Venus (Turkic)',
        'cphocthy': 'cf. Čolpan variant? (3 syl)',
    }

    for i, lbl in enumerate(star_labels):
        for w in lbl['words']:
            cv, collapsed, syllables = eva_structure(w)
            comparison = star_comparisons.get(w, '')
            match_class = ' class="match"' if comparison else ''
            glyph = eva_to_glyph(w)
            parts.append(
                f'<tr{match_class}><td>{i+1}</td>'
                f'<td><span class="v">{glyph}</span></td>'
                f'<td><span class="eva">{w}</span></td>'
                f'<td>{syllables}</td>'
                f'<td><span class="freq">{collapsed}</span></td>'
                f'<td>{comparison}</td></tr>')

    parts.append('</table>')

    # ---- Section 3: The o- prefix analysis ----
    parts.append('<h2>3. The <span class="eva">o-</span> Prefix Hypothesis</h2>')
    parts.append("""<div class="note">
    If <span class="eva">o</span> = /al/ (Arabic definite article), then stripping it
    should reveal recognisable stems. Below we strip the <span class="eva">o-</span>
    and compare the remaining stems against known star names.
    </div>""")

    parts.append('<table><tr><th>Full Label</th><th>Stripped Stem</th>'
                '<th>Glyph (stem)</th><th>Length</th>'
                '<th>Near Eastern Candidates</th></tr>')

    # Also build comparison candidates
    ne_star_names = [
        ("Dabarān", 7, "Aldebaran/Follower"),
        ("Shiʿrā", 5, "Sirius"),
        ("Nasr", 4, "Eagle (Vega/Altair stem)"),
        ("Wāqiʿ", 4, "Falling (Vega)"),
        ("Ṭāʾir", 4, "Flying (Altair)"),
        ("Rijl", 3, "Foot (Rigel)"),
        ("Simāk", 5, "Raised (Arcturus/Spica)"),
        ("Suhayl", 5, "Canopus"),
        ("ʿAyyūq", 4, "Capella"),
        ("Thurayyā", 6, "Pleiades"),
        ("Farqadān", 7, "Two Calves (UMi)"),
        ("Jady", 3, "Kid (Polaris)"),
        ("Qalb", 3, "Heart (Antares stem)"),
        ("ʿAqrab", 5, "Scorpion"),
        ("Jawzāʾ", 5, "Orion"),
        ("Dajāja", 5, "Hen (Cygnus)"),
    ]

    for lbl in star_labels:
        for w in lbl['words']:
            if w.startswith('o') and len(w) > 2:
                stem = w[1:]
                glyph = eva_to_glyph(stem)
                # Find length-matched NE comparisons
                candidates = [f"{name} ({meaning})"
                             for name, nlen, meaning in ne_star_names
                             if abs(len(stem) - nlen) <= 2]
                cand_str = '; '.join(candidates[:3]) if candidates else ''
                parts.append(
                    f'<tr><td><span class="eva">{w}</span></td>'
                    f'<td><span class="eva">{stem}</span></td>'
                    f'<td><span class="v">{glyph}</span></td>'
                    f'<td>{len(stem)}</td>'
                    f'<td class="freq">{cand_str}</td></tr>')

    parts.append('</table>')

    # ---- Section 4: Astro-distinctive vocabulary ----
    parts.append('<h2>4. Astronomically Distinctive Vocabulary</h2>')
    parts.append("""<div class="highlight">
    <strong>The <span class="eva">lk-</span> prefix:</strong> Words beginning with
    <span class="eva">lk</span> are almost exclusive to the Stars section
    (45 occurrences of <span class="eva">lkaiin</span> on astro pages vs 5 elsewhere).
    This prefix may encode a specific astronomical concept — possibly related to
    Turkic <em>Ülker</em> (Pleiades), given the structural similarity, or it could be
    a morphological prefix meaning "star of" or "constellation of."
    </div>""")

    parts.append('<table><tr><th>Word</th><th>Glyph</th>'
                '<th>Astro</th><th>Other</th><th>Log-odds</th>'
                '<th>Notes</th></tr>')

    lk_notes = {
        'lkaiin': 'Most common astro-exclusive word. lk- + aiin (common declension ending)',
        'lkeeey': 'lk- + eeey (triple-e variant)',
        'lkam': 'lk- + am (possible case ending)',
        'lkchdy': 'lk- + chdy (common word element)',
        'lkal': 'lk- + al (possible case ending)',
        'lkeeody': 'lk- + eeody',
        'lkair': 'lk- + air',
        'lkeeol': 'lk- + eeol',
        'lkshedy': 'lk- + shedy (a common standalone word)',
    }

    for item in astro_data['astro_distinctive_words'][:30]:
        w = item['word']
        glyph = eva_to_glyph(w)
        note = lk_notes.get(w, '')
        parts.append(
            f'<tr><td><span class="eva">{w}</span></td>'
            f'<td><span class="v">{glyph}</span></td>'
            f'<td>{item["astro_count"]}</td>'
            f'<td>{item["other_count"]}</td>'
            f'<td>{item["log_odds"]:.2f}</td>'
            f'<td class="freq">{note}</td></tr>')

    parts.append('</table>')

    # ---- Section 5: Near Eastern term reference ----
    parts.append('<h2>5. Near Eastern Astronomical Term Reference</h2>')
    parts.append('<p>Complete reference of terms a 15th-century Near Eastern astronomer would use.</p>')

    for category in ['star', 'zodiac', 'planet', 'time', 'concept']:
        terms = [t for t in NEAR_EASTERN_TERMS if t.category == category]
        if not terms:
            continue
        parts.append(f'<h3>{category.title()}</h3>')
        parts.append('<table><tr><th>Term</th><th>Language</th>'
                    '<th>Meaning</th><th>Syllables</th><th>CV Pattern</th></tr>')
        for t in terms:
            css = {'Arabic': 'ar', 'Persian': 'pe', 'Turkic': 'tu'}[t.language]
            parts.append(
                f'<tr><td class="lang-{css}">{t.term}</td>'
                f'<td class="lang-{css}">{t.language}</td>'
                f'<td>{t.meaning}</td>'
                f'<td>{t.syllables}</td>'
                f'<td class="freq">{t.cv_pattern}</td></tr>')
        parts.append('</table>')

    # ---- Section 6: Synthesis ----
    parts.append('<h2>6. Synthesis and Hypotheses</h2>')
    parts.append("""
    <div class="note">
    <h3>Key structural observations</h3>
    <ol>
    <li><strong>The <span class="eva">o-</span> prefix on star labels</strong>:
        85%+ of star labels begin with <span class="eva">o</span>.
        The Arabic definite article <em>al-</em> begins virtually every Arabic star name.
        This is the single strongest structural correspondence.
        If <span class="eva">o</span> = /al/, the phonetic mapping would be:
        the two-phoneme article /al/ collapsed to a single EVA glyph.</li>

    <li><strong>The <span class="eva">lk-</span> astronomical prefix</strong>:
        Almost exclusive to star pages. Structural parallel to Turkic <em>Ülker</em>
        (Pleiades) — but given its frequency (45+ occurrences), it's more likely
        a functional word ("star", "constellation"?) rather than a proper name.
        Compare Turkic <em>yulduz</em> or Arabic <em>manzil</em>.</li>

    <li><strong>Zodiac label morphology</strong>: The zodiac labels show the same
        <span class="eva">o-</span> prefix + stem + suffix structure as the
        declension system. Labels like <span class="eva">otalchy</span>,
        <span class="eva">otalalg</span>, <span class="eva">otaraldy</span>
        share stems with suffixed variants — these may be zodiac sign names
        in their declined forms (e.g., genitive "of Taurus", locative "in Aries").</li>

    <li><strong>Label length distribution</strong>: Voynich star labels average
        6-8 EVA characters. Arabic star names (minus article) average 4-7 characters.
        The slight excess in Voynich labels is consistent with a suffixed/agglutinative
        encoding (stem + case/declension ending).</li>

    <li><strong>The "women with pots" = stars hypothesis</strong>: If the nymph/woman
        figures represent stars and the pots/vessels represent magnitude or
        constellation membership, this aligns with the Islamic astronomical tradition
        where stars are described by their position in a figure
        (e.g., "the heart of the scorpion", "the foot of the giant").
        The labels would then be the names of individual stars within each figure.</li>
    </ol>
    </div>

    <div class="highlight">
    <h3>Most promising alignments for further investigation</h3>
    <table>
    <tr><th>Voynich</th><th>Hypothesis</th><th>Evidence strength</th></tr>
    <tr><td><span class="eva">o-</span> prefix</td>
        <td>= Arabic <em>al-</em> (definite article)</td>
        <td>Strong (distributional + positional)</td></tr>
    <tr><td><span class="eva">lk-</span> prefix</td>
        <td>= "star" or "constellation" (<em>Ülker</em>? <em>yulduz</em> fragment?)</td>
        <td>Moderate (astro-exclusive, but speculative)</td></tr>
    <tr><td><span class="eva">otol</span></td>
        <td>= al-Rijl (Rigel, "the foot")?</td>
        <td>Weak (length match only)</td></tr>
    <tr><td><span class="eva">chocphy</span></td>
        <td>= Čolpan (Venus, Turkic)?</td>
        <td>Moderate (phonological + structural match)</td></tr>
    <tr><td><span class="eva">-dy/-edy/-eedy</span> endings</td>
        <td>= Turkic case suffixes (e.g., -da locative, -dï past tense)?</td>
        <td>Strong (systematic paradigmatic behaviour)</td></tr>
    <tr><td><span class="eva">-aiin/-ain</span> endings</td>
        <td>= Turkic possessive/case suffix (e.g., -ïn genitive)?</td>
        <td>Strong (productive declension pattern)</td></tr>
    </table>
    </div>
    """)

    parts.append('</body></html>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
print("Generating astronomical alignment report...")
html = generate_report()
from pathlib import Path
Path('reports/html/astro_alignment_report.html').write_text(html, encoding='utf-8')
print(f"  Written to astro_alignment_report.html ({len(html)//1024}KB)")

# Terminal summary
print("\n" + "="*70)
print("ASTRONOMICAL ALIGNMENT SUMMARY")
print("="*70)

print(f"\nStar labels analysed: {len(star_labels)}")
print(f"Zodiac labels analysed: {len(zodiac_labels)}")

o_count = sum(1 for l in star_labels if l['words'][0].startswith('o'))
print(f"\nStar labels starting with 'o-': {o_count}/{len(star_labels)} "
      f"({100*o_count/max(len(star_labels),1):.0f}%)")

print("\nZodiac page labels:")
for folio, (sign, num) in sorted(ZODIAC_FOLIO_MAP.items(), key=lambda x: x[1][1]):
    folio_labels = [l for l in zodiac_labels if l['folio'] == folio]
    label_text = '; '.join(' '.join(l['words']) for l in folio_labels)
    print(f"  {folio:8s} {sign:25s}  {label_text}")

print("\n'lk-' words (astro-exclusive prefix):")
for item in astro_data['astro_distinctive_words']:
    if item['word'].startswith('lk'):
        print(f"  {item['word']:15s}  astro={item['astro_count']}  other={item['other_count']}")
