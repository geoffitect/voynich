"""
Voynich Manuscript — Preliminary Lexicon
==========================================
Compiled from distributional analysis, visual cross-referencing,
structural tokenization, and declension analysis.

Confidence levels:
  A = Strong structural/distributional evidence
  B = Moderate evidence from multiple analyses
  C = Tentative, single-source evidence
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# Load all data
with open('data/transcription/voynich_nlp.json') as f:
    nlp = json.load(f)
with open('data/lexicon/label_network.json') as f:
    net = json.load(f)
with open('data/analysis/plant_analysis.json') as f:
    plants = json.load(f)

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

# Word frequency
freq = Counter()
section_freq = defaultdict(Counter)
for s in sentences:
    sec = folio_type.get(s['folio'], '?')
    freq.update(s['words'])
    section_freq[sec].update(s['words'])

# ---------------------------------------------------------------------------
# Build the lexicon
# ---------------------------------------------------------------------------

LEXICON = []

def add(eva, category, meaning, confidence, evidence, paradigm=None,
        section_profile=None, see_also=None):
    LEXICON.append({
        'eva': eva,
        'glyph': g(eva),
        'category': category,
        'meaning': meaning,
        'confidence': confidence,
        'evidence': evidence,
        'paradigm': paradigm or [],
        'frequency': freq.get(eva, 0),
        'section_profile': section_profile or '',
        'see_also': see_also or [],
    })

# =========================================================================
# GRAMMATICAL ELEMENTS
# =========================================================================

add('daiin', 'function word', 'Most common word; likely preposition, copula, or demonstrative '
    '("this", "the", "is", "in")', 'A',
    'Highest frequency (829). Appears in all sections. Head of the da- paradigm with 7+ '
    'declined forms. Dominates herbal section (54%) suggesting it introduces/connects descriptions.',
    paradigm=['daiin', 'dain', 'dair', 'dar', 'dal', 'dam', 'dan'],
    section_profile='H:54% S:16% P:13% B:11%')

add('ol', 'function word', 'Common function word; possibly "water", "of", or a case marker', 'B',
    'Second most common (534). Appears as prefix in water-page compounds (olkain, olshey, orol). '
    'Bio/bath section enriched (48%). Forms word-initial element in pool/water vocabulary.',
    paradigm=['ol', 'or', 'al', 'ar', 'am'],
    section_profile='B:48% S:21% H:15%',
    see_also=['or', 'ar', 'al'])

add('or', 'function word', 'Declined form of ol/or/ar/al paradigm', 'A',
    'Vector analogy ol:or::al:ar confirmed (cosine 0.72). Same paradigm as ol.',
    paradigm=['ol', 'or', 'al', 'ar', 'am'])

add('ar', 'function word', 'Declined form of ol/or/ar/al paradigm', 'A',
    'Stars-heavy (48%). Vector analogy confirmed.',
    paradigm=['ol', 'or', 'al', 'ar', 'am'],
    section_profile='S:48%')

add('al', 'function word', 'Declined form of ol/or/ar/al paradigm', 'A',
    'Stars-heavy (63%). Strongly enriched on astronomical pages.',
    paradigm=['ol', 'or', 'al', 'ar', 'am'],
    section_profile='S:63%')

add('qo-', 'prefix/determiner', 'Word-initial prefix; determiner, relative pronoun, or definite article', 'A',
    'Token QO has positional entropy 0.098 (virtually always word-initial). 97.6% q+o co-occurrence. '
    'Attaches to stems from all domains (qokedy=Bio, qoor=Pharma, qokedaiin=Stars). '
    'Forms the most productive prefix paradigm in the manuscript.',
    section_profile='all sections')

add('o-', 'prefix', 'Word-initial element on labels; possibly Arabic definite article al- or '
    'a general determiner', 'B',
    '67% of star labels and 90%+ of zodiac labels begin with o-. Arabic star names universally '
    'begin with al-. May be the same element as standalone "ol/or/al/ar" in declined form.',
    section_profile='labels: Astro, Zodiac, Bio, Pharma')

add('dy', 'suffix/particle', 'Common word-final element; possibly locative case marker ("in/at") '
    'or verbal suffix', 'B',
    'Appears in all sections. Sentence-initial clustering in Language B (chi2=75.3). '
    'As suffix: -dy/-edy/-eedy form the dominant Language B ending pattern.',
    section_profile='H:53% B:24%')

add('s', 'particle', 'Short function word; possibly conjunction ("and") or sentence particle', 'B',
    'Appears in all 7 section types. High frequency (191). Herbal-enriched (67%).',
    section_profile='H:67%')

add('y', 'particle/suffix', 'Common word-final token; vowel or grammatical suffix', 'A',
    'Token Y has mean position 0.879 (strongly word-final). 11.9% of all tokens.',
    section_profile='all sections')

# =========================================================================
# INFLECTIONAL MORPHOLOGY
# =========================================================================

add('-aiin/-ain', 'suffix', 'Case/declension ending; short vs long vowel marks grammatical '
    'distinction (possibly nominative vs accusative, or definite vs indefinite)', 'A',
    'AII+N has positional entropy 0.217 (virtually always word-final). PMI 4.89. '
    'daiin/dain differ in collocation (Jaccard <0.2 in Lang A). 4.64x within-family '
    'vector similarity vs random baseline.',
    paradigm=['daiin/dain', 'okaiin/okain', 'aiin/ain', 'saiin/sain', 'otaiin/otain'])

add('-edy/-eedy', 'suffix', 'Case/declension ending; Language B dominant suffix. '
    'Vowel length (ee vs e) marks grammatical distinction', 'A',
    'chedy/cheedy, shedy/sheedy, qokedy/qokeedy all show different collocations. '
    '-edy is the signature morpheme of Currier Language B.',
    paradigm=['chedy/cheedy', 'shedy/sheedy', 'qokedy/qokeedy', 'otedy/oteedy'])

add('-ey/-eey', 'suffix', 'Case/declension ending; related to -edy system', 'A',
    'chey/cheey, shey/sheey show systematic paradigmatic behavior.',
    paradigm=['chey/cheey', 'shey/sheey', 'qokey/qokeey', 'okey/okeey'])

add('-ol/-or/-al/-ar/-am', 'suffix set', 'Five-way case ending system on stems; '
    'the core declension paradigm', 'A',
    'Vector analogy ol:or::al:ar confirmed. Five endings attach to all major stems: '
    'ch-ol/-or, sh-ol/-or, ok-al/-ar/-am, ot-al/-ar. Suffix agreement ratio 1.70x (Lang A).',
    paradigm=['chol/chor', 'shol/shor', 'okal/okar/okam', 'otal/otar'])

# =========================================================================
# CONTENT WORDS — ASTRONOMICAL
# =========================================================================

add('lk-', 'stem (astronomical)', 'Astronomical domain marker; possibly "star", "constellation", '
    'or "celestial body"', 'A',
    'Almost exclusively on Stars section pages (lkaiin: 45 astro / 5 other). Takes full '
    'declension endings (-aiin, -eey, -am, -al, -chdy). Too frequent for a proper noun; '
    'likely a common noun.',
    paradigm=['lkaiin', 'lkeey', 'lkeeey', 'lkam', 'lkal', 'lkchdy', 'lkshedy'],
    section_profile='Stars: 90%+')

add('otol', 'label (cross-ref)', 'Label appearing across Astro/Bio/Pharma/Zodiac; '
    'a key cross-reference term linking star diagrams to bath procedures to recipes', 'A',
    'The most connected label: f68r1 (star label), f71r (zodiac), f77r+f77v (bath tubes), '
    'f102v2 (pharma fragment). Four-way section bridge.',
    see_also=['otoldy', 'otoly'])

add('otoldy', 'label (cross-ref)', 'Declined form of otol; labels bath tube (f82v) and '
    'pharma fragments (f89r1, f89r2, f99r, f99v)', 'A',
    'Strongest Bio↔Pharma bridge (5 folios). Appears as both container label and fragment label.',
    see_also=['otol', 'otoly'])

add('otaly', 'label (plant name?)', 'Name of a brown multi-fingered root plant; visually confirmed '
    'across pharma pages. Cross-references zodiac (Scorpio?) and bath section.', 'A',
    'Visual verification: f88r and f99v both show brown multi-fingered root fragments. '
    'f73r labels nymph on Scorpio zodiac page. f84r labels bath flow. Complete pharma reference chain: '
    'zodiac timing → bath procedure → recipe ingredient.',
    see_also=['otal', 'otaldy', 'otalchy'])

add('am', 'label modifier', 'Appears as second word in compound labels (okain am, otor am, sar am); '
    'possibly a unit marker, quantity, or temporal modifier', 'B',
    'Bridges Astro/Pharma/Zodiac. On zodiac pages appears with sign labels (otalchy tar am dy). '
    'On pharma pages modifies fragment labels.',
    section_profile='Astro, Zodiac, Pharma labels')

# =========================================================================
# CONTENT WORDS — BOTANICAL / DESCRIPTIVE
# =========================================================================

add('ch-', 'stem', 'Major content stem; takes all declension endings. Herbal and Bio sections. '
    'Possibly a general botanical term or common noun', 'B',
    'Second largest paradigm (585 endings, 5004 occurrences). CHol, CHor, CHey, CHedy are '
    'among the most common words.',
    paradigm=['chol', 'chor', 'chey', 'cheey', 'chedy', 'cheedy', 'cheol', 'cheor'])

add('sh-', 'stem', 'Major content stem parallel to ch-; same endings. '
    'Bio section enriched (60% for shedy)', 'B',
    'Third largest paradigm. SH- and CH- behave as parallel stems (vector analogy chol:chor::shol:shor '
    'confirmed). May be two related nouns or a grammatical alternation.',
    paradigm=['shol', 'shor', 'shey', 'sheey', 'shedy', 'sheedy', 'sheol'])

add('cthy', 'content word', 'Herbal-exclusive word (88% herbal). Likely describes a plant '
    'property or processing step', 'B',
    'Appears on 44 herbal folios. Nearest neighbors: daiin, chor, tchol, qotchy, chol — '
    'all herbal vocabulary.',
    section_profile='H:88%')

add('qol', 'content word', 'Bio/bath section word (78% bio). Likely relates to bath/water '
    'procedure or nymph-associated concept', 'B',
    'Strongly enriched on pool/nymph pages. Nearest neighbors: qokain, chedy, ly, oly — '
    'all bio-section words.',
    section_profile='B:78%')

# =========================================================================
# CONTENT WORDS — PHARMACEUTICAL
# =========================================================================

add('cheol', 'content word', 'Pharma-enriched word (24%). Possibly names a container type, '
    'preparation method, or ingredient category', 'B',
    'Appears on jar-illustration pages. Related to cheor (also pharma-enriched). '
    'The -eol ending may mark pharma-specific vocabulary.',
    section_profile='P:24%, H:19%',
    see_also=['cheor', 'okeol', 'sheol'])

add('okeol', 'content word', 'Strongly pharma-associated (52%). Likely names a specific '
    'preparation or container contents', 'B',
    'Nearest neighbor: qockheol (0.77). The -eol ending cluster is pharma-specific.',
    section_profile='P:52%',
    see_also=['cheol', 'qokeol', 'okeeol'])

# =========================================================================
# ADJECTIVE CANDIDATES (from plant feature analysis)
# =========================================================================

add('okchol', 'adjective?', 'Enriched exclusively on blue-flower plant pages; '
    'candidate for color term "blue"', 'C',
    'Log-odds 4.73 for blue_flowers feature. Appears only on folios with blue-flowered plants.',
    section_profile='H (blue flower pages)')

add('key', 'adjective?', 'Enriched exclusively on blue-flower plant pages; '
    'short word, strong color term candidate for "blue"', 'C',
    'Log-odds 4.31 for blue_flowers feature. 3 occurrences, all on blue-flower pages.',
    section_profile='H (blue flower pages)')

add('okedy', 'adjective?', 'Enriched on brown-root plant pages; candidate for "brown" or "dark"', 'C',
    'Log-odds 3.63 for brown_root feature.',
    section_profile='H (brown root pages)')

add('chkar', 'adjective?', 'Enriched exclusively on flat-topped-root pages; '
    'candidate for "flat" or "truncated"', 'C',
    'Log-odds 5.29 for flat_topped_root. 4 occurrences, all on flat-root pages.',
    section_profile='H (flat root pages)')

add('shed', 'adjective?', 'Enriched on calyx-type flower pages; candidate for "cup-shaped" '
    'or descriptor of calyx structure', 'C',
    'Log-odds 5.08 for calyx feature.',
    section_profile='H (calyx pages)')

add('yky', 'adjective?', 'Enriched on seed-bearing plant pages; candidate for "seed" or "grain"', 'C',
    'Log-odds 3.46 for seeds feature.',
    section_profile='H (seed pages)')

add('okol', 'adjective?', 'Enriched on single-flower plant pages; candidate for "single" or "one"', 'C',
    'Log-odds 3.42 for single_flower feature. Also appears as zodiac/pharma label.',
    section_profile='H (single flower), Z, P')

add('shee', 'adjective?', 'Enriched on root-platform plant pages; candidate for '
    '"platform", "base", or "earth"', 'C',
    'Log-odds 6.34 for root_platform feature. 4 occurrences, all on platform-root pages.',
    section_profile='H (root platform pages)')

# =========================================================================
# SECTION-EXCLUSIVE VOCABULARY
# =========================================================================

add('qolchedy', 'bath vocabulary', 'Bio/bath exclusive (100%). Labels or describes '
    'something specific to the bath/nymph illustrations', 'B',
    '11 occurrences, all on Bio pages. Never in herbal or stars.',
    section_profile='B:100%')

add('oroly', 'bath vocabulary', 'Bath/water exclusive (100%). Contains ol- prefix. '
    'Likely water/pool-related', 'B',
    '6 occurrences, all on pool/water pages.',
    section_profile='B:100% (water pages)')

add('ctho', 'herbal vocabulary', 'Herbal exclusive (100%). Likely describes a plant '
    'property, processing step, or part name', 'B',
    '16 occurrences, all herbal. Never in other sections.',
    section_profile='H:100%')

add('shedain', 'stellar vocabulary', 'Stars exclusive (100%). Likely a stellar/astronomical term', 'B',
    '10 occurrences, all on Stars pages.',
    section_profile='S:100%')

# =========================================================================
# STRUCTURAL / WRITING SYSTEM
# =========================================================================

add('c (EVA)', 'structural', 'NOT a phoneme. Structural left-bracket of the ch ligature and '
    'gallows cartouche. 99.8% followed by h/t/k/p/f.', 'A',
    'Only 29/12645 instances of c not followed by h/t/k/p/f. Functions as glyph connector.')

add('ch (EVA)', 'glyph unit', 'Single consonant unit (ligated cc). The most common consonant '
    'in the script.', 'A',
    '10412 occurrences. Takes gallows modifiers: cth, ckh, cph, cfh. '
    'Position: mean 0.211 (word-initial tendency).')

add('gallows (t,k,p,f)', 'modifiers', 'Not standalone consonants but modifiers of adjacent characters. '
    'Create the cartouche construct c-gallows-h. May encode aspiration, emphasis, or '
    'phonological features.', 'B',
    '63% of t preceded by o. 48% of p followed by c (→pch compound). '
    'True consonant inventory is ~8 base consonants with modifier variants.')

# =========================================================================
# Generate output
# =========================================================================

# Sort by confidence then frequency
LEXICON.sort(key=lambda x: ('ABC'.index(x['confidence']), -x['frequency']))

# Print
print("=" * 100)
print("VOYNICH MANUSCRIPT — PRELIMINARY LEXICON")
print("=" * 100)
print(f"\nTotal entries: {len(LEXICON)}")
print(f"Confidence A (strong): {sum(1 for x in LEXICON if x['confidence'] == 'A')}")
print(f"Confidence B (moderate): {sum(1 for x in LEXICON if x['confidence'] == 'B')}")
print(f"Confidence C (tentative): {sum(1 for x in LEXICON if x['confidence'] == 'C')}")

for entry in LEXICON:
    print(f"\n{'━' * 100}")
    print(f"  {entry['eva']:20s}  {entry['glyph']}  "
          f"[{entry['confidence']}] {entry['category']}")
    print(f"  Meaning: {entry['meaning']}")
    print(f"  Freq: {entry['frequency']}  {entry['section_profile']}")
    print(f"  Evidence: {entry['evidence']}")
    if entry['paradigm']:
        print(f"  Paradigm: {', '.join(entry['paradigm'])}")
    if entry['see_also']:
        print(f"  See also: {', '.join(entry['see_also'])}")

# JSON export
export = {
    'title': 'Voynich Manuscript Preliminary Lexicon',
    'date': '2026-04-13',
    'methodology': (
        'Compiled from: (1) distributional word vectors (PPMI+SVD, 50d), '
        '(2) visual cross-referencing against illustrated content, '
        '(3) structural tokenization removing Latin alphabetic bias, '
        '(4) declension/paradigm analysis, '
        '(5) plant feature enrichment analysis, '
        '(6) label network cross-referencing across sections. '
        'Phonetic values are NOT assigned; all analysis is structural/distributional.'
    ),
    'entries': LEXICON,
    'statistics': {
        'total_entries': len(LEXICON),
        'confidence_A': sum(1 for x in LEXICON if x['confidence'] == 'A'),
        'confidence_B': sum(1 for x in LEXICON if x['confidence'] == 'B'),
        'confidence_C': sum(1 for x in LEXICON if x['confidence'] == 'C'),
    },
}

with open('data/lexicon/lexicon.json', 'w') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)
print(f"\n\nExported to lexicon.json")

# HTML
html = ['<!DOCTYPE html><html><head><meta charset="UTF-8">',
    '<title>Voynich Preliminary Lexicon</title>',
    "<style>@font-face{font-family:'VoynichEVA';src:url('fonts/Voynich/VoynichEVA.ttf')format('truetype')}",
    "body{font-family:'Segoe UI',system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9;line-height:1.6}",
    "h1{color:#58a6ff;border-bottom:2px solid #1f6feb}h2{color:#79c0ff;margin-top:30px}",
    ".v{font-family:'VoynichEVA',serif;font-size:1.8em;color:#ffa657;letter-spacing:3px}",
    ".eva{font-family:monospace;background:#161b22;padding:2px 8px;border-radius:3px;color:#7ee787;font-size:1.1em}",
    "table{border-collapse:collapse;margin:10px 0;background:#161b22;width:100%}",
    "th,td{border:1px solid #30363d;padding:10px 12px;text-align:left;vertical-align:top}",
    "th{background:#21262d;color:#58a6ff}",
    "tr:hover{background:#1c2128}",
    ".conf-A{color:#7ee787;font-weight:bold}.conf-B{color:#d29922}.conf-C{color:#8b949e}",
    ".note{background:#1c2128;border-left:4px solid #1f6feb;padding:12px 15px;margin:15px 0}",
    ".paradigm{font-family:monospace;color:#d2a8ff;font-size:0.9em}",
    "</style></head><body>",
    "<h1>Voynich Manuscript — Preliminary Lexicon</h1>",
    "<div class='note'>",
    f"<strong>{len(LEXICON)} entries</strong> compiled from distributional analysis, ",
    "visual cross-referencing, structural tokenization, and declension analysis. ",
    "No phonetic values assigned — all analysis is structural/distributional. ",
    f"<br>Confidence: <span class='conf-A'>A (strong)</span> = {sum(1 for x in LEXICON if x['confidence']=='A')}, ",
    f"<span class='conf-B'>B (moderate)</span> = {sum(1 for x in LEXICON if x['confidence']=='B')}, ",
    f"<span class='conf-C'>C (tentative)</span> = {sum(1 for x in LEXICON if x['confidence']=='C')}",
    "</div>",
]

for cat in ['function word', 'prefix/determiner', 'prefix', 'suffix/particle', 'particle',
            'particle/suffix', 'suffix', 'suffix set',
            'stem (astronomical)', 'label (cross-ref)', 'label (plant name?)', 'label modifier',
            'stem', 'content word',
            'bath vocabulary', 'herbal vocabulary', 'stellar vocabulary',
            'adjective?',
            'structural', 'glyph unit', 'modifiers']:
    entries = [e for e in LEXICON if e['category'] == cat]
    if not entries:
        continue
    html.append(f"<h2>{cat.title()}</h2>")
    html.append("<table><tr><th>Glyph</th><th>EVA</th><th>Conf</th>"
               "<th>Meaning</th><th>Evidence</th></tr>")
    for e in entries:
        conf_css = f"conf-{e['confidence']}"
        paradigm_html = (f"<br><span class='paradigm'>Paradigm: {', '.join(e['paradigm'])}</span>"
                        if e['paradigm'] else '')
        html.append(
            f"<tr><td><span class='v'>{e['glyph']}</span></td>"
            f"<td><span class='eva'>{e['eva']}</span></td>"
            f"<td class='{conf_css}'>{e['confidence']}</td>"
            f"<td>{e['meaning']}{paradigm_html}</td>"
            f"<td>{e['evidence'][:200]}{'...' if len(e['evidence'])>200 else ''}</td></tr>")
    html.append("</table>")

html.append("</body></html>")
Path('reports/html/lexicon.html').write_text('\n'.join(html), encoding='utf-8')
print("HTML report: lexicon.html")
