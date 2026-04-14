"""
Seven Planets / Seven Metals Analysis
=======================================
In medieval alchemy, seven planets correspond to seven metals:
  Sun (Shams/Kün)       = Gold (dhahab)
  Moon (Qamar/Ay)        = Silver (fidda)
  Mercury (ʿUṭārid)     = Quicksilver (zaybaq)
  Venus (Zuhra/Čolpan)   = Copper (nuḥās)
  Mars (Mirrīkh/Bahrām)  = Iron (ḥadīd)
  Jupiter (Mushtarī/Hurmuzd) = Tin (qaṣdīr)
  Saturn (Zuḥal/Kayvān)  = Lead (raṣāṣ)

If the Voynich contains alchemical recipes timed by planetary position,
we should find:
1. A set of ~7 words that CO-OCCUR across multiple sections (herbal + astro + pharma)
2. Words that appear in formulaic/recipe-like positions
3. Cross-section vocabulary bridging herbal (H) and astronomical (A/Z/S) pages
"""

import json
import math
from collections import Counter, defaultdict

with open('data/transcription/voynich_nlp.json') as f:
    data = json.load(f)

sentences = data['sentences']
metadata = data['metadata']

# Classify folios by illustration type
folio_type = {}
for folio, meta in metadata.items():
    folio_type[folio] = meta.get('illustration', '?')

# Type labels
TYPE_NAMES = {
    'H': 'Herbal', 'A': 'Astro', 'Z': 'Zodiac', 'S': 'Stars',
    'B': 'Bio', 'C': 'Cosmo', 'P': 'Pharma', 'T': 'Text'
}

# Build per-section word frequencies
section_freq = defaultdict(Counter)  # type -> word -> count
section_sents = defaultdict(list)
word_sections = defaultdict(set)     # word -> set of section types
word_total = Counter()

for s in sentences:
    folio = s['folio']
    sec = folio_type.get(folio, '?')
    section_freq[sec].update(s['words'])
    section_sents[sec].append(s)
    for w in s['words']:
        word_sections[w].add(sec)
        word_total[w] += 1

print("Section sizes:")
for sec, freq in sorted(section_freq.items()):
    total = sum(freq.values())
    unique = len(freq)
    name = TYPE_NAMES.get(sec, sec)
    print(f"  {sec} ({name:8s}): {total:6d} tokens, {unique:5d} types")

# =====================================================================
# TEST 1: Words that bridge Herbal + Astronomical sections
# =====================================================================
print("\n" + "="*80)
print("TEST 1: CROSS-SECTION VOCABULARY (Herbal ↔ Astro/Stars/Zodiac)")
print("="*80)

astro_types = {'A', 'Z', 'S', 'C'}
herbal_types = {'H'}
pharma_types = {'P'}

# Words appearing in BOTH herbal and astronomical sections
bridge_words = []
for w, sections in word_sections.items():
    in_herbal = bool(sections & herbal_types)
    in_astro = bool(sections & astro_types)
    in_pharma = bool(sections & pharma_types)
    
    if in_herbal and in_astro:
        h_count = sum(section_freq[s][w] for s in herbal_types)
        a_count = sum(section_freq[s][w] for s in astro_types)
        p_count = sum(section_freq[s][w] for s in pharma_types)
        other = word_total[w] - h_count - a_count - p_count
        bridge_words.append({
            'word': w, 'herbal': h_count, 'astro': a_count,
            'pharma': p_count, 'other': other, 'total': word_total[w],
            'n_sections': len(sections), 'in_pharma': in_pharma,
        })

bridge_words.sort(key=lambda x: -min(x['herbal'], x['astro']))

print(f"\n{len(bridge_words)} words appear in both Herbal AND Astronomical sections")
print(f"\nTop 40 bridge words (sorted by min(herbal, astro) count):")
print(f"{'Word':20s} {'Herbal':>7s} {'Astro':>6s} {'Pharma':>7s} {'Other':>6s} {'Total':>6s} {'Sects':>5s}")
print("-"*70)
for bw in bridge_words[:40]:
    p_flag = " *P*" if bw['in_pharma'] else ""
    print(f"  {bw['word']:18s} {bw['herbal']:7d} {bw['astro']:6d} "
          f"{bw['pharma']:7d} {bw['other']:6d} {bw['total']:6d} {bw['n_sections']:5d}{p_flag}")

# =====================================================================
# TEST 2: Find sets of ~7 words that co-occur in formulaic patterns
# =====================================================================
print("\n" + "="*80)
print("TEST 2: RECURRING WORD GROUPS (looking for sets of ~7)")
print("="*80)

# Strategy: find words that appear together in the same sentence
# significantly more often than chance. Focus on words that appear
# across ALL major section types.

# Words appearing in 4+ section types (ubiquitous vocabulary)
ubiquitous = {w for w, secs in word_sections.items() 
              if len(secs) >= 4 and word_total[w] >= 20}

print(f"\nUbiquitous words (4+ section types, freq≥20): {len(ubiquitous)}")

# Co-occurrence matrix for ubiquitous words
cooccur = defaultdict(Counter)
for s in sentences:
    words_in_sent = set(s['words']) & ubiquitous
    for w1 in words_in_sent:
        for w2 in words_in_sent:
            if w1 < w2:
                cooccur[w1][w2] += 1

# Find cliques: groups of words that frequently co-occur
# Use PMI (pointwise mutual information) to find strong associations
n_sents = len(sentences)
word_sent_count = Counter()
for s in sentences:
    for w in set(s['words']):
        word_sent_count[w] += 1

print(f"\nStrongest co-occurring pairs (ubiquitous words):")
print(f"{'Word 1':18s} {'Word 2':18s} {'Co-occur':>8s} {'PMI':>6s}")
print("-"*55)

pmi_pairs = []
for w1, partners in cooccur.items():
    for w2, count in partners.items():
        if count >= 5:
            p_joint = count / n_sents
            p_w1 = word_sent_count[w1] / n_sents
            p_w2 = word_sent_count[w2] / n_sents
            pmi = math.log2(p_joint / max(p_w1 * p_w2, 1e-10))
            pmi_pairs.append((w1, w2, count, pmi))

pmi_pairs.sort(key=lambda x: -x[3])
for w1, w2, count, pmi in pmi_pairs[:30]:
    print(f"  {w1:18s} {w2:18s} {count:8d} {pmi:6.2f}")

# =====================================================================
# TEST 3: Short, frequent words that could be planet/metal names
# =====================================================================
print("\n" + "="*80)
print("TEST 3: SHORT FREQUENT WORDS (planet/metal name candidates)")
print("="*80)

# Planets have short names: kün (3), ay (2), shams (4), qamar (4)
# Look for short words (2-4 chars) that appear across many sections
short_candidates = []
for w, count in word_total.items():
    if 1 <= len(w) <= 5 and count >= 15:
        secs = word_sections[w]
        h = sum(section_freq[s][w] for s in herbal_types)
        a = sum(section_freq[s][w] for s in astro_types)
        p = sum(section_freq[s][w] for s in pharma_types)
        b = sum(section_freq[s].get(w, 0) for s in ('B',))
        short_candidates.append({
            'word': w, 'len': len(w), 'total': count,
            'sections': len(secs), 'sec_list': sorted(secs),
            'H': h, 'A': a, 'P': p, 'B': b,
        })

short_candidates.sort(key=lambda x: (-x['sections'], -x['total']))

print(f"\nShort words (1-5 chars, freq≥15), sorted by section spread:")
print(f"{'Word':10s} {'Len':>3s} {'Total':>5s} {'Sects':>5s} {'H':>4s} {'A':>4s} "
      f"{'P':>4s} {'B':>4s} Sections")
print("-"*75)
for sc in short_candidates[:50]:
    print(f"  {sc['word']:8s} {sc['len']:3d} {sc['total']:5d} {sc['sections']:5d} "
          f"{sc['H']:4d} {sc['A']:4d} {sc['P']:4d} {sc['B']:4d} "
          f"{','.join(sc['sec_list'])}")

# =====================================================================
# TEST 4: Formulaic position analysis
# =====================================================================
print("\n" + "="*80)
print("TEST 4: RECIPE-LIKE PATTERNS (formulaic sentence structures)")
print("="*80)

# In alchemical recipes, planet names appear at specific positions
# e.g., "When [PLANET] is in [SIGN], take [INGREDIENT]..."
# Look for words that appear in FIXED positions relative to other words

# Check sentence-initial words across sections
initial_by_section = defaultdict(Counter)
for s in sentences:
    sec = folio_type.get(s['folio'], '?')
    if s['words']:
        initial_by_section[sec][s['words'][0]] += 1

# Words that serve as sentence-starters in multiple sections
print("\nSentence-initial words appearing in 3+ section types:")
init_words = defaultdict(lambda: defaultdict(int))
for sec, counts in initial_by_section.items():
    for w, c in counts.items():
        init_words[w][sec] = c

for w, sec_counts in sorted(init_words.items(), 
                             key=lambda x: (-len(x[1]), -sum(x[1].values()))):
    if len(sec_counts) >= 3 and sum(sec_counts.values()) >= 5:
        parts = ', '.join(f"{TYPE_NAMES.get(s,s)}:{c}" for s, c in sorted(sec_counts.items()))
        print(f"  {w:18s}  total={sum(sec_counts.values()):3d}  {parts}")

# =====================================================================
# TEST 5: Groups of exactly 7 that share distributional properties
# =====================================================================
print("\n" + "="*80)
print("TEST 5: CANDIDATE 'SEVEN' GROUPS")
print("="*80)

# Look for groups of words with similar frequency and section distribution
# that could represent the 7 planets or 7 metals

# Cluster short ubiquitous words by their section distribution profile
from itertools import combinations

candidates_7 = []
for sc in short_candidates:
    if sc['sections'] >= 3 and sc['total'] >= 20 and sc['len'] <= 5:
        # Compute section profile
        profile = tuple(
            sum(section_freq[s].get(sc['word'], 0) for s in types)
            for types in [herbal_types, astro_types, pharma_types, {'B'}, {'T'}]
        )
        candidates_7.append((sc['word'], sc['total'], profile, sc['sections']))

print(f"\nCandidates for planet/metal words ({len(candidates_7)} short, ubiquitous words):")
print(f"{'Word':10s} {'Total':>5s} {'H':>5s} {'A/S/Z':>5s} {'P':>5s} {'B':>5s} {'T':>5s}")
print("-"*50)
for w, total, profile, nsec in candidates_7[:30]:
    print(f"  {w:8s} {total:5d} {profile[0]:5d} {profile[1]:5d} "
          f"{profile[2]:5d} {profile[3]:5d} {profile[4]:5d}")

# Now look for actual co-occurrence groups
# Find words that tend to appear in the same sentences
print("\n--- Checking for 'list-like' passages with multiple short words ---")

for sec_type in ['H', 'P', 'S', 'A', 'B']:
    sents_in_sec = section_sents.get(sec_type, [])
    if not sents_in_sec:
        continue
    
    # Find sentences with many short frequent words
    for s in sents_in_sec:
        short_in_sent = [w for w in s['words'] if len(w) <= 5 and word_total.get(w, 0) >= 15]
        unique_short = set(short_in_sent)
        if len(unique_short) >= 5:
            print(f"\n  [{TYPE_NAMES.get(sec_type, sec_type)}] {s['folio']}:{s['unit']} "
                  f"({len(unique_short)} short words):")
            print(f"    Full: {' '.join(s['words'][:30])}")
            print(f"    Short words: {sorted(unique_short)}")

# =====================================================================
# TEST 6: The seven days of the week / planetary hours
# =====================================================================
print("\n" + "="*80)
print("TEST 6: RECURRING PATTERNS OF 7 IN THE TEXT")
print("="*80)

# Look for sequences where the same structural pattern repeats ~7 times
# This could indicate a list of days, planets, or metals

# Check for paragraphs with internal repetition of exactly 7 items
for s in sentences:
    words = s['words']
    if len(words) < 14:
        continue
    
    # Count distinct word types
    wc = Counter(words)
    
    # Look for words that appear exactly 7 times (or close)
    sevens = {w: c for w, c in wc.items() if 6 <= c <= 8 and len(w) >= 2}
    if sevens:
        sec = folio_type.get(s['folio'], '?')
        print(f"\n  [{TYPE_NAMES.get(sec, sec)}] {s['folio']}:{s['unit']} "
              f"— words appearing ~7 times: {sevens}")

# Also check: do any words appear exactly 7 times on individual pages?
print("\nWords appearing exactly 7 times on a single page (across all sentences on that page):")
folio_word_counts = defaultdict(Counter)
for s in sentences:
    folio_word_counts[s['folio']].update(s['words'])

seven_hits = defaultdict(list)
for folio, wc in folio_word_counts.items():
    for w, c in wc.items():
        if c == 7 and len(w) >= 2 and word_total[w] >= 15:
            sec = folio_type.get(folio, '?')
            seven_hits[w].append((folio, sec))

for w, hits in sorted(seven_hits.items(), key=lambda x: -len(x[1])):
    if len(hits) >= 2:
        folios = ', '.join(f"{f}({TYPE_NAMES.get(s,s)})" for f, s in hits[:5])
        print(f"  {w:15s}  appears 7× on {len(hits)} pages: {folios}")

