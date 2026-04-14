"""
Voynich Declension Analysis
============================
Test whether Voynichese shows evidence of a declined (case-marking) language,
analysed separately for Currier Language A and Language B.

Key diagnostics:
1. Word-order freedom (high = declined language signal)
2. Suffix paradigms (same stem, multiple endings)
3. Concordance / agreement between adjacent words
4. Vowel-doubling as inflectional morphology
"""

import json
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

# Load pre-built data
with open('data/transcription/voynich_nlp.json') as f:
    data = json.load(f)

sentences_raw = data['sentences']
metadata = data['metadata']

# Split sentences by Currier language
lang_a_sents = []
lang_b_sents = []
for s in sentences_raw:
    folio = s['folio']
    lang = metadata.get(folio, {}).get('language', '?')
    if lang == 'A':
        lang_a_sents.append(s)
    elif lang == 'B':
        lang_b_sents.append(s)


def analyse_language(sentences, label):
    """Full morphological analysis of one Currier language."""
    print(f"\n{'#'*80}")
    print(f"# CURRIER LANGUAGE {label}")
    print(f"# {len(sentences)} paragraph sentences")
    print(f"{'#'*80}")

    # ------------------------------------------------------------------
    # 1. WORD ORDER FREEDOM
    # ------------------------------------------------------------------
    # For each frequent word, measure how freely it moves around in sentences.
    # Declined languages: most words have HIGH positional entropy.
    # Fixed-order languages: function words have LOW entropy.

    word_positions = defaultdict(list)  # word -> list of normalised positions
    freq = Counter()

    for s in sentences:
        words = s['words']
        n = len(words)
        if n < 3:
            continue
        for i, w in enumerate(words):
            freq[w] += 1
            word_positions[w].append(i / (n - 1))

    total_tokens = sum(freq.values())
    unique_types = len(freq)

    print(f"\n--- 1. WORD ORDER FREEDOM ---")
    print(f"Tokens: {total_tokens}  Types: {unique_types}  TTR: {unique_types/max(total_tokens,1):.3f}")

    # Compute positional entropy for frequent words
    entropies = {}
    for word, positions in word_positions.items():
        if len(positions) < 8:
            continue
        bins = [0] * 5
        for p in positions:
            b = min(4, int(p * 5))
            bins[b] += 1
        total = len(positions)
        entropy = 0.0
        for count in bins:
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        entropies[word] = round(entropy, 3)

    # Distribution of entropies
    if entropies:
        all_ent = list(entropies.values())
        mean_ent = sum(all_ent) / len(all_ent)
        low_ent = sum(1 for e in all_ent if e < 1.5)
        high_ent = sum(1 for e in all_ent if e > 2.0)
        max_possible = math.log2(5)  # 2.32 for 5 uniform bins

        print(f"Positional entropy of {len(entropies)} frequent words (min_freq=8):")
        print(f"  Mean entropy: {mean_ent:.3f} / {max_possible:.3f} max")
        print(f"  Words with LOW entropy (<1.5): {low_ent} ({100*low_ent/len(entropies):.0f}%)")
        print(f"  Words with HIGH entropy (>2.0): {high_ent} ({100*high_ent/len(entropies):.0f}%)")
        print(f"  → {'FREE word order (declension signal)' if mean_ent > 2.0 else 'MODERATE order freedom' if mean_ent > 1.8 else 'CONSTRAINED word order'}")

    # ------------------------------------------------------------------
    # 2. SUFFIX PARADIGMS
    # ------------------------------------------------------------------
    # Extract potential stems and endings. If declined, the same stem
    # should appear with multiple systematic endings.

    print(f"\n--- 2. SUFFIX PARADIGM DETECTION ---")

    # Strategy: for words >= 4 chars, try splitting at each position
    # from char 2 onwards. Look for stems that combine with many endings.
    stem_endings = defaultdict(Counter)  # stem -> {ending: count}
    for w, c in freq.items():
        if len(w) < 4:
            continue
        # Try stems of length 2..len-1
        for split in range(2, len(w)):
            stem = w[:split]
            ending = w[split:]
            if len(ending) >= 1:
                stem_endings[stem][ending] += c

    # Find stems with many distinct endings (paradigm candidates)
    paradigms = []
    for stem, endings in stem_endings.items():
        distinct = len(endings)
        if distinct >= 4 and sum(endings.values()) >= 20:
            paradigms.append((stem, distinct, sum(endings.values()), endings))

    paradigms.sort(key=lambda x: -x[1])

    print(f"Stems with 4+ distinct endings and 20+ total occurrences: {len(paradigms)}")
    print(f"\nTop 25 paradigm candidates:")
    for stem, n_endings, total, endings in paradigms[:25]:
        top5 = endings.most_common(5)
        endings_str = ', '.join(f'-{e}({c})' for e, c in top5)
        print(f"  {stem:12s}  {n_endings:2d} endings  [{total:4d}]  {endings_str}")

    # ------------------------------------------------------------------
    # 3. VOWEL DOUBLING AS INFLECTION
    # ------------------------------------------------------------------
    print(f"\n--- 3. VOWEL DOUBLING AS INFLECTION ---")

    # For each word with doubled vowels, check if the doubled and
    # undoubled forms have DIFFERENT syntactic contexts (preceding word classes).
    # If doubling = inflection, the context should differ systematically.

    # Build "word classes" by clustering words with similar distributions
    # (simplified: just use the word itself as its class for now)

    # Instead: check if doubled forms appear after DIFFERENT words than undoubled
    doubling_pairs = [
        ('dain', 'daiin'), ('ain', 'aiin'), ('okain', 'okaiin'),
        ('otain', 'otaiin'), ('sain', 'saiin'), ('kain', 'kaiin'),
        ('chedy', 'cheedy'), ('shedy', 'sheedy'), ('chey', 'cheey'),
        ('shey', 'sheey'), ('qokedy', 'qokeedy'), ('otedy', 'oteedy'),
        ('okey', 'okeey'),
    ]

    # For each pair, compute preceding-word distribution divergence (KL-like)
    print(f"{'Pair':25s} {'n_short':>7s} {'n_long':>6s} {'Prev overlap':>13s} {'Next overlap':>13s} {'Verdict':>10s}")
    print("-" * 85)

    for short, long in doubling_pairs:
        short_prev = Counter()
        long_prev = Counter()
        short_next = Counter()
        long_next = Counter()

        for s in sentences:
            words = s['words']
            for i, w in enumerate(words):
                if w == short:
                    if i > 0: short_prev[words[i-1]] += 1
                    if i < len(words)-1: short_next[words[i+1]] += 1
                elif w == long:
                    if i > 0: long_prev[words[i-1]] += 1
                    if i < len(words)-1: long_next[words[i+1]] += 1

        n_short = freq.get(short, 0)
        n_long = freq.get(long, 0)

        if not short_prev or not long_prev:
            continue

        # Jaccard on top-10 predecessors
        sp = set(w for w, _ in short_prev.most_common(10))
        lp = set(w for w, _ in long_prev.most_common(10))
        prev_jac = len(sp & lp) / max(len(sp | lp), 1)

        sn = set(w for w, _ in short_next.most_common(10))
        ln = set(w for w, _ in long_next.most_common(10))
        next_jac = len(sn & ln) / max(len(sn | ln), 1)

        # Verdict
        if prev_jac < 0.2 and next_jac < 0.2:
            verdict = "INFLECT"
        elif prev_jac < 0.3 or next_jac < 0.3:
            verdict = "likely"
        else:
            verdict = "unclear"

        print(f"  {short:10s}/{long:10s} {n_short:7d} {n_long:6d} {prev_jac:13.2f} {next_jac:13.2f} {verdict:>10s}")

    # ------------------------------------------------------------------
    # 4. AGREEMENT PATTERNS (adjacent-word suffix concordance)
    # ------------------------------------------------------------------
    print(f"\n--- 4. SUFFIX AGREEMENT (adjacent words sharing endings) ---")

    # In declined languages, adjacent words often share the same case ending
    # (adjective-noun agreement). Test: do adjacent words share endings
    # more often than chance?

    def get_ending(word, n_chars=2):
        return word[-n_chars:] if len(word) >= n_chars else word

    # Observed vs expected agreement
    same_ending_count = 0
    total_pairs = 0
    ending_freq = Counter()

    for s in sentences:
        words = s['words']
        for i in range(len(words) - 1):
            if len(words[i]) >= 3 and len(words[i+1]) >= 3:
                e1 = get_ending(words[i])
                e2 = get_ending(words[i+1])
                ending_freq[e1] += 1
                ending_freq[e2] += 1
                total_pairs += 1
                if e1 == e2:
                    same_ending_count += 1

    if total_pairs > 0:
        observed_rate = same_ending_count / total_pairs
        # Expected by chance: sum of p(ending)^2
        total_endings = sum(ending_freq.values())
        expected_rate = sum((c/total_endings)**2 for c in ending_freq.values())

        ratio = observed_rate / max(expected_rate, 0.001)
        print(f"Adjacent words sharing final 2 chars:")
        print(f"  Observed: {observed_rate:.4f} ({same_ending_count}/{total_pairs})")
        print(f"  Expected (chance): {expected_rate:.4f}")
        print(f"  Ratio (obs/exp): {ratio:.2f}")
        print(f"  → {'AGREEMENT detected (declension signal)' if ratio > 1.3 else 'No significant agreement' if ratio < 1.1 else 'Weak agreement signal'}")

    # Also test with 3-char endings
    same3 = 0
    total3 = 0
    ending3_freq = Counter()
    for s in sentences:
        words = s['words']
        for i in range(len(words) - 1):
            if len(words[i]) >= 4 and len(words[i+1]) >= 4:
                e1 = get_ending(words[i], 3)
                e2 = get_ending(words[i+1], 3)
                ending3_freq[e1] += 1
                ending3_freq[e2] += 1
                total3 += 1
                if e1 == e2:
                    same3 += 1

    if total3 > 0:
        obs3 = same3 / total3
        total3_endings = sum(ending3_freq.values())
        exp3 = sum((c/total3_endings)**2 for c in ending3_freq.values())
        ratio3 = obs3 / max(exp3, 0.001)
        print(f"\n  3-char ending agreement:")
        print(f"  Observed: {obs3:.4f}  Expected: {exp3:.4f}  Ratio: {ratio3:.2f}")

    # ------------------------------------------------------------------
    # 5. WORD-ORDER FREEDOM TEST: sentence permutation consistency
    # ------------------------------------------------------------------
    print(f"\n--- 5. BIGRAM PREDICTABILITY (low = free order) ---")

    # In rigid-order languages, knowing word[i] strongly predicts word[i+1].
    # In free-order languages, bigram entropy is high.

    bigram_count = Counter()
    unigram_count = Counter()

    for s in sentences:
        words = s['words']
        for i in range(len(words)):
            unigram_count[words[i]] += 1
            if i < len(words) - 1:
                bigram_count[(words[i], words[i+1])] += 1

    # Compute conditional entropy H(w2|w1)
    # H(w2|w1) = -sum over all bigrams p(w1,w2) * log2(p(w2|w1))
    total_bigrams = sum(bigram_count.values())
    cond_entropy = 0.0
    for (w1, w2), count in bigram_count.items():
        p_bigram = count / total_bigrams
        p_cond = count / unigram_count[w1]
        cond_entropy -= p_bigram * math.log2(p_cond)

    unigram_entropy = 0.0
    total_unigrams = sum(unigram_count.values())
    for w, c in unigram_count.items():
        p = c / total_unigrams
        unigram_entropy -= p * math.log2(p)

    predictability = 1 - (cond_entropy / max(unigram_entropy, 0.001))
    print(f"  Unigram entropy H(w):    {unigram_entropy:.3f} bits")
    print(f"  Conditional entropy H(w2|w1): {cond_entropy:.3f} bits")
    print(f"  Predictability: {predictability:.3f}")
    print(f"  → {'RIGID word order' if predictability > 0.4 else 'FREE word order (declension-compatible)' if predictability < 0.25 else 'MODERATE order constraint'}")

    # ------------------------------------------------------------------
    # 6. ENDING DISTRIBUTION BY SENTENCE POSITION
    # ------------------------------------------------------------------
    print(f"\n--- 6. CASE-LIKE ENDING DISTRIBUTION BY POSITION ---")
    print("(In declined languages, certain endings cluster in specific syntactic slots)")

    # For the most common 2-char endings, show positional distribution
    all_endings = Counter()
    ending_positions = defaultdict(lambda: [0]*5)

    for s in sentences:
        words = s['words']
        n = len(words)
        if n < 3:
            continue
        for i, w in enumerate(words):
            if len(w) < 3:
                continue
            e = get_ending(w)
            all_endings[e] += 1
            bucket = min(4, int((i / (n-1)) * 5))
            ending_positions[e][bucket] += 1

    print(f"{'Ending':>8s} {'Count':>6s}  {'Init':>5s} {'Ear':>5s} {'Mid':>5s} {'Late':>5s} {'Fin':>5s}  Chi2-like")
    print("-" * 70)
    for ending, count in all_endings.most_common(20):
        buckets = ending_positions[ending]
        total = sum(buckets)
        expected = total / 5
        # Chi-squared-like measure of non-uniformity
        chi2 = sum((b - expected)**2 / max(expected, 1) for b in buckets)
        pcts = [f"{100*b/total:4.0f}%" for b in buckets]
        star = " ***" if chi2 > 15 else " **" if chi2 > 8 else " *" if chi2 > 4 else ""
        print(f"  {ending:>6s} {count:6d}  {' '.join(pcts)}  {chi2:6.1f}{star}")


# Run both languages
analyse_language(lang_a_sents, "A")
analyse_language(lang_b_sents, "B")

# ------------------------------------------------------------------
# CROSS-LANGUAGE COMPARISON
# ------------------------------------------------------------------
print(f"\n{'#'*80}")
print("# CROSS-LANGUAGE COMPARISON")
print(f"{'#'*80}")

# Compare the stem paradigms across A and B
print("\nShared stems with different ending distributions:")

freq_a = Counter()
freq_b = Counter()
for s in lang_a_sents:
    freq_a.update(s['words'])
for s in lang_b_sents:
    freq_b.update(s['words'])

# Find words that appear in both but with different frequency ratios
# This reveals morphological differences between the "dialects"
stem_a = defaultdict(Counter)
stem_b = defaultdict(Counter)

for w, c in freq_a.items():
    if len(w) >= 4:
        for split in range(max(2, len(w)-4), len(w)):
            stem = w[:split]
            ending = w[split:]
            if 1 <= len(ending) <= 4:
                stem_a[stem][ending] += c

for w, c in freq_b.items():
    if len(w) >= 4:
        for split in range(max(2, len(w)-4), len(w)):
            stem = w[:split]
            ending = w[split:]
            if 1 <= len(ending) <= 4:
                stem_b[stem][ending] += c

# Find stems shared between A and B with significantly different ending profiles
print(f"\n{'Stem':12s} {'Lang':>4s}  Top endings")
print("-" * 80)

shared_stems = set(stem_a) & set(stem_b)
divergent = []
for stem in shared_stems:
    ea = stem_a[stem]
    eb = stem_b[stem]
    if sum(ea.values()) < 15 or sum(eb.values()) < 15:
        continue
    # Compute ending distribution divergence
    all_endings_set = set(ea) | set(eb)
    total_a = sum(ea.values())
    total_b = sum(eb.values())
    divergence = 0
    for e in all_endings_set:
        pa = ea[e] / total_a if e in ea else 0
        pb = eb[e] / total_b if e in eb else 0
        if pa > 0 and pb > 0:
            divergence += pa * math.log2(pa / pb)
        elif pa > 0:
            divergence += pa * 3  # penalty for missing in B
    divergent.append((stem, divergence, ea, eb))

divergent.sort(key=lambda x: -x[1])

for stem, div, ea, eb in divergent[:15]:
    a_str = ', '.join(f'-{e}({c})' for e, c in ea.most_common(5))
    b_str = ', '.join(f'-{e}({c})' for e, c in eb.most_common(5))
    print(f"  {stem:12s}   A:  {a_str}")
    print(f"  {'':12s}   B:  {b_str}")
    print(f"  {'':12s}   divergence={div:.2f}")
    print()

