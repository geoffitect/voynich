"""
Voynich Manuscript IVTFF Transcript Parser
==========================================
Parses the interlinear EVA transcription file, builds consensus readings
across multiple transcribers, and produces data structures for NLP analysis
(word order / SVO vs SOV detection).

Transcriber codes (primary):
  H = Takeshi Takahashi (complete)
  C = Currier + voynich list members
  F = First Study Group (Friedman)
  N = Gabriel Landini
  U = Jorge Stolfi
  V = John Grove

EVA conventions:
  '.' = definite word break
  ',' = dubious word break
  '-' = line break (line-final) or major gap
  '=' = paragraph break (line-final)
  '!' = filler (no character here)
  '%' = filler (no information)
  '?' = unreadable character
  '{...}' = inline comment
  '<...>' = special notation (weirdos, etc.)
"""

import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriberLine:
    """One transcriber's reading of a single manuscript line."""
    folio: str          # e.g. "f1r"
    line_num: str       # e.g. "1"
    unit: str           # e.g. "P0", "Pt", "L1"
    transcriber: str    # e.g. "H", "C", "F"
    raw: str            # raw EVA text as-is
    words: list[str] = field(default_factory=list)  # cleaned word tokens


@dataclass
class ConsensusLine:
    """Consensus reading for one manuscript line, built from all transcribers."""
    folio: str
    line_num: str
    unit: str
    consensus_words: list[str]          # majority-vote word sequence
    transcriber_readings: dict[str, list[str]] = field(default_factory=dict)
    agreement_scores: list[float] = field(default_factory=list)  # per-word


@dataclass
class FolioMetadata:
    """Metadata parsed from page header lines."""
    folio: str
    illustration: str = ""   # T,H,A,Z,B,C,P,S
    quire: str = ""
    page_in_quire: str = ""
    language: str = ""       # Currier A or B
    hand: str = ""


@dataclass
class Sentence:
    """A 'sentence' (paragraph-delimited sequence) for NLP analysis."""
    folio: str
    unit: str
    words: list[str]
    raw_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches data lines: <f1r.3,+P0;H>  text...
LINE_RE = re.compile(
    r'^<(?P<folio>f\d+[rv]\d?)\.(?P<line>\d+)'
    r',(?P<break_char>[+@*=])(?P<unit>[^;]+);(?P<transcriber>[A-Z])>'
    r'\s+(?P<text>.+?)\s*$'
)

# Page/unit header with parsable info
HEADER_RE = re.compile(
    r'^<(?P<folio>f\d+[rv]\d?)>\s+<!\s*(?P<vars>[^>]+)>'
)

# Inline comments  {stuff}
COMMENT_RE = re.compile(r'\{[^}]*\}')
# Weirdo / special notation  <stuff>
WEIRDO_RE = re.compile(r'<[^>]*>')

# Primary transcribers we want for consensus
PRIMARY_TRANSCRIBERS = {'H', 'C', 'F', 'N', 'U', 'V'}


def clean_eva_text(raw: str) -> str:
    """Strip comments, weirdos, and fillers from EVA text."""
    text = COMMENT_RE.sub('', raw)
    text = WEIRDO_RE.sub('', text)
    # Remove fillers
    text = text.replace('!', '').replace('%', '')
    # Normalise break chars at end of line
    text = text.rstrip('-=')
    return text.strip()


def tokenize_eva(cleaned: str) -> list[str]:
    """Split cleaned EVA text into word tokens on '.' and ','."""
    tokens = re.split(r'[.,]+', cleaned)
    return [t.strip() for t in tokens if t.strip() and t.strip() != '?']


def parse_metadata(line: str) -> Optional[FolioMetadata]:
    m = HEADER_RE.match(line)
    if not m:
        return None
    meta = FolioMetadata(folio=m.group('folio'))
    for var_match in re.finditer(r'\$(\w)=(\S+)', m.group('vars')):
        key, val = var_match.group(1), var_match.group(2)
        if key == 'I': meta.illustration = val
        elif key == 'Q': meta.quire = val
        elif key == 'P': meta.page_in_quire = val
        elif key == 'L': meta.language = val
        elif key == 'H': meta.hand = val
    return meta


def parse_transcript(filepath: str) -> tuple[
    list[TranscriberLine],
    dict[str, FolioMetadata],
]:
    """Parse the IVTFF file into structured data."""
    lines_out: list[TranscriberLine] = []
    metadata: dict[str, FolioMetadata] = {}

    with open(filepath, 'r', encoding='latin-1') as fh:
        for raw_line in fh:
            raw_line = raw_line.rstrip('\n')

            # Try metadata header
            meta = parse_metadata(raw_line)
            if meta:
                metadata[meta.folio] = meta
                continue

            # Try data line
            m = LINE_RE.match(raw_line)
            if not m:
                continue

            cleaned = clean_eva_text(m.group('text'))
            words = tokenize_eva(cleaned)

            tl = TranscriberLine(
                folio=m.group('folio'),
                line_num=m.group('line'),
                unit=m.group('unit'),
                transcriber=m.group('transcriber'),
                raw=m.group('text').strip(),
                words=words,
            )
            lines_out.append(tl)

    return lines_out, metadata


# ---------------------------------------------------------------------------
# Consensus building
# ---------------------------------------------------------------------------

def build_consensus(
    lines: list[TranscriberLine],
    transcribers: set[str] | None = None,
) -> list[ConsensusLine]:
    """Build majority-vote consensus for each manuscript line.

    Groups by (folio, line_num) and aligns word positions across transcribers.
    For each position, the most common non-'?' reading wins.
    """
    if transcribers is None:
        transcribers = PRIMARY_TRANSCRIBERS

    # Group by (folio, line_num)
    groups: dict[tuple[str, str], list[TranscriberLine]] = defaultdict(list)
    for tl in lines:
        if tl.transcriber in transcribers:
            groups[(tl.folio, tl.line_num)].append(tl)

    consensus_lines: list[ConsensusLine] = []

    for (folio, line_num), group in sorted(groups.items(), key=lambda x: _sort_key(x[0])):
        readings = {tl.transcriber: tl.words for tl in group}
        unit = group[0].unit

        # Find max word count across transcribers
        max_words = max(len(w) for w in readings.values()) if readings else 0

        consensus_words = []
        agreement_scores = []

        for i in range(max_words):
            votes: Counter = Counter()
            total_voters = 0
            for tr_words in readings.values():
                if i < len(tr_words):
                    word = tr_words[i]
                    if word and word != '?':
                        votes[word] += 1
                        total_voters += 1

            if votes:
                winner, count = votes.most_common(1)[0]
                consensus_words.append(winner)
                agreement_scores.append(count / max(total_voters, 1))
            elif total_voters == 0 and any(i < len(r) for r in readings.values()):
                # All voters said '?'
                consensus_words.append('???')
                agreement_scores.append(0.0)

        consensus_lines.append(ConsensusLine(
            folio=folio,
            line_num=line_num,
            unit=unit,
            consensus_words=consensus_words,
            transcriber_readings=readings,
            agreement_scores=agreement_scores,
        ))

    return consensus_lines


def _sort_key(key: tuple[str, str]) -> tuple[int, str, int]:
    """Sort key for (folio, line_num) pairs."""
    folio, line = key
    # Extract numeric part of folio
    m = re.match(r'f(\d+)([rv]\d?)', folio)
    if m:
        return (int(m.group(1)), m.group(2), int(line))
    return (0, folio, int(line) if line.isdigit() else 0)


# ---------------------------------------------------------------------------
# Sentence extraction (paragraph-delimited)
# ---------------------------------------------------------------------------

def extract_sentences(
    consensus_lines: list[ConsensusLine],
) -> list[Sentence]:
    """Group consensus lines into 'sentences' by paragraph unit.

    Each text unit (P0, P1, etc.) within a folio is treated as a sentence.
    """
    groups: dict[tuple[str, str], list[ConsensusLine]] = defaultdict(list)
    for cl in consensus_lines:
        groups[(cl.folio, cl.unit)].append(cl)

    sentences = []
    for (folio, unit), cls in sorted(groups.items()):
        words = []
        raw_lines = []
        for cl in cls:
            words.extend(cl.consensus_words)
            raw_lines.append(' '.join(cl.consensus_words))
        if words:
            sentences.append(Sentence(
                folio=folio,
                unit=unit,
                words=words,
                raw_lines=raw_lines,
            ))
    return sentences


# ---------------------------------------------------------------------------
# Vocabulary & word-order analysis
# ---------------------------------------------------------------------------

@dataclass
class VocabStats:
    total_tokens: int = 0
    unique_types: int = 0
    hapax_legomena: int = 0  # words appearing exactly once
    freq: dict[str, int] = field(default_factory=dict)
    type_token_ratio: float = 0.0


def build_vocabulary(sentences: list[Sentence]) -> VocabStats:
    freq: Counter = Counter()
    for s in sentences:
        freq.update(s.words)

    total = sum(freq.values())
    unique = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)

    return VocabStats(
        total_tokens=total,
        unique_types=unique,
        hapax_legomena=hapax,
        freq=dict(freq.most_common()),
        type_token_ratio=unique / total if total else 0.0,
    )


def positional_frequency(sentences: list[Sentence], top_n: int = 30) -> dict:
    """Compute word frequency by position in sentence.

    Key insight for SVO vs SOV detection:
    - SVO languages: verbs cluster in middle positions
    - SOV languages: verbs cluster at end positions

    We bucket positions into: initial (first 20%), medial (middle 60%), final (last 20%).
    """
    initial_freq: Counter = Counter()
    medial_freq: Counter = Counter()
    final_freq: Counter = Counter()

    for s in sentences:
        n = len(s.words)
        if n < 3:
            continue
        boundary1 = max(1, int(n * 0.2))
        boundary2 = max(boundary1 + 1, int(n * 0.8))

        for i, w in enumerate(s.words):
            if i < boundary1:
                initial_freq[w] += 1
            elif i >= boundary2:
                final_freq[w] += 1
            else:
                medial_freq[w] += 1

    return {
        'initial': dict(initial_freq.most_common(top_n)),
        'medial': dict(medial_freq.most_common(top_n)),
        'final': dict(final_freq.most_common(top_n)),
    }


def bigram_analysis(sentences: list[Sentence], top_n: int = 50) -> dict:
    """Compute bigram frequencies to detect rigid word-pair patterns.

    Consistent bigram patterns suggest fixed phrase structures.
    In SVO: subject-verb pairs should be common.
    In SOV: object-verb pairs at sentence ends should be common.
    """
    bigrams: Counter = Counter()
    sentence_final_bigrams: Counter = Counter()

    for s in sentences:
        for i in range(len(s.words) - 1):
            bg = (s.words[i], s.words[i + 1])
            bigrams[bg] += 1
            if i == len(s.words) - 2:
                sentence_final_bigrams[bg] += 1

    return {
        'overall': {f"{a} {b}": c for (a, b), c in bigrams.most_common(top_n)},
        'sentence_final': {f"{a} {b}": c for (a, b), c in sentence_final_bigrams.most_common(top_n)},
    }


def word_position_entropy(sentences: list[Sentence], min_freq: int = 5) -> dict[str, float]:
    """For frequent words, compute the entropy of their position distribution.

    Words with LOW entropy appear in fixed positions (function words: articles,
    postpositions, case markers).
    Words with HIGH entropy appear freely (content words: nouns).

    In SOV languages, postpositions have low entropy at sentence ends.
    In SVO, prepositions have low entropy at sentence beginnings.
    """
    import math

    word_positions: dict[str, list[float]] = defaultdict(list)
    for s in sentences:
        n = len(s.words)
        if n < 3:
            continue
        for i, w in enumerate(s.words):
            word_positions[w].append(i / (n - 1))  # normalised 0..1

    entropies = {}
    for word, positions in word_positions.items():
        if len(positions) < min_freq:
            continue
        # Bin into 5 buckets
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

    # Sort by entropy (low = positionally rigid)
    return dict(sorted(entropies.items(), key=lambda x: x[1]))


def detect_word_order(sentences: list[Sentence]) -> dict:
    """Aggregate heuristics for SVO vs SOV detection.

    Returns a report with evidence for each hypothesis.
    """
    vocab = build_vocabulary(sentences)
    pos_freq = positional_frequency(sentences)
    bigrams = bigram_analysis(sentences)
    entropy = word_position_entropy(sentences)

    # Heuristic: words that strongly prefer final position
    final_heavy = set(pos_freq['final'].keys()) - set(pos_freq['initial'].keys())
    initial_heavy = set(pos_freq['initial'].keys()) - set(pos_freq['final'].keys())

    # Low-entropy words by position preference
    low_entropy_words = {w: e for w, e in entropy.items() if e < 1.5}

    # Check if low-entropy words cluster at ends (SOV signal) or beginnings (SVO signal)
    end_rigid = []
    start_rigid = []
    for w in low_entropy_words:
        positions = []
        for s in sentences:
            n = len(s.words)
            if n < 3:
                continue
            for i, sw in enumerate(s.words):
                if sw == w:
                    positions.append(i / (n - 1))
        if positions:
            mean_pos = sum(positions) / len(positions)
            if mean_pos > 0.7:
                end_rigid.append((w, round(mean_pos, 2), low_entropy_words[w]))
            elif mean_pos < 0.3:
                start_rigid.append((w, round(mean_pos, 2), low_entropy_words[w]))

    # Sentence-final word diversity (SOV has less diversity at ends = verb-final)
    final_words: Counter = Counter()
    for s in sentences:
        if s.words:
            final_words[s.words[-1]] += 1
    final_diversity = len(final_words) / max(len(sentences), 1)

    initial_words: Counter = Counter()
    for s in sentences:
        if s.words:
            initial_words[s.words[0]] += 1
    initial_diversity = len(initial_words) / max(len(sentences), 1)

    return {
        'vocab_summary': {
            'total_tokens': vocab.total_tokens,
            'unique_types': vocab.unique_types,
            'hapax_legomena': vocab.hapax_legomena,
            'type_token_ratio': round(vocab.type_token_ratio, 4),
            'top_20_words': dict(list(vocab.freq.items())[:20]),
        },
        'positional_frequency': pos_freq,
        'bigrams': bigrams,
        'positionally_rigid_words': {
            'end_rigid (SOV signal)': end_rigid[:15],
            'start_rigid (SVO signal)': start_rigid[:15],
        },
        'sentence_boundary_diversity': {
            'initial_word_diversity': round(initial_diversity, 4),
            'final_word_diversity': round(final_diversity, 4),
            'note': 'SOV languages tend to have LOWER final diversity (verb-final)',
        },
        'sentence_final_bigrams': bigrams['sentence_final'],
    }


# ---------------------------------------------------------------------------
# Character-repetition variant analysis
# ---------------------------------------------------------------------------

def skeleton(word: str) -> str:
    """Collapse consecutive repeated chars: 'daiin' -> 'dain'."""
    return re.sub(r'(.)\1+', r'\1', word)


def find_repetition_families(
    sentences: list[Sentence],
    metadata: dict[str, FolioMetadata],
    min_total_freq: int = 10,
) -> dict:
    """Find word families that differ only by character repetition.

    For each family, compute:
    - Frequency of each variant
    - Language A vs B distribution
    - Positional distribution (5 buckets)
    - Collocation overlap (Jaccard of top-15 following words)
    - Co-occurrence on same folios vs segregation
    """
    # Full word frequency
    freq: Counter = Counter()
    for s in sentences:
        freq.update(s.words)

    # Group by skeleton
    skel_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for w, c in freq.items():
        sk = skeleton(w)
        skel_groups[sk].append((w, c))

    # Keep multi-variant families above threshold
    families_raw = {}
    for sk, members in skel_groups.items():
        if len(members) > 1 and sum(c for _, c in members) >= min_total_freq:
            families_raw[sk] = sorted(members, key=lambda x: -x[1])

    # Build word -> language counts and word -> following-word counter
    word_lang: dict[str, dict[str, int]] = defaultdict(lambda: {'A': 0, 'B': 0, '?': 0})
    word_next: dict[str, Counter] = defaultdict(Counter)
    word_positions: dict[str, list[int]] = defaultdict(list)  # bucket indices

    for s in sentences:
        folio = s.folio if isinstance(s, Sentence) else s['folio']
        words = s.words if isinstance(s, Sentence) else s['words']
        lang_raw = ''
        if isinstance(metadata.get(folio), FolioMetadata):
            lang_raw = metadata[folio].language
        elif isinstance(metadata.get(folio), dict):
            lang_raw = metadata[folio].get('language', '')
        lang = lang_raw if lang_raw in ('A', 'B') else '?'

        n = len(words)
        for i, w in enumerate(words):
            word_lang[w][lang] += 1
            if i < n - 1:
                word_next[w][words[i + 1]] += 1
            if n >= 3:
                bucket = min(4, int((i / max(n - 1, 1)) * 5))
                word_positions[w].append(bucket)

    # Word -> folio counter for clustering test
    word_folios: dict[str, Counter] = defaultdict(Counter)
    for s in sentences:
        folio = s.folio if isinstance(s, Sentence) else s['folio']
        words = s.words if isinstance(s, Sentence) else s['words']
        for w in words:
            word_folios[w][folio] += 1

    # Analyse each family
    results = []
    for sk, members in sorted(families_raw.items(), key=lambda x: -sum(c for _, c in x[1])):
        total = sum(c for _, c in members)
        # Identify which character is being repeated
        repeated_chars = set()
        for w, _ in members:
            for m in re.finditer(r'(.)\1+', w):
                repeated_chars.add(m.group(1))

        variants = []
        for w, c in members:
            lang_dist = dict(word_lang[w])
            pos_buckets = [0] * 5
            for b in word_positions[w]:
                pos_buckets[b] += 1
            pos_total = sum(pos_buckets)
            pos_pct = [round(x / max(pos_total, 1), 2) for x in pos_buckets]
            variants.append({
                'word': w,
                'count': c,
                'lang_A': lang_dist.get('A', 0),
                'lang_B': lang_dist.get('B', 0),
                'position_pct': pos_pct,
                'top_followers': dict(word_next[w].most_common(10)),
            })

        # Collocation overlap between longest and shortest form
        if len(members) >= 2:
            shortest = members[-1][0]
            longest = members[0][0]
            s_top = set(w for w, _ in word_next[shortest].most_common(15))
            l_top = set(w for w, _ in word_next[longest].most_common(15))
            union = s_top | l_top
            jaccard = len(s_top & l_top) / max(len(union), 1)
        else:
            jaccard = 1.0

        # Folio clustering
        if len(members) >= 2:
            w1, w2 = members[0][0], members[-1][0]
            f1 = set(word_folios[w1])
            f2 = set(word_folios[w2])
            both = len(f1 & f2)
            only1 = len(f1 - f2)
            only2 = len(f2 - f1)
        else:
            both = only1 = only2 = 0

        results.append({
            'skeleton': sk,
            'total_freq': total,
            'repeated_chars': sorted(repeated_chars),
            'variants': variants,
            'collocation_jaccard': round(jaccard, 2),
            'folio_overlap': {'both': both, 'only_long': only1, 'only_short': only2},
        })

    return {
        'families': results,
        'summary': {
            'total_families': len(results),
            'by_repeated_char': dict(Counter(
                ch for r in results for ch in r['repeated_chars']
            ).most_common()),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else 'data/transcription/transcript.txt'

    print("Parsing IVTFF transcript...")
    lines, metadata = parse_transcript(filepath)
    print(f"  Parsed {len(lines)} transcriber lines across {len(metadata)} folios")

    transcriber_counts = Counter(tl.transcriber for tl in lines)
    print(f"  Transcriber line counts: {dict(transcriber_counts.most_common())}")

    print("\nBuilding consensus (primary transcribers: {})...".format(
        ', '.join(sorted(PRIMARY_TRANSCRIBERS))
    ))
    consensus = build_consensus(lines)
    print(f"  Built {len(consensus)} consensus lines")

    # Show a sample
    print("\n--- Sample consensus (f1r, first 5 lines) ---")
    for cl in consensus[:5]:
        score = sum(cl.agreement_scores) / len(cl.agreement_scores) if cl.agreement_scores else 0
        print(f"  <{cl.folio}.{cl.line_num}> [{score:.0%}] {' '.join(cl.consensus_words)}")
        for tr, words in sorted(cl.transcriber_readings.items()):
            print(f"    {tr}: {' '.join(words)}")

    print("\nExtracting sentences...")
    sentences = extract_sentences(consensus)
    print(f"  Extracted {len(sentences)} sentence units")

    # Filter to paragraph-type units for word-order analysis
    para_sentences = [s for s in sentences if s.unit.startswith('P')]
    print(f"  Of which {len(para_sentences)} are paragraph text (unit P*)")

    print("\n" + "=" * 70)
    print("WORD ORDER ANALYSIS (SVO vs SOV detection)")
    print("=" * 70)

    report = detect_word_order(para_sentences)

    print(f"\nVocabulary:")
    vs = report['vocab_summary']
    print(f"  Tokens: {vs['total_tokens']}  Types: {vs['unique_types']}  "
          f"Hapax: {vs['hapax_legomena']}  TTR: {vs['type_token_ratio']}")
    print(f"  Top 20: {list(vs['top_20_words'].items())}")

    print(f"\nPositional frequency (top words by sentence zone):")
    for zone in ('initial', 'medial', 'final'):
        top5 = list(report['positional_frequency'][zone].items())[:5]
        print(f"  {zone:8s}: {top5}")

    print(f"\nPositionally rigid words:")
    for label, items in report['positionally_rigid_words'].items():
        print(f"  {label}:")
        for word, mean_pos, ent in items[:10]:
            print(f"    {word:20s}  mean_pos={mean_pos}  entropy={ent}")

    bd = report['sentence_boundary_diversity']
    print(f"\nSentence boundary diversity:")
    print(f"  Initial: {bd['initial_word_diversity']}")
    print(f"  Final:   {bd['final_word_diversity']}")
    print(f"  ({bd['note']})")

    print(f"\nTop sentence-final bigrams:")
    for bg, count in list(report['sentence_final_bigrams'].items())[:10]:
        print(f"  {bg:30s}  {count}")

    # Variant / repetition analysis
    print("\n" + "=" * 70)
    print("CHARACTER REPETITION VARIANT ANALYSIS")
    print("=" * 70)

    variant_report = find_repetition_families(sentences, metadata)
    vs = variant_report['summary']
    print(f"\nFound {vs['total_families']} variant families")
    print(f"Most-repeated characters: {vs['by_repeated_char']}")

    print(f"\nTop 20 families by frequency:")
    print(f"{'Skeleton':20s} {'Total':>5s}  Variants (word: count, langA/B, position%)")
    print("-" * 100)
    for fam in variant_report['families'][:20]:
        sk = fam['skeleton']
        total = fam['total_freq']
        jac = fam['collocation_jaccard']
        fo = fam['folio_overlap']
        print(f"\n{sk:20s} [{total:4d}]  colloc_overlap={jac:.2f}  "
              f"folios: both={fo['both']} long_only={fo['only_long']} short_only={fo['only_short']}")
        for v in fam['variants']:
            pos = v['position_pct']
            print(f"  {v['word']:18s} n={v['count']:4d}  A/B={v['lang_A']:3d}/{v['lang_B']:3d}  "
                  f"pos=[{pos[0]:.0%} {pos[1]:.0%} {pos[2]:.0%} {pos[3]:.0%} {pos[4]:.0%}]")

    # Summarise the i-doubling vs e-doubling patterns
    i_families = [f for f in variant_report['families'] if 'i' in f['repeated_chars']]
    e_families = [f for f in variant_report['families'] if 'e' in f['repeated_chars']]

    print(f"\n{'='*70}")
    print("I-DOUBLING vs E-DOUBLING: Language A/B bias")
    print(f"{'='*70}")

    for label, fams in [("I-doubling", i_families), ("E-doubling", e_families)]:
        long_a = long_b = short_a = short_b = 0
        for f in fams:
            if len(f['variants']) < 2:
                continue
            longest = f['variants'][0]  # most frequent = usually the long form
            shortest = f['variants'][-1]
            # Determine which is actually longer
            if len(longest['word']) < len(shortest['word']):
                longest, shortest = shortest, longest
            long_a += longest['lang_A']
            long_b += longest['lang_B']
            short_a += shortest['lang_A']
            short_b += shortest['lang_B']

        ratio_a = long_a / max(short_a, 1)
        ratio_b = long_b / max(short_b, 1)
        print(f"\n{label}:")
        print(f"  Long forms:  Lang A={long_a:5d}  Lang B={long_b:5d}")
        print(f"  Short forms: Lang A={short_a:5d}  Lang B={short_b:5d}")
        print(f"  Long/Short ratio:  A={ratio_a:.2f}  B={ratio_b:.2f}")

    # Export for downstream NLP
    export = {
        'sentences': [asdict(s) for s in para_sentences],
        'word_order_report': report,
        'variant_analysis': variant_report,
        'metadata': {k: asdict(v) for k, v in metadata.items()},
    }
    with open('data/transcription/voynich_nlp.json', 'w') as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\nExported {len(para_sentences)} paragraph sentences + variant analysis to voynich_nlp.json")


if __name__ == '__main__':
    main()
