#!/usr/bin/env python3
"""
Voynich Manuscript: Astronomical Vocabulary Analysis

Identifies words disproportionately common on astronomical pages
(illustration types A=Astronomical, Z=Zodiac, S=Stars, C=Cosmological)
versus the rest of the manuscript.

Uses log-odds ratio to measure distinctiveness.
Also extracts label text (unit codes starting with "L") from the transcript.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
JSON_PATH = BASE / "data/transcription/voynich_nlp.json"
TRANSCRIPT_PATH = BASE / "transcript.txt"
OUTPUT_PATH = BASE / "astro_vocab.json"

ASTRO_TYPES = {"A", "Z", "S", "C"}


def load_data():
    with open(JSON_PATH) as f:
        data = json.load(f)
    return data


def get_astro_folios(metadata):
    """Return set of folio IDs with astronomical illustration types."""
    astro = set()
    non_astro = set()
    for folio_id, meta in metadata.items():
        if meta.get("illustration") in ASTRO_TYPES:
            astro.add(folio_id)
        else:
            non_astro.add(folio_id)
    return astro, non_astro


def build_word_counts(sentences, astro_folios):
    """Build word frequency tables for astro vs non-astro pages."""
    astro_words = Counter()
    other_words = Counter()
    astro_total = 0
    other_total = 0

    for sent in sentences:
        folio = sent["folio"]
        words = sent["words"]
        if folio in astro_folios:
            astro_words.update(words)
            astro_total += len(words)
        else:
            other_words.update(words)
            other_total += len(words)

    return astro_words, other_words, astro_total, other_total


def compute_log_odds(astro_words, other_words, astro_total, other_total, min_count=2):
    """
    Compute log-odds ratio for each word.
    Positive = more common on astro pages.
    Uses additive smoothing to avoid division by zero.
    """
    all_words = set(astro_words) | set(other_words)
    vocab_size = len(all_words)
    alpha = 0.5  # smoothing

    results = []
    for word in all_words:
        a_count = astro_words.get(word, 0)
        o_count = other_words.get(word, 0)
        total_count = a_count + o_count

        if total_count < min_count:
            continue

        # Smoothed probabilities
        p_astro = (a_count + alpha) / (astro_total + alpha * vocab_size)
        p_other = (o_count + alpha) / (other_total + alpha * vocab_size)

        log_odds = math.log2(p_astro / p_other)

        # Also compute a simple ratio for interpretability
        astro_rate = a_count / astro_total if astro_total else 0
        other_rate = o_count / other_total if other_total else 0

        results.append({
            "word": word,
            "astro_count": a_count,
            "other_count": o_count,
            "total_count": total_count,
            "log_odds": round(log_odds, 3),
            "astro_rate_per_1k": round(astro_rate * 1000, 2),
            "other_rate_per_1k": round(other_rate * 1000, 2),
        })

    # Sort by log-odds descending
    results.sort(key=lambda x: x["log_odds"], reverse=True)
    return results


def extract_labels_from_transcript(transcript_path, astro_folios):
    """
    Extract label text from transcript.txt for astronomical folios.
    Labels have unit codes starting with L (e.g., @L0, @Ls, @La, @Lp, @Lx).
    We prefer the H (Herbal/primary) transcription, falling back to others.
    """
    label_pattern = re.compile(
        r'^<(f[\w]+)\.(\d+),[@=+]?(L\w*);(\w)>\s+(.+)$'
    )

    # Collect all readings per (folio, line_num, label_type)
    raw_labels = {}
    with open(transcript_path) as f:
        for line in f:
            line = line.rstrip()
            m = label_pattern.match(line)
            if not m:
                continue
            folio = m.group(1)
            line_num = m.group(2)
            label_unit = m.group(3)
            transcriber = m.group(4)
            text = m.group(5)

            # Normalize folio name to match metadata keys
            if folio not in astro_folios:
                continue

            key = (folio, line_num, label_unit)
            if key not in raw_labels:
                raw_labels[key] = {}
            raw_labels[key][transcriber] = text

    # Pick best transcription (prefer H, then V, then U, then first available)
    priority = ['H', 'V', 'U', 'C', 'F', 'N', 'G', 'J', 'X', 'D']
    labels = []
    for (folio, line_num, label_unit), readings in sorted(raw_labels.items()):
        chosen = None
        for p in priority:
            if p in readings:
                chosen = readings[p]
                break
        if chosen is None:
            chosen = list(readings.values())[0]

        # Clean up the text: remove comments, special markers
        text = re.sub(r'<[^>]*>', '', chosen)  # remove <...> tags
        text = re.sub(r'[!%?]', '', text)       # remove uncertainty markers
        text = text.strip()
        if not text:
            continue

        # Split into words (dots are word separators in EVA)
        words = [w.strip() for w in text.split('.') if w.strip()]

        labels.append({
            "folio": folio,
            "line": line_num,
            "unit": label_unit,
            "text": text,
            "words": words,
        })

    return labels


def get_illustration_type_map(metadata):
    """Map folio to its illustration type."""
    return {k: v.get("illustration", "?") for k, v in metadata.items()}


def main():
    data = load_data()
    metadata = data["metadata"]
    sentences = data["sentences"]

    astro_folios, non_astro_folios = get_astro_folios(metadata)
    illus_map = get_illustration_type_map(metadata)

    print(f"=== Voynich Astronomical Vocabulary Analysis ===")
    print(f"Astro folios (A/Z/S/C): {len(astro_folios)}")
    print(f"Other folios: {len(non_astro_folios)}")
    print()

    # Breakdown by type
    type_counts = Counter(illus_map[f] for f in astro_folios)
    for t in sorted(type_counts):
        label = {"A": "Astronomical", "Z": "Zodiac", "S": "Stars", "C": "Cosmological"}[t]
        print(f"  {t} ({label}): {type_counts[t]} folios")
    print()

    # Build word frequency tables
    astro_words, other_words, astro_total, other_total = build_word_counts(sentences, astro_folios)
    print(f"Total words on astro pages: {astro_total}")
    print(f"Total words on other pages: {other_total}")
    print(f"Unique words on astro pages: {len(astro_words)}")
    print(f"Unique words on other pages: {len(other_words)}")
    print(f"Words ONLY on astro pages: {len(set(astro_words) - set(other_words))}")
    print(f"Words ONLY on other pages: {len(set(other_words) - set(astro_words))}")
    print()

    # Compute log-odds
    results = compute_log_odds(astro_words, other_words, astro_total, other_total, min_count=2)

    # Show top astro-distinctive words
    print("=" * 70)
    print("TOP 50 ASTRO-DISTINCTIVE WORDS (by log-odds ratio)")
    print("=" * 70)
    print(f"{'Word':<20} {'LogOdds':>8} {'Astro':>6} {'Other':>6} {'A/1k':>7} {'O/1k':>7}")
    print("-" * 70)
    for r in results[:50]:
        print(f"{r['word']:<20} {r['log_odds']:>8.3f} {r['astro_count']:>6} {r['other_count']:>6} "
              f"{r['astro_rate_per_1k']:>7.2f} {r['other_rate_per_1k']:>7.2f}")
    print()

    # Words exclusive to astro pages (appearing 3+ times)
    astro_exclusive = [r for r in results if r["other_count"] == 0 and r["astro_count"] >= 3]
    astro_exclusive.sort(key=lambda x: x["astro_count"], reverse=True)
    print("=" * 70)
    print("WORDS FOUND ONLY ON ASTRO PAGES (count >= 3)")
    print("=" * 70)
    for r in astro_exclusive:
        print(f"  {r['word']:<25} count={r['astro_count']}")
    print(f"  Total: {len(astro_exclusive)} words")
    print()

    # Breakdown by astro sub-type
    print("=" * 70)
    print("WORD COUNTS BY ASTRO SUB-TYPE")
    print("=" * 70)
    for atype in ["A", "Z", "S", "C"]:
        type_folios = {f for f in astro_folios if illus_map.get(f) == atype}
        type_words = Counter()
        for sent in sentences:
            if sent["folio"] in type_folios:
                type_words.update(sent["words"])
        total = sum(type_words.values())
        label = {"A": "Astronomical", "Z": "Zodiac", "S": "Stars", "C": "Cosmological"}[atype]
        print(f"\n  {atype} ({label}): {len(type_folios)} folios, {total} words, {len(type_words)} unique")
        print(f"  Top 15: {', '.join(f'{w}({c})' for w, c in type_words.most_common(15))}")

    print()

    # Extract labels from transcript
    print("=" * 70)
    print("LABEL TEXT FROM ASTRONOMICAL FOLIOS (from transcript.txt)")
    print("=" * 70)
    labels = extract_labels_from_transcript(TRANSCRIPT_PATH, astro_folios)
    print(f"Total label entries: {len(labels)}")
    print()

    # Group by folio type
    label_by_type = {}
    for lab in labels:
        atype = illus_map.get(lab["folio"], "?")
        label_by_type.setdefault(atype, []).append(lab)

    all_label_words = Counter()
    for lab in labels:
        all_label_words.update(lab["words"])

    for atype in ["A", "Z", "S", "C"]:
        type_labels = label_by_type.get(atype, [])
        if not type_labels:
            continue
        label_name = {"A": "Astronomical", "Z": "Zodiac", "S": "Stars", "C": "Cosmological"}[atype]
        print(f"\n--- {atype} ({label_name}) labels ---")
        for lab in type_labels:
            print(f"  {lab['folio']}.{lab['line']} [{lab['unit']}]: {lab['text']}  ->  {lab['words']}")

    print()
    print("=" * 70)
    print("ALL LABEL WORDS FROM ASTRO PAGES (frequency)")
    print("=" * 70)
    for word, count in all_label_words.most_common():
        print(f"  {word:<25} {count}")
    print(f"  Total label words: {sum(all_label_words.values())}, unique: {len(all_label_words)}")

    # Save results to JSON
    output = {
        "summary": {
            "astro_folio_count": len(astro_folios),
            "other_folio_count": len(non_astro_folios),
            "astro_word_total": astro_total,
            "other_word_total": other_total,
            "astro_unique_words": len(astro_words),
            "other_unique_words": len(other_words),
            "astro_exclusive_words": len(set(astro_words) - set(other_words)),
        },
        "astro_distinctive_words": results[:100],
        "astro_exclusive_words": astro_exclusive,
        "label_entries": labels,
        "label_word_frequencies": dict(all_label_words.most_common()),
        "all_log_odds": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
