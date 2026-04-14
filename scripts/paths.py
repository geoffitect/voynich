"""
Centralized path helper for Voynich analysis scripts.
All scripts should `from paths import *` to get these defined locations.
Run scripts from the project root.
"""
from pathlib import Path

# Root is the parent of the scripts/ directory
ROOT = Path(__file__).resolve().parent.parent

# --- Input data ---
TRANSCRIPT = ROOT / 'data' / 'transcription' / 'transcript.txt'
EVA_UNICODE = ROOT / 'data' / 'transcription' / 'EVA_unicode.txt'
NLP = ROOT / 'data' / 'transcription' / 'voynich_nlp.json'
TOKENIZED = ROOT / 'data' / 'transcription' / 'tokenized_corpus.json'

# --- Analysis outputs ---
ANALYSIS = ROOT / 'data' / 'analysis'
VECTORS = ANALYSIS / 'word_vectors.json'
ASTRO_VOCAB = ANALYSIS / 'astro_vocab.json'
PLANT = ANALYSIS / 'plant_analysis.json'
COLOR = ANALYSIS / 'color_crossref.json'
VISUAL_XREF = ANALYSIS / 'visual_crossref.json'
HAPAX = ANALYSIS / 'hapax_analysis.json'
DECLENSION = ANALYSIS / 'declension_data.json'
GLYPH_RE = ANALYSIS / 'glyph_reanalysis.json'

# --- Lexicon and label data ---
LEXICON_DIR = ROOT / 'data' / 'lexicon'
LEXICON = LEXICON_DIR / 'lexicon.json'
LABEL_NETWORK = LEXICON_DIR / 'label_network.json'
LABEL_XREF = LEXICON_DIR / 'label_crossref.json'
FIGURES = LEXICON_DIR / 'figure_database.json'

# --- Visual pipeline ---
VISUAL_DIR = ROOT / 'data' / 'visual'
SEGMENTATION = VISUAL_DIR / 'segmentation_data.json'
OBJECT_CATALOG = VISUAL_DIR / 'object_catalog.json'

# --- Images (gitignored) ---
FOLIOS = ROOT / 'folios'
CROPS = ROOT / 'crops'
ANNOTATED = ROOT / 'annotated'
SOURCE_PDF = ROOT / 'source' / 'voynich_scan_yale.pdf'

# --- Reports ---
REPORTS = ROOT / 'reports'
HTML_REPORTS = REPORTS / 'html'
TEXT_REPORTS = REPORTS / 'text'

# --- Fonts ---
FONTS = ROOT / 'fonts'
VOYNICH_FONT = FONTS / 'Voynich' / 'VoynichEVA.ttf'
