import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# ── NLTK data bootstrap ───────────────────────────────────────────────────────
for _resource, _path in [
    ("punkt",       "tokenizers/punkt"),
    ("punkt_tab",   "tokenizers/punkt_tab"),
    ("stopwords",   "corpora/stopwords"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FILLER_WORDS = frozenset({
    "um", "uh", "like", "you know", "actually", "basically",
    "literally", "yeah", "hmm", "ah",
})


class Preprocessor:
    """Handles text cleaning and tokenization for meeting transcripts."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english")).union(FILLER_WORDS)

    def clean_text(self, text: str) -> str:
        """Normalize text: lowercase, remove special chars, and strip filler words.

        Args:
            text: Raw input string.

        Returns:
            Cleaned string suitable for feature extraction.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s\.,!?]", "", text)
        words = word_tokenize(text)
        cleaned = [w for w in words if w not in FILLER_WORDS]
        return " ".join(cleaned)

    def tokenize_sentences(self, text: str) -> list:
        """Split text into a list of sentences."""
        return sent_tokenize(text)

    def tokenize_words(self, text: str) -> list:
        """Split text into a list of word tokens."""
        return word_tokenize(text)

    def process_for_extractive(self, text: str) -> tuple:
        """Return original sentences and their cleaned counterparts.

        Args:
            text: Raw transcript string.

        Returns:
            Tuple of (original_sentences, cleaned_sentences).
        """
        sentences = self.tokenize_sentences(text)
        cleaned_sentences = [self.clean_text(s) for s in sentences]
        return sentences, cleaned_sentences


if __name__ == "__main__":
    p = Preprocessor()
    sample = "Alice: Uh, yeah, basically we need to, like, deploy this tomorrow!"
    print("Cleaned:", p.clean_text(sample))
