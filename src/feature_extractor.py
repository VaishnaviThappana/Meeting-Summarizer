import re
import sys
import spacy
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning)


class FeatureExtractor:
    """Extracts keywords and named entities from text using TF-IDF and spaCy."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Use the current Python executable to avoid calling the wrong venv
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            self.nlp = spacy.load("en_core_web_sm")

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20)

    def extract_keywords(self, text: str, top_n: int = 5) -> list:
        """Extract top keywords from text using TF-IDF scoring.

        Args:
            text: Raw input text.
            top_n: Number of top keywords to return.

        Returns:
            A list of keyword strings.
        """
        sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 5]
        if not sentences:
            return []

        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices]
        except ValueError:
            return []

    def extract_entities(self, text: str) -> dict:
        """Perform Named Entity Recognition using spaCy.

        Args:
            text: Raw input text.

        Returns:
            A dict mapping entity label -> list of unique entity strings.
        """
        doc = self.nlp(text)
        entities: dict = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, set()).add(ent.text)
        return {k: list(v) for k, v in entities.items()}


if __name__ == "__main__":
    extractor = FeatureExtractor()
    sample_text = "Alice scheduled a meeting with Google next Monday to discuss the API latency issue."
    print("Keywords:", extractor.extract_keywords(sample_text))
    print("Entities:", extractor.extract_entities(sample_text))
