import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FeatureExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model is not downloaded
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=20)

    def extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """Extract top keywords using TF-IDF."""
        # TF-IDF requires an iterable of documents. We will treat sentences as documents.
        # If input is a single string, we split by basic punctuation.
        import re
        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 5]
        
        if not sentences:
            return []

        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Sum tfidf scores across all sentences
            scores = tfidf_matrix.sum(axis=0).A1
            
            # Sort by score
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices]
        except ValueError:
            return []

    def extract_entities(self, text: str) -> dict:
        """Perform NER using spaCy and return entities grouped by label."""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
            
        # Convert sets to lists for JSON serialization if needed
        return {k: list(v) for k, v in entities.items()}

if __name__ == "__main__":
    extractor = FeatureExtractor()
    sample_text = "Alice scheduled a meeting with Google next Monday to discuss the API latency issue."
    print("Keywords:", extractor.extract_keywords(sample_text))
    print("Entities:", extractor.extract_entities(sample_text))
