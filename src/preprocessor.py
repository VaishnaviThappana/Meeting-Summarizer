import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# We assume 'nltk' data is already downloaded (punkt, stopwords)
# Fallback downloading inside script just in case
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally", "yeah", "hmm", "ah"}

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')).union(FILLER_WORDS)

    def clean_text(self, text: str) -> str:
        """
        Normalizes text, removes punctuation, and removes filler words.
        Returns a single cleaned string.
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation but keep basic sentence enders for context if needed, 
        # though for strict cleaning we usually remove all non-alphanumeric.
        # Let's remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)

        # Tokenize words to remove filler words easily
        words = word_tokenize(text)
        cleaned_words = [word for word in words if word not in FILLER_WORDS]
        
        # Rejoin text
        return ' '.join(cleaned_words)

    def tokenize_sentences(self, text: str) -> list[str]:
        """Tokenize text into sentences."""
        return sent_tokenize(text)

    def tokenize_words(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def process_for_extractive(self, text: str) -> list[str]:
        """
        Returns a list of original sentences, and a parallel list of cleaned sentences 
        which can be used for feature extraction (like TF-IDF).
        """
        sentences = self.tokenize_sentences(text)
        cleaned_sentences = [self.clean_text(s) for s in sentences]
        return sentences, cleaned_sentences

if __name__ == "__main__":
    p = Preprocessor()
    sample = "Alice: Uh, yeah, basically we need to, like, deploy this tomorrow!"
    print("Cleaned:", p.clean_text(sample))
