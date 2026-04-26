import warnings
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# BART can only handle ~1024 tokens. Truncate input to avoid silent failures.
_BART_MAX_WORDS = 900


class ExtractiveSummarizer:
    """Extractive summarizer using TF-IDF + cosine similarity (TextRank-style)."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def summarize(self, sentences: list, top_n: int = 3) -> str:
        """Select the most representative sentences from the transcript.

        Args:
            sentences: List of sentence strings.
            top_n: Number of sentences to include in the summary.

        Returns:
            A summary string made up of the top-ranked sentences.
        """
        if not sentences:
            return ""
        if len(sentences) <= top_n:
            return " ".join(sentences)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            scores = similarity_matrix.sum(axis=1)
            ranked_indices = np.argsort(scores)[::-1][:top_n]
            # Restore original sentence order for readability
            ranked_indices = sorted(ranked_indices)
            return " ".join(sentences[i] for i in ranked_indices)
        except Exception as e:
            print(f"Extractive summarization error: {e}")
            return " ".join(sentences[:top_n])


class AbstractiveSummarizer:
    """Abstractive summarizer backed by a HuggingFace seq2seq model (BART by default)."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.summarizer = pipeline("summarization", model=model_name, device=device)
        except Exception as e:
            print(f"Failed to load abstractive model '{model_name}': {e}")
            self.summarizer = None

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """Generate an abstractive summary of the input text.

        Args:
            text: The full transcript text.
            max_length: Maximum number of tokens in the output.
            min_length: Minimum number of tokens in the output.

        Returns:
            A human-readable summary string.
        """
        if not self.summarizer:
            return "Abstractive summarizer is not available."
        if not text.strip():
            return ""

        # Truncate to avoid exceeding BART's token limit
        words = text.split()
        if len(words) > _BART_MAX_WORDS:
            text = " ".join(words[:_BART_MAX_WORDS])

        try:
            input_words = len(text.split())
            # Ensure adjusted lengths are sane
            adjusted_max = min(max_length, max(min_length + 10, int(input_words * 0.6)))
            adjusted_min = min(min_length, int(adjusted_max * 0.5))

            result = self.summarizer(
                text,
                max_length=adjusted_max,
                min_length=adjusted_min,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"]
        except Exception as e:
            print(f"Abstractive summarization error: {e}")
            return ""


if __name__ == "__main__":
    ext = ExtractiveSummarizer()
    sample_sentences = [
        "Alice: Hi everyone, thanks for joining the product sync.",
        "Bob: I've finished the backend API, but I need to integrate the frontend.",
        "Charlie: Once that's done, we need to test it thoroughly.",
        "Alice: Charlie, can you write the test cases by Wednesday?",
    ]
    print("Extractive:", ext.summarize(sample_sentences))
