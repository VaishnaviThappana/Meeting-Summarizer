import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ExtractiveSummarizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def summarize(self, sentences: list[str], top_n: int = 3) -> str:
        """
        Extractive summarization using TF-IDF and Cosine Similarity (TextRank approach).
        """
        if not sentences or len(sentences) <= top_n:
            return " ".join(sentences)

        try:
            # 1. Generate TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # 2. Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # 3. Simple TextRank: score each sentence as sum of its similarities to others
            scores = similarity_matrix.sum(axis=1)
            
            # 4. Get top_n sentence indices
            ranked_indices = np.argsort(scores)[::-1][:top_n]
            
            # 5. Sort by original order to maintain flow
            ranked_indices = sorted(ranked_indices)
            
            summary_sentences = [sentences[i] for i in ranked_indices]
            return " ".join(summary_sentences)
        except Exception as e:
            print(f"Extractive summarization error: {e}")
            return " ".join(sentences[:top_n])


class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        # Load pipeline for easy inference
        try:
            self.summarizer = pipeline("summarization", model=self.model_name, device=self.device)
        except Exception as e:
            print(f"Failed to load abstractive model: {e}")
            self.summarizer = None

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Abstractive summarization using HuggingFace Transformers.
        """
        if not self.summarizer:
            return "Abstractive summarizer is not available."
            
        if not text.strip():
            return ""

        try:
            # Ensure max_length doesn't exceed input length excessively
            input_words = len(text.split())
            adjusted_max = min(max_length, max(min_length + 10, int(input_words * 0.8)))
            adjusted_min = min(min_length, int(adjusted_max * 0.5))

            summary = self.summarizer(text, max_length=adjusted_max, min_length=adjusted_min, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Abstractive summarization error: {e}")
            return ""

if __name__ == "__main__":
    ext = ExtractiveSummarizer()
    text = ["Alice: Hi everyone, thanks for joining the product sync.", 
            "Let's start with updates.", 
            "Bob, how is the new feature coming along?",
            "Bob: I've finished the backend API, but I need to integrate the frontend.", 
            "I will do this by Friday.",
            "Charlie: Great. Once that's done, we need to test it thoroughly.",
            "Alice: Charlie, write the test cases by Wednesday."]
    print("Extractive:", ext.summarize(text))
    
    # Optional testing of abstractive, but it downloads weights, so we leave it to user
    # abs_summ = AbstractiveSummarizer("t5-small") # smaller for test
    # print("Abstractive:", abs_summ.summarize(" ".join(text)))
