from rouge_score import rouge_scorer
import pandas as pd

class Evaluator:
    def __init__(self):
        # Initialize scorer for ROUGE-1, ROUGE-2, and ROUGE-L
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_summary(self, reference_summary: str, generated_summary: str) -> dict:
        """
        Evaluate a single generated summary against a reference summary.
        Returns a dictionary with precision, recall, and fmeasure for each metric.
        """
        scores = self.scorer.score(reference_summary, generated_summary)
        
        # Format the output for better readability
        formatted_scores = {}
        for metric, score in scores.items():
            formatted_scores[metric] = {
                'precision': score.precision,
                'recall': score.recall,
                'f1': score.fmeasure
            }
        return formatted_scores

    def evaluate_dataset(self, df: pd.DataFrame, summarizer) -> dict:
        """
        Evaluate a summarizer over an entire dataset (DataFrame).
        Assumes df has 'transcript' and 'summary' columns.
        """
        all_scores = {
            'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
            'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
            'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
        }
        
        num_samples = len(df)
        if num_samples == 0:
            return all_scores

        for index, row in df.iterrows():
            transcript = row['transcript']
            ref_summary = row['summary']
            
            # Generate summary depending on the type of summarizer
            # Assuming summarizer has a summarize(text) method 
            gen_summary = summarizer.summarize(transcript)
            
            scores = self.evaluate_summary(ref_summary, gen_summary)
            
            # Accumulate scores
            for metric in all_scores.keys():
                all_scores[metric]['precision'] += scores[metric]['precision']
                all_scores[metric]['recall'] += scores[metric]['recall']
                all_scores[metric]['f1'] += scores[metric]['f1']

        # Average scores
        for metric in all_scores.keys():
            all_scores[metric]['precision'] /= num_samples
            all_scores[metric]['recall'] /= num_samples
            all_scores[metric]['f1'] /= num_samples

        return all_scores

if __name__ == "__main__":
    evaluator = Evaluator()
    ref = "The team decided to launch next week."
    gen = "Team is launching the product next week."
    print(evaluator.evaluate_summary(ref, gen))
