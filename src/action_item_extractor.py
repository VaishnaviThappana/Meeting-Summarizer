import re
import spacy

class ActionItemExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None

        # Regex patterns for rule-based extraction
        self.task_patterns = [
            r"(?i)\b(i will|we will|need to|have to|going to|let's|could you|can you|please)\b(.+)",
            r"(?i)\b(action item:|todo:|to do:)\b(.*)"
        ]
        
        self.decision_patterns = [
            r"(?i)\b(we decided|agreed that|the plan is|conclusion is)\b(.+)",
            r"(?i)\b(decision:)\b(.*)"
        ]

    def extract_action_items(self, text: str) -> dict:
        """
        Extract tasks, decisions, and deadlines from text.
        Returns a dictionary with lists of strings for each category.
        """
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            sentences = nltk.sent_tokenize(text)

        action_items = {
            "tasks": [],
            "decisions": [],
            "deadlines": []
        }

        for sentence in sentences:
            # 1. Rule-based Task Extraction
            for pattern in self.task_patterns:
                match = re.search(pattern, sentence)
                if match:
                    action_items["tasks"].append(sentence.strip())
                    break # Avoid duplicate task appending

            # 2. Rule-based Decision Extraction
            for pattern in self.decision_patterns:
                match = re.search(pattern, sentence)
                if match:
                    action_items["decisions"].append(sentence.strip())
                    break

            # 3. Deadline Extraction using spaCy NER (looking for DATE or TIME)
            if self.nlp:
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    if ent.label_ in ["DATE", "TIME"] and self._is_deadline_context(sentence):
                        action_items["deadlines"].append({
                            "deadline": ent.text,
                            "context": sentence.strip()
                        })
                        break # One deadline record per sentence is usually enough
                        
        return action_items

    def _is_deadline_context(self, sentence: str) -> bool:
        """Helper to check if a sentence containing a date/time looks like a deadline."""
        deadline_keywords = ["by", "before", "due", "deadline", "end of", "eod", "tomorrow", "next"]
        sentence_lower = sentence.lower()
        return any(keyword in sentence_lower for keyword in deadline_keywords)

if __name__ == "__main__":
    extractor = ActionItemExtractor()
    sample = "We decided to launch the product next week. I will finish the API by Friday."
    print(extractor.extract_action_items(sample))
