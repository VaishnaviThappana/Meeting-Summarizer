import re
import sys
import subprocess
import nltk
import spacy

# Download NLTK punkt tokenizer data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class ActionItemExtractor:
    """Extracts tasks, decisions, and deadlines from meeting transcripts."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None

        # Regex patterns for rule-based extraction
        self.task_patterns = [
            r"(?i)\b(i will|we will|need to|have to|going to|let's|could you|can you|please)\b(.+)",
            r"(?i)\b(action item:|todo:|to do:)\b(.*)",
        ]
        self.decision_patterns = [
            r"(?i)\b(we decided|agreed that|the plan is|conclusion is)\b(.+)",
            r"(?i)\b(decision:)\b(.*)",
        ]
        # Matches "Name:" or "First Last:" at the start of a line
        self.speaker_pattern = r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*):"

    def extract_action_items(self, text: str) -> dict:
        """
        Extract tasks, decisions, and deadlines from text.

        Args:
            text: Raw meeting transcript string.

        Returns:
            A dict with keys 'tasks', 'decisions', and 'deadlines'.
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            sentences = nltk.sent_tokenize(text)

        action_items: dict = {"tasks": [], "decisions": [], "deadlines": []}
        current_speaker = "Unknown"

        for sentence in sentences:
            # Track the current speaker across sentences
            speaker_match = re.search(self.speaker_pattern, sentence)
            if speaker_match:
                current_speaker = speaker_match.group(1)

            # 1. Task Extraction
            for pattern in self.task_patterns:
                if re.search(pattern, sentence):
                    display_text = sentence.strip()
                    if current_speaker != "Unknown" and current_speaker not in display_text:
                        display_text = f"**{current_speaker}**: {display_text}"
                    action_items["tasks"].append(display_text)
                    break  # Only add once per sentence

            # 2. Decision Extraction
            for pattern in self.decision_patterns:
                if re.search(pattern, sentence):
                    action_items["decisions"].append(sentence.strip())
                    break

            # 3. Deadline Extraction via spaCy NER
            if self.nlp:
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    if ent.label_ in ("DATE", "TIME") and self._is_deadline_context(sentence):
                        action_items["deadlines"].append(
                            {"deadline": ent.text, "context": sentence.strip()}
                        )
                        break  # One deadline per sentence

        return action_items

    def _is_deadline_context(self, sentence: str) -> bool:
        """Return True if the sentence appears to describe a deadline."""
        deadline_keywords = ["by", "before", "due", "deadline", "end of", "eod", "tomorrow", "next"]
        sentence_lower = sentence.lower()
        return any(kw in sentence_lower for kw in deadline_keywords)


if __name__ == "__main__":
    extractor = ActionItemExtractor()
    sample = "Alice: We decided to launch next week. I will finish the API by Friday."
    print(extractor.extract_action_items(sample))
