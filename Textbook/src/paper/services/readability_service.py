"""
Readability Service for AI-Native Software Development & Physical AI project
Calculates Flesch-Kincaid Grade Level scores to ensure content meets academic standards
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ReadabilityResult:
    """Represents the results of a readability analysis"""
    flesch_kincaid_grade_level: float
    flesch_reading_ease: float
    word_count: int
    sentence_count: int
    syllable_count: int
    score_category: str
    is_within_target_range: bool


class ReadabilityService:
    """
    Service to calculate readability scores using Flesch-Kincaid formulas
    Ensures content meets academic standards (Grade Level 10-12)
    """

    def __init__(self):
        """Initialize the readability service"""
        pass

    def calculate_readability(self, text: str) -> ReadabilityResult:
        """
        Calculate Flesch-Kincaid readability scores for the given text

        Args:
            text: The text to analyze

        Returns:
            ReadabilityResult with various readability metrics
        """
        if not text or not text.strip():
            return ReadabilityResult(
                flesch_kincaid_grade_level=0.0,
                flesch_reading_ease=0.0,
                word_count=0,
                sentence_count=0,
                syllable_count=0,
                score_category="No content",
                is_within_target_range=False
            )

        # Clean and normalize the text
        clean_text = self._clean_text(text)

        # Calculate components
        word_count = self._count_words(clean_text)
        sentence_count = self._count_sentences(clean_text)
        syllable_count = self._count_syllables(clean_text)

        # Calculate Flesch Reading Ease
        flesch_reading_ease = self._calculate_flesch_reading_ease(word_count, sentence_count, syllable_count)

        # Calculate Flesch-Kincaid Grade Level
        flesch_kincaid_grade_level = self._calculate_flesch_kincaid_grade_level(word_count, sentence_count, syllable_count)

        # Determine category
        score_category = self._get_score_category(flesch_kincaid_grade_level)

        # Check if within target range (Grade Level 10-12)
        is_within_target_range = 10.0 <= flesch_kincaid_grade_level <= 12.0

        return ReadabilityResult(
            flesch_kincaid_grade_level=flesch_kincaid_grade_level,
            flesch_reading_ease=flesch_reading_ease,
            word_count=word_count,
            sentence_count=sentence_count,
            syllable_count=syllable_count,
            score_category=score_category,
            is_within_target_range=is_within_target_range
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for readability analysis"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence-ending punctuation
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        return text.strip()

    def _count_words(self, text: str) -> int:
        """Count the number of words in the text"""
        if not text:
            return 0

        # Split on whitespace and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        return len(words)

    def _count_sentences(self, text: str) -> int:
        """Count the number of sentences in the text"""
        if not text:
            return 0

        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings
        sentences = [s for s in sentences if s.strip()]
        return len(sentences)

    def _count_syllables(self, text: str) -> int:
        """Count the number of syllables in the text"""
        if not text:
            return 0

        # Split text into words
        words = text.split()
        total_syllables = 0

        for word in words:
            syllables = self._count_syllables_in_word(word)
            total_syllables += syllables

        return total_syllables

    def _count_syllables_in_word(self, word: str) -> int:
        """Count the number of syllables in a single word"""
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-zA-Z]', '', word)

        if not word:
            return 0

        # Count vowel groups
        vowels = 'aeiouAEIOU'
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e' at the end
        if word.endswith(('e', 'E')) and syllable_count > 1:
            syllable_count -= 1

        # Every word has at least one syllable
        return max(1, syllable_count)

    def _calculate_flesch_reading_ease(self, word_count: int, sentence_count: int, syllable_count: int) -> float:
        """
        Calculate Flesch Reading Ease score
        Formula: 206.835 - (1.015 × average words per sentence) - (84.6 × average syllables per word)
        """
        if sentence_count == 0 or word_count == 0:
            return 0.0

        avg_words_per_sentence = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count

        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        return round(score, 2)

    def _calculate_flesch_kincaid_grade_level(self, word_count: int, sentence_count: int, syllable_count: int) -> float:
        """
        Calculate Flesch-Kincaid Grade Level
        Formula: (0.39 × average words per sentence) + (11.8 × average syllables per word) - 15.59
        """
        if sentence_count == 0 or word_count == 0:
            return 0.0

        avg_words_per_sentence = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count

        score = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
        return round(score, 2)

    def _get_score_category(self, grade_level: float) -> str:
        """Get the readability category based on grade level"""
        if grade_level < 1:
            return "Below 1st Grade"
        elif grade_level < 4:
            return "4th Grade or Below"
        elif grade_level < 6:
            return "5th-6th Grade"
        elif grade_level < 8:
            return "7th-8th Grade"
        elif grade_level < 10:
            return "9th-10th Grade"
        elif grade_level < 12:
            return "11th-12th Grade"
        elif grade_level < 16:
            return "College Level"
        else:
            return "College Graduate Level"

    def validate_readability_range(self, text: str, min_grade: float = 10.0, max_grade: float = 12.0) -> Tuple[bool, ReadabilityResult, List[str]]:
        """
        Validate that the text falls within the specified readability range

        Args:
            text: The text to validate
            min_grade: Minimum acceptable grade level (default 10.0)
            max_grade: Maximum acceptable grade level (default 12.0)

        Returns:
            Tuple of (is_valid, readability_result, list_of_issues)
        """
        result = self.calculate_readability(text)
        issues = []

        if result.flesch_kincaid_grade_level < min_grade:
            issues.append(
                f"Text is too easy to read (Grade Level: {result.flesch_kincaid_grade_level}). "
                f"Minimum Grade Level {min_grade} required for academic audience."
            )

        if result.flesch_kincaid_grade_level > max_grade:
            issues.append(
                f"Text is too difficult to read (Grade Level: {result.flesch_kincaid_grade_level}). "
                f"Maximum Grade Level {max_grade} allowed for academic audience."
            )

        return len(issues) == 0, result, issues

    def analyze_text_structure(self, text: str) -> Dict[str, int]:
        """
        Analyze the structural components of the text

        Args:
            text: The text to analyze

        Returns:
            Dictionary with structural statistics
        """
        result = self.calculate_readability(text)

        return {
            "words": result.word_count,
            "sentences": result.sentence_count,
            "syllables": result.syllable_count,
            "average_words_per_sentence": round(result.word_count / max(1, result.sentence_count), 2),
            "average_syllables_per_word": round(result.syllable_count / max(1, result.word_count), 2)
        }

    def suggest_improvements(self, text: str) -> List[str]:
        """
        Suggest improvements to bring the text within the target readability range

        Args:
            text: The text to analyze

        Returns:
            List of suggestions for improving readability
        """
        result = self.calculate_readability(text)
        suggestions = []

        if result.flesch_kincaid_grade_level > 12.0:
            # Text is too complex - suggest simplification
            suggestions.append("Consider simplifying complex sentences to improve readability.")
            suggestions.append("Try using shorter sentences with fewer clauses.")
            suggestions.append("Consider replacing complex words with simpler alternatives.")

        elif result.flesch_kincaid_grade_level < 10.0:
            # Text is too simple - suggest enhancement
            suggestions.append("Consider adding more technical terminology to increase academic rigor.")
            suggestions.append("Try using more complex sentence structures to match academic standards.")
            suggestions.append("Consider expanding explanations to add depth.")

        # Additional suggestions based on structure
        structure = self.analyze_text_structure(text)
        if structure["average_words_per_sentence"] > 20:
            suggestions.append("Sentences are quite long on average. Consider breaking them into shorter segments.")

        return suggestions


# Example usage and testing
def test_readability_service():
    """Test the readability service with sample texts"""
    service = ReadabilityService()

    # Sample academic text
    academic_text = """
    Physical artificial intelligence represents a paradigm shift in robotics research.
    Rather than treating AI as merely a controller for mechanical systems,
    Physical AI systems embody intelligence in physical form. This approach
    acknowledges that intelligence emerges through interaction with the physical
    environment. The field encompasses humanoid robotics, where machines exhibit
    human-like behaviors and capabilities. These systems require sophisticated
    integration of perception, cognition, and action. The development process
    involves complex considerations of embodiment, where the physical form
    influences cognitive processes. Researchers in this domain must address
    numerous challenges related to sim-to-real transfer, where models trained
    in simulation environments must function effectively in real-world scenarios.
    """

    # Test readability calculation
    result = service.calculate_readability(academic_text)
    print("Readability Analysis:")
    print(f"Flesch-Kincaid Grade Level: {result.flesch_kincaid_grade_level}")
    print(f"Flesch Reading Ease: {result.flesch_reading_ease}")
    print(f"Word Count: {result.word_count}")
    print(f"Sentence Count: {result.sentence_count}")
    print(f"Syllable Count: {result.syllable_count}")
    print(f"Category: {result.score_category}")
    print(f"Within Target Range (10-12): {result.is_within_target_range}")
    print()

    # Test validation
    is_valid, validation_result, issues = service.validate_readability_range(academic_text)
    print("Validation Results:")
    print(f"Is Valid: {is_valid}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues found!")
    print()

    # Test suggestions
    suggestions = service.suggest_improvements(academic_text)
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")


if __name__ == "__main__":
    test_readability_service()