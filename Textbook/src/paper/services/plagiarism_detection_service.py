"""
Plagiarism Detection Service for AI-Native Software Development & Physical AI project
Ensures 0% plagiarism tolerance as required by academic standards
"""

import hashlib
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PlagiarismCheckResult:
    """Represents the results of a plagiarism check"""
    is_original: bool
    similarity_percentage: float
    matches_found: int
    matched_sources: List[Dict[str, str]]
    timestamp: datetime
    confidence_level: str


class PlagiarismDetectionService:
    """
    Service to detect plagiarism in research content
    Implements 0% tolerance policy for academic integrity
    """

    def __init__(self):
        """Initialize the plagiarism detection service"""
        self.content_database = {}  # Simulated database of known content
        self.min_similarity_threshold = 0.0  # 0% tolerance
        self.max_similarity_for_warning = 5.0  # Warning threshold

    def check_content_originality(self, content: str, author_id: str = None) -> PlagiarismCheckResult:
        """
        Check the originality of content against known sources

        Args:
            content: The content to check for plagiarism
            author_id: Optional author identifier for tracking

        Returns:
            PlagiarismCheckResult with detection results
        """
        if not content or not content.strip():
            return PlagiarismCheckResult(
                is_original=True,
                similarity_percentage=0.0,
                matches_found=0,
                matched_sources=[],
                timestamp=datetime.now(),
                confidence_level="N/A - No content"
            )

        # Clean the content for comparison
        clean_content = self._normalize_text(content)

        # Calculate document hash for identification
        content_hash = self._generate_content_hash(clean_content)

        # Check for matches in our database
        matches, similarity = self._compare_against_known_content(clean_content)

        # Determine if content is original
        is_original = similarity <= self.min_similarity_threshold

        # Determine confidence level
        confidence_level = self._determine_confidence_level(similarity)

        # Prepare matched sources list
        matched_sources = self._prepare_matched_sources(matches, similarity)

        return PlagiarismCheckResult(
            is_original=is_original,
            similarity_percentage=similarity,
            matches_found=len(matches),
            matched_sources=matched_sources,
            timestamp=datetime.now(),
            confidence_level=confidence_level
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and standardizing formatting"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?,;:]', ' ', text)
        # Convert to lowercase for comparison
        return text.lower().strip()

    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content to identify it uniquely"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _compare_against_known_content(self, content: str) -> Tuple[List[Dict], float]:
        """
        Compare content against known content in the database

        Args:
            content: The normalized content to check

        Returns:
            Tuple of (list of matches, similarity percentage)
        """
        # This is a simplified implementation - in a real system, this would involve:
        # - Querying a large database of academic papers, articles, and other sources
        # - Using advanced NLP techniques for semantic similarity
        # - Checking against multiple content repositories

        # For this implementation, we'll simulate the check
        # In a real implementation, this would be much more sophisticated
        content_words = set(content.split())

        # Simulate finding matches in our database
        matches = []
        max_similarity = 0.0

        # Check against some sample content in our database
        for source_id, source_content in self.content_database.items():
            source_words = set(source_content.split())

            # Calculate overlap
            common_words = content_words.intersection(source_words)
            if len(common_words) > 0:
                similarity = (len(common_words) / max(len(content_words), len(source_words))) * 100

                if similarity > max_similarity:
                    max_similarity = similarity

                    matches.append({
                        "source_id": source_id,
                        "similarity": similarity,
                        "matched_text": " ".join(list(common_words)[:10]),  # Show first 10 matched words
                        "source_title": f"Source {source_id}"
                    })

        return matches, max_similarity

    def _determine_confidence_level(self, similarity: float) -> str:
        """Determine confidence level based on similarity percentage"""
        if similarity <= self.min_similarity_threshold:
            return "High - Original Content"
        elif similarity <= self.max_similarity_for_warning:
            return "Medium - Some Similarity Detected"
        else:
            return "Low - Significant Similarity Detected"

    def _prepare_matched_sources(self, matches: List[Dict], overall_similarity: float) -> List[Dict[str, str]]:
        """Prepare matched sources for the result"""
        return matches

    def add_known_content(self, content: str, source_id: str, title: str = ""):
        """
        Add content to the known content database for future comparisons

        Args:
            content: The content to add
            source_id: Unique identifier for the source
            title: Optional title for the source
        """
        normalized_content = self._normalize_text(content)
        self.content_database[source_id] = normalized_content

    def validate_academic_integrity(self, content: str, author_id: str = None) -> Tuple[bool, List[str]]:
        """
        Validate that content meets academic integrity standards (0% plagiarism tolerance)

        Args:
            content: The content to validate
            author_id: Optional author identifier

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for plagiarism
        result = self.check_content_originality(content, author_id)

        if not result.is_original:
            issues.append(
                f"Content has {result.similarity_percentage:.2f}% similarity to known sources. "
                f"Academic policy requires 0% similarity (original content only)."
            )

        # Additional checks could be added here:
        # - Self-plagiarism detection
        # - Citation verification
        # - Quotation marking validation

        return len(issues) == 0, issues

    def generate_plagiarism_report(self, content: str, author_id: str = None) -> Dict:
        """
        Generate a comprehensive plagiarism report

        Args:
            content: The content to analyze
            author_id: Optional author identifier

        Returns:
            Dictionary with detailed plagiarism analysis
        """
        check_result = self.check_content_originality(content, author_id)

        return {
            "content_hash": self._generate_content_hash(self._normalize_text(content)),
            "timestamp": check_result.timestamp.isoformat(),
            "originality_check": {
                "is_original": check_result.is_original,
                "similarity_percentage": check_result.similarity_percentage,
                "confidence_level": check_result.confidence_level
            },
            "matches": {
                "total_matches_found": check_result.matches_found,
                "matched_sources": check_result.matched_sources
            },
            "validation": {
                "meets_academic_standards": check_result.similarity_percentage <= self.min_similarity_threshold,
                "required_standard": f"< {self.min_similarity_threshold}% similarity",
                "status": "PASS" if check_result.is_original else "FAIL"
            }
        }

    def batch_check_content(self, contents: List[str]) -> List[PlagiarismCheckResult]:
        """
        Batch check multiple content items for plagiarism

        Args:
            contents: List of content strings to check

        Returns:
            List of plagiarism check results
        """
        results = []
        for content in contents:
            result = self.check_content_originality(content)
            results.append(result)
        return results

    def get_integrity_statistics(self) -> Dict[str, int]:
        """
        Get statistics about content integrity in the system

        Returns:
            Dictionary with integrity statistics
        """
        total_sources = len(self.content_database)

        # This would be more detailed in a real implementation
        return {
            "total_known_sources": total_sources,
            "estimated_unique_content_ratio": 100 if total_sources == 0 else 75,  # Placeholder
            "last_update": datetime.now().isoformat()
        }


# Example usage and testing
def test_plagiarism_detection_service():
    """Test the plagiarism detection service with sample content"""
    service = PlagiarismDetectionService()

    # Add some sample content to the database
    sample_source = """
    Physical artificial intelligence represents a paradigm shift in robotics research.
    Rather than treating AI as merely a controller for mechanical systems,
    Physical AI systems embody intelligence in physical form. This approach
    acknowledges that intelligence emerges through interaction with the physical
    environment. The field encompasses humanoid robotics, where machines exhibit
    human-like behaviors and capabilities.
    """

    service.add_known_content(sample_source, "source_001", "Physical AI Research Paper")

    # Test with original content
    original_content = """
    Artificial intelligence in physical form is transforming robotics research.
    Modern approaches recognize that intelligence develops through interaction
    with the real world. These systems demonstrate advanced capabilities in
    humanoid robotics applications.
    """

    print("Testing Original Content:")
    result = service.check_content_originality(original_content)
    print(f"Is Original: {result.is_original}")
    print(f"Similarity: {result.similarity_percentage}%")
    print(f"Matches Found: {result.matches_found}")
    print(f"Confidence: {result.confidence_level}")
    print()

    # Test with potentially plagiarized content
    similar_content = """
    Physical artificial intelligence represents a paradigm shift in robotics research.
    Rather than treating AI as merely a controller for mechanical systems,
    Physical AI systems embody intelligence in physical form. This approach
    acknowledges that intelligence emerges through interaction with the physical
    environment. The field encompasses humanoid robotics, where machines exhibit
    human-like behaviors and capabilities.
    """

    print("Testing Potentially Plagiarized Content:")
    result = service.check_content_originality(similar_content)
    print(f"Is Original: {result.is_original}")
    print(f"Similarity: {result.similarity_percentage}%")
    print(f"Matches Found: {result.matches_found}")
    print(f"Confidence: {result.confidence_level}")
    if result.matched_sources:
        print("Matched Sources:")
        for match in result.matched_sources:
            print(f"  - {match['source_title']}: {match['similarity']:.2f}% similarity")
    print()

    # Test validation
    is_valid, issues = service.validate_academic_integrity(similar_content)
    print("Validation Results:")
    print(f"Meets Academic Standards: {is_valid}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    print()

    # Generate report
    report = service.generate_plagiarism_report(similar_content)
    print("Plagiarism Report:")
    print(f"Content Hash: {report['content_hash'][:10]}...")
    print(f"Originality: {report['originality_check']['is_original']}")
    print(f"Status: {report['validation']['status']}")


if __name__ == "__main__":
    test_plagiarism_detection_service()