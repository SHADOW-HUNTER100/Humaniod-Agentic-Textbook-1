"""
Citation Validation Service for AI-Native Software Development & Physical AI project
Ensures all citations follow APA 7th edition format and academic standards
"""

from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
from ..models.citation import Citation, SourceType


class CitationValidationService:
    """
    Service to validate citations according to APA 7th edition standards
    Performs comprehensive checks on citation format, completeness, and academic compliance
    """

    def __init__(self):
        """Initialize the citation validation service"""
        self.apa_7th_patterns = self._load_apa_7th_patterns()

    def _load_apa_7th_patterns(self) -> Dict[str, re.Pattern]:
        """
        Load regex patterns for different types of APA 7th edition citations
        These are simplified patterns - a full implementation would be more comprehensive
        """
        patterns = {
            # Journal article pattern: Author, A. A. (Year). Title. Journal Name, Volume(Issue), pages. DOI/URL
            "journal": re.compile(
                r'^\s*[A-Z][a-z]+,\s*[A-Z]\.(\s*[A-Z]\.)?\s*\([^)]{4}\)\.\s*.+?\.\s*[A-Z][a-zA-Z\s&-]+,\s*\d+\(\d+\),\s*\d+-\d+\.'
            ),

            # Book pattern: Author, A. A. (Year). Title. Publisher.
            "book": re.compile(
                r'^\s*[A-Z][a-z]+,\s*[A-Z]\.(\s*[A-Z]\.)?\s*\([^)]{4}\)\.\s*.+?\.\s*[A-Z][a-zA-Z\s-]+\.'
            ),

            # Conference paper pattern: Author, A. A. (Year). Title. Conference Name, pages.
            "conference": re.compile(
                r'^\s*[A-Z][a-z]+,\s*[A-Z]\.(\s*[A-Z]\.)?\s*\([^)]{4}\)\.\s*.+?\.\s*[A-Z][a-zA-Z\s&-]+,'
            ),

            # Website pattern: Author, A. A. (Year, Month Day). Title. URL
            "web": re.compile(
                r'^\s*[A-Z][a-z]+,\s*[A-Z]\.(\s*[A-Z]\.)?\s*\([^)]{4},\s*[A-Z][a-z]+\s+\d+\)\.\s*.+?\.\s*https?://'
            )
        }

        return patterns

    def validate_citation_format(self, citation: Citation) -> Tuple[bool, List[str]]:
        """
        Validate that a citation follows APA 7th edition format

        Args:
            citation: The citation to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check that citation text is not empty
        if not citation.citation_text or not citation.citation_text.strip():
            errors.append("Citation text cannot be empty")
            return False, errors

        # Determine expected pattern based on source type
        source_type = citation.source_type.value
        expected_pattern = self._get_expected_pattern(source_type)

        if expected_pattern and not expected_pattern.search(citation.citation_text):
            errors.append(f"Citation does not match expected APA 7th edition format for {source_type}")

        # Additional validation checks
        if not self._has_valid_author_format(citation.citation_text):
            errors.append("Citation does not follow proper author format (Author, A. A.)")

        if not self._has_valid_date_format(citation.citation_text):
            errors.append("Citation does not follow proper date format (Year in parentheses)")

        if not self._has_valid_punctuation(citation.citation_text):
            errors.append("Citation is missing required punctuation (. after title, etc.)")

        return len(errors) == 0, errors

    def _get_expected_pattern(self, source_type: str) -> re.Pattern:
        """Get the expected pattern based on source type"""
        # Map source type to pattern
        type_to_pattern = {
            SourceType.JOURNAL.value: self.apa_7th_patterns.get("journal"),
            SourceType.CONFERENCE.value: self.apa_7th_patterns.get("conference"),
            SourceType.TECHNICAL_REPORT.value: self.apa_7th_patterns.get("journal"),
            SourceType.PREPRINT.value: self.apa_7th_patterns.get("journal"),
            SourceType.WHITEPAPER.value: self.apa_7th_patterns.get("web")
        }

        return type_to_pattern.get(source_type)

    def _has_valid_author_format(self, citation_text: str) -> bool:
        """Check if citation has valid author format (Author, A. A.)"""
        # Look for pattern: Lastname, A. A.
        author_pattern = re.compile(r'[A-Z][a-z]+,\s*[A-Z]\.(\s*[A-Z]\.)?')
        return bool(author_pattern.search(citation_text))

    def _has_valid_date_format(self, citation_text: str) -> bool:
        """Check if citation has valid date format (Year in parentheses)"""
        # Look for pattern: (Year) or (Year, Month Day)
        date_pattern = re.compile(r'\(\d{4}([^\)]*)?\)')
        return bool(date_pattern.search(citation_text))

    def _has_valid_punctuation(self, citation_text: str) -> bool:
        """Check if citation has proper APA punctuation"""
        # Check for period at the end
        has_period_at_end = citation_text.strip().endswith('.')

        # Check for period after author initial(s)
        has_period_after_initial = '.' in citation_text

        return has_period_at_end and has_period_after_initial

    def validate_peer_reviewed_status(self, citation: Citation) -> Tuple[bool, List[str]]:
        """
        Validate that peer-reviewed status is appropriate for the source type

        Args:
            citation: The citation to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Certain source types are typically peer-reviewed
        typically_peer_reviewed = [
            SourceType.JOURNAL.value,
            SourceType.CONFERENCE.value,
            SourceType.TECHNICAL_REPORT.value
        ]

        # Certain source types are typically not peer-reviewed
        typically_not_peer_reviewed = [
            SourceType.PREPRINT.value,
            SourceType.WHITEPAPER.value
        ]

        if citation.source_type.value in typically_peer_reviewed and not citation.is_peer_reviewed:
            warnings.append(
                f"Source type '{citation.source_type.value}' is typically peer-reviewed, "
                f"but 'is_peer_reviewed' is set to False. Please verify this is correct."
            )

        if citation.source_type.value in typically_not_peer_reviewed and citation.is_peer_reviewed:
            warnings.append(
                f"Source type '{citation.source_type.value}' is typically not peer-reviewed, "
                f"but 'is_peer_reviewed' is set to True. Please verify this is correct."
            )

        return True, warnings  # These are warnings, not errors, so always return True

    def validate_citation_completeness(self, citation: Citation) -> Tuple[bool, List[str]]:
        """
        Validate that the citation has all required information

        Args:
            citation: The citation to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check that required fields are populated
        if not citation.title or not citation.title.strip():
            errors.append("Citation title is required")

        if not citation.authors or len(citation.authors) == 0:
            errors.append("Citation must have at least one author")

        if not citation.publication_date:
            errors.append("Citation must have a publication date")

        # Check for DOI or URL if needed
        if citation.source_type in [SourceType.JOURNAL, SourceType.CONFERENCE] and not citation.doi and not citation.url:
            errors.append(f"Citation of type {citation.source_type.value} should have a DOI or URL")

        return len(errors) == 0, errors

    def validate_all_citations(self, citations: List[Citation]) -> Dict[str, Any]:
        """
        Validate a list of citations and return comprehensive results

        Args:
            citations: List of citations to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "total_citations": len(citations),
            "valid_citations": 0,
            "invalid_citations": 0,
            "errors": [],
            "warnings": [],
            "compliance_percentage": 0.0
        }

        for i, citation in enumerate(citations):
            # Validate format
            format_valid, format_errors = self.validate_citation_format(citation)

            # Validate completeness
            completeness_valid, completeness_errors = self.validate_citation_completeness(citation)

            # Validate peer-reviewed status (returns warnings, not errors)
            _, peer_review_warnings = self.validate_peer_reviewed_status(citation)

            # Combine results
            all_errors = format_errors + completeness_errors
            all_warnings = peer_review_warnings

            if all_errors:
                results["errors"].extend([f"Citation {i+1} ({citation.title}): {error}" for error in all_errors])
                results["invalid_citations"] += 1
            else:
                results["valid_citations"] += 1

            if all_warnings:
                results["warnings"].extend([f"Citation {i+1} ({citation.title}): {warning}" for warning in all_warnings])

        # Calculate compliance percentage
        if results["total_citations"] > 0:
            results["compliance_percentage"] = (results["valid_citations"] / results["total_citations"]) * 100

        return results

    def validate_citation_list_compliance(self, citations: List[Citation], min_peer_reviewed_percentage: float = 50.0) -> Tuple[bool, List[str]]:
        """
        Validate that the citation list meets academic requirements

        Args:
            citations: List of citations to validate
            min_peer_reviewed_percentage: Minimum percentage of peer-reviewed sources (default 50%)

        Returns:
            Tuple of (is_compliant, list_of_errors)
        """
        errors = []

        total_citations = len(citations)
        if total_citations < 15:
            errors.append(f"Minimum 15 sources required, only {total_citations} provided")

        peer_reviewed_count = sum(1 for citation in citations if citation.is_peer_reviewed)
        if total_citations > 0:
            peer_reviewed_percentage = (peer_reviewed_count / total_citations) * 100
            if peer_reviewed_percentage < min_peer_reviewed_percentage:
                errors.append(
                    f"Minimum {min_peer_reviewed_percentage}% peer-reviewed sources required, "
                    f"only {peer_reviewed_percentage:.1f}% provided ({peer_reviewed_count}/{total_citations})"
                )

        return len(errors) == 0, errors


# Example usage and testing
def test_citation_validation_service():
    """Test the citation validation service with sample citations"""
    from datetime import date

    service = CitationValidationService()

    # Sample citations
    sample_citations = [
        Citation(
            source_type=SourceType.JOURNAL,
            title="Sample Journal Article",
            authors=["Smith, J. A.", "Johnson, B. C."],
            publication_date=date(2023, 5, 15),
            citation_text="Smith, J. A., & Johnson, B. C. (2023). Sample journal article. Journal of Examples, 45(2), 123-145.",
            is_peer_reviewed=True
        ),
        Citation(
            source_type=SourceType.CONFERENCE,
            title="Sample Conference Paper",
            authors=["Brown, K. L."],
            publication_date=date(2022, 10, 10),
            citation_text="Brown, K. L. (2022). Sample conference paper. Proceedings of the International Conference on Examples, 67-72.",
            is_peer_reviewed=True
        )
    ]

    # Test format validation
    for citation in sample_citations:
        is_valid, errors = service.validate_citation_format(citation)
        print(f"Citation: {citation.title}")
        print(f"Valid: {is_valid}")
        if errors:
            print(f"Errors: {errors}")
        print("---")

    # Test list compliance
    is_compliant, compliance_errors = service.validate_citation_list_compliance(sample_citations)
    print(f"All Citations Compliant: {is_compliant}")
    if compliance_errors:
        print(f"Compliance Errors: {compliance_errors}")


if __name__ == "__main__":
    test_citation_validation_service()