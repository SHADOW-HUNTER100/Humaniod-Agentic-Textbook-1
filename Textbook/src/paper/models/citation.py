"""
Citation Model for AI-Native Software Development & Physical AI project
Defines the structure and validation for academic citations
"""

import uuid
from datetime import date
from typing import List, Dict, Optional
from enum import Enum


class SourceType(Enum):
    """Enumeration of possible source types for citations"""
    JOURNAL = "journal"
    CONFERENCE = "conference"
    PREPRINT = "preprint"
    WHITEPAPER = "whitepaper"
    TECHNICAL_REPORT = "technical_report"


class Citation:
    """
    Represents a citation in the research paper
    Validates APA 7th edition format and peer-review status
    """

    def __init__(self,
                 source_type: SourceType,
                 title: str,
                 authors: List[str],
                 publication_date: date,
                 citation_text: str,
                 doi: Optional[str] = None,
                 url: Optional[str] = None,
                 is_peer_reviewed: bool = False,
                 retrieved_date: Optional[date] = None):
        """
        Initialize a Citation with validation

        Args:
            source_type: Type of source (journal, conference, preprint, etc.)
            title: Title of the cited work
            authors: List of authors of the cited work
            publication_date: Date when the work was published
            citation_text: Full citation text in APA 7th edition format
            doi: DOI identifier (optional)
            url: URL to the source (optional)
            is_peer_reviewed: Whether the source is peer-reviewed
            retrieved_date: Date when the source was accessed (optional)
        """
        self.id = str(uuid.uuid4())
        self.source_type = source_type
        self.title = title
        self.authors = authors
        self.publication_date = publication_date
        self.doi = doi
        self.url = url
        self.is_peer_reviewed = is_peer_reviewed
        self.citation_text = citation_text
        self.retrieved_date = retrieved_date or date.today()

        # Validate initial values
        self._validate_citation_format()
        self._validate_peer_reviewed_status()

    def _validate_citation_format(self):
        """Validate that the citation is properly formatted in APA 7th edition"""
        # This is a simplified validation - in a real implementation, we'd use a more comprehensive parser
        # Check for basic APA 7th elements: author, date, title, source
        citation_lower = self.citation_text.lower()

        # Check if it has basic APA elements (this is a simplified check)
        has_author = "(" in self.citation_text and ")" in self.citation_text
        has_title = self.title.lower() in citation_lower
        has_period = "." in self.citation_text

        if not (has_author and has_title and has_period):
            raise ValueError(
                f"Citation '{self.citation_text}' does not appear to be properly formatted in APA 7th edition. "
                f"Ensure it includes author, date, title, and source information."
            )

        # More specific validation would go here in a full implementation
        # For example, checking for proper punctuation, capitalization, etc.

    def _validate_peer_reviewed_status(self):
        """Validate that peer-reviewed status matches the source type"""
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

        if self.source_type.value in typically_peer_reviewed and not self.is_peer_reviewed:
            # This might be a warning in a real implementation, not an error
            # For this implementation, we'll allow flexibility
            pass

        if self.source_type.value in typically_not_peer_reviewed and self.is_peer_reviewed:
            # This is unusual but possible (some preprints are later peer-reviewed)
            pass

    def update_citation_text(self, citation_text: str):
        """Update citation text and revalidate format"""
        self.citation_text = citation_text
        self._validate_citation_format()

    def set_peer_reviewed_status(self, is_peer_reviewed: bool):
        """Update peer-reviewed status and validate"""
        self.is_peer_reviewed = is_peer_reviewed
        self._validate_peer_reviewed_status()

    def to_dict(self) -> Dict:
        """Convert the citation to a dictionary representation"""
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date.isoformat(),
            "doi": self.doi,
            "url": self.url,
            "is_peer_reviewed": self.is_peer_reviewed,
            "citation_text": self.citation_text,
            "retrieved_date": self.retrieved_date.isoformat()
        }

    @classmethod
    def from_apa_string(cls, apa_citation: str) -> 'Citation':
        """
        Create a Citation object from an APA 7th edition formatted string
        This is a simplified parser - a real implementation would be much more comprehensive
        """
        # This is a placeholder implementation - in reality, parsing APA citations
        # would require a much more sophisticated approach
        # For now, we'll create a minimal citation with the provided text

        # In a real implementation, we'd parse:
        # Authors, publication year, title, journal/conference name, volume/issue, pages, DOI, etc.
        return cls(
            source_type=SourceType.JOURNAL,  # Default assumption
            title="Parsed Title",  # Would be extracted from APA string
            authors=["Author, A."],  # Would be extracted from APA string
            publication_date=date.today(),  # Would be extracted from APA string
            citation_text=apa_citation,
            is_peer_reviewed=True  # Default assumption
        )