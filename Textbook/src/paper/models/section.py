"""
Section Model for AI-Native Software Development & Physical AI project
Defines the structure and validation for paper sections
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
from .citation import Citation


class SectionType(Enum):
    """Enumeration of possible section types in a research paper"""
    OVERVIEW = "overview"
    FOUNDATION = "foundation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    APPENDIX = "appendix"


class Section:
    """
    Represents a section within a research paper
    Validates content requirements like plagiarism and citation density
    """

    def __init__(self,
                 title: str,
                 content: str = "",
                 section_type: SectionType = SectionType.OVERVIEW,
                 order: int = 0,
                 readability_score: float = 0.0):
        """
        Initialize a Section with validation

        Args:
            title: The title of the section
            content: The text content of the section
            section_type: Type of section (overview, foundation, analysis, synthesis, appendix)
            order: Order of the section in the paper
            readability_score: Flesch-Kincaid readability score (10.0-12.0 range)
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.content = content
        self.section_type = section_type
        self.order = order
        self.readability_score = readability_score
        self.created_date = datetime.now()
        self.last_modified = datetime.now()
        self.citations = []

        # Validate initial values
        self._validate_readability_score()
        self._validate_plagiarism()

    def _validate_readability_score(self):
        """Validate that readability score is within required range (10.0-12.0)"""
        if self.readability_score != 0.0 and (self.readability_score < 10.0 or self.readability_score > 12.0):
            raise ValueError(
                f"Readability score {self.readability_score} is outside required range of 10.0-12.0 "
                f"(Flesch-Kincaid Grade Level)"
            )

    def _validate_plagiarism(self):
        """
        Validate that content passes plagiarism check (0% tolerance)
        In a real implementation, this would integrate with a plagiarism detection service
        """
        # Placeholder implementation - in real system would check against databases
        # For now, we assume content is original unless proven otherwise
        pass

    def _validate_citation_density(self):
        """
        Validate that content cites at least 1 source per 500 words
        """
        word_count = len(self.content.split())
        required_citations = max(1, word_count // 500)  # At least 1 citation per 500 words

        if len(self.citations) < required_citations:
            raise ValueError(
                f"Section requires at least {required_citations} citations for {word_count} words, "
                f"but only {len(self.citations)} provided"
            )

    def update_content(self, content: str, readability_score: float = None):
        """Update section content and validate requirements"""
        self.content = content
        self.last_modified = datetime.now()

        if readability_score is not None:
            self.readability_score = readability_score
            self._validate_readability_score()

        # Validate citation density based on new content
        self._validate_citation_density()
        self._validate_plagiarism()

    def add_citation(self, citation: Citation):
        """Add a citation to the section"""
        self.citations.append(citation)
        self.last_modified = datetime.now()
        # Re-validate citation density after adding citation
        self._validate_citation_density()

    def calculate_readability_score(self, content: str = None) -> float:
        """
        Calculate Flesch-Kincaid Grade Level for the content
        Formula: 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
        This is a simplified implementation - in practice would use a library like textstat
        """
        text = content or self.content

        if not text.strip():
            return 0.0

        # Simple approximation for demonstration
        # In real implementation, would use proper text analysis
        import re

        # Count words
        words = len(re.findall(r'\b\w+\b', text))

        # Count sentences (approximate)
        sentences = max(1, len(re.split(r'[.!?]+', text)) - 1)

        # This is a placeholder - in a real implementation we'd calculate proper readability
        # For now, return a score that can be overridden by the caller
        return self.readability_score if self.readability_score > 0 else 11.0

    def to_dict(self) -> Dict:
        """Convert the section to a dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type.value,
            "order": self.order,
            "readability_score": self.readability_score,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "citations": [citation.to_dict() for citation in self.citations]
        }