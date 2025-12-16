"""
Research Paper Model for AI-Native Software Development & Physical AI project
Defines the structure and validation for research papers in the Physical AI domain
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
from .citation import Citation
from .author import Author


class PaperStatus(Enum):
    """Enumeration of possible statuses for a research paper"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"


class ResearchPaper:
    """
    Represents a research paper in the Physical AI domain
    Validates academic requirements like word count, citation count, and readability
    """

    def __init__(self,
                 title: str,
                 abstract: str = "",
                 word_count: int = 0,
                 status: PaperStatus = PaperStatus.DRAFT,
                 authors: List[Author] = None,
                 citations: List[Citation] = None):
        """
        Initialize a Research Paper with validation

        Args:
            title: The title of the research paper
            abstract: Brief summary of the paper (max 5000 chars)
            word_count: Current word count of the paper (5000-7000 range)
            status: Current status of the paper (draft, review, approved, published)
            authors: List of authors contributing to the paper
            citations: List of citations used in the paper
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.abstract = abstract
        self.word_count = word_count
        self.status = status
        self.created_date = datetime.now()
        self.last_modified = datetime.now()
        self.authors = authors or []
        self.citations = citations or []

        # Validate initial values
        self._validate_word_count()
        self._validate_abstract_length()
        self._validate_citation_requirements()

    def _validate_word_count(self):
        """Validate that word count is within required range (5000-7000)"""
        if self.word_count < 5000 or self.word_count > 7000:
            raise ValueError(f"Word count {self.word_count} is outside required range of 5000-7000")

    def _validate_abstract_length(self):
        """Validate that abstract is within character limit"""
        if len(self.abstract) > 5000:
            raise ValueError(f"Abstract exceeds 5000 character limit")

    def _validate_citation_requirements(self):
        """Validate minimum 15 sources with 50%+ peer-reviewed requirement"""
        total_sources = len(self.citations)
        if total_sources < 15:
            raise ValueError(f"Minimum 15 sources required, only {total_sources} provided")

        peer_reviewed_count = sum(1 for citation in self.citations if citation.is_peer_reviewed)
        peer_reviewed_percentage = (peer_reviewed_count / total_sources) * 100

        if peer_reviewed_percentage < 50.0:
            raise ValueError(
                f"Minimum 50% peer-reviewed sources required, "
                f"only {peer_reviewed_percentage:.1f}% provided ({peer_reviewed_count}/{total_sources})"
            )

    def update_content(self, content: str):
        """Update paper content and recalculate word count"""
        self.word_count = len(content.split())
        self._validate_word_count()
        self.last_modified = datetime.now()

    def add_author(self, author: Author):
        """Add an author to the paper"""
        self.authors.append(author)
        self.last_modified = datetime.now()

    def add_citation(self, citation: Citation):
        """Add a citation to the paper and validate requirements"""
        self.citations.append(citation)
        self._validate_citation_requirements()
        self.last_modified = datetime.now()

    def get_citation_summary(self) -> Dict[str, int]:
        """Get summary of citation types and peer-review status"""
        summary = {
            "total": len(self.citations),
            "peer_reviewed": 0,
            "journal": 0,
            "conference": 0,
            "preprint": 0,
            "whitepaper": 0,
            "technical_report": 0
        }

        for citation in self.citations:
            if citation.is_peer_reviewed:
                summary["peer_reviewed"] += 1

            if hasattr(citation, 'source_type'):
                if citation.source_type == "journal":
                    summary["journal"] += 1
                elif citation.source_type == "conference":
                    summary["conference"] += 1
                elif citation.source_type == "preprint":
                    summary["preprint"] += 1
                elif citation.source_type == "whitepaper":
                    summary["whitepaper"] += 1
                elif citation.source_type == "technical_report":
                    summary["technical_report"] += 1

        return summary

    def to_dict(self) -> Dict:
        """Convert the paper to a dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "word_count": self.word_count,
            "status": self.status.value,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "authors": [author.to_dict() for author in self.authors],
            "citations": [citation.to_dict() for citation in self.citations]
        }