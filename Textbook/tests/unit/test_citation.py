import unittest
from datetime import datetime
from src.paper.models.citation import Citation, SourceType


class TestCitation(unittest.TestCase):
    """Unit tests for the Citation model"""

    def test_citation_creation_with_valid_data(self):
        """Test creating a citation with valid data."""
        citation = Citation(
            source_type=SourceType.JOURNAL,
            title="Sample Paper",
            authors=["John Doe", "Jane Smith"],
            publication_date=datetime.now(),
            doi="10.1234/example",
            url="https://example.com",
            is_peer_reviewed=True,
            citation_text="Doe, J., & Smith, J. (2023). Sample paper. Journal, 1(1), 1-10."
        )

        self.assertEqual(citation.title, "Sample Paper")
        self.assertEqual(len(citation.authors), 2)
        self.assertEqual(citation.source_type, SourceType.JOURNAL.value)
        self.assertTrue(citation.is_peer_reviewed)
        self.assertEqual(citation.doi, "10.1234/example")
        self.assertEqual(citation.url, "https://example.com")

    def test_citation_creation_with_minimal_data(self):
        """Test creating a citation with minimal required data."""
        citation = Citation(
            source_type=SourceType.CONFERENCE,
            title="Minimal Paper",
            authors=["Single Author"],
            publication_date=datetime.now(),
            is_peer_reviewed=False
        )

        self.assertEqual(citation.title, "Minimal Paper")
        self.assertEqual(len(citation.authors), 1)
        self.assertEqual(citation.source_type, SourceType.CONFERENCE.value)
        self.assertFalse(citation.is_peer_reviewed)
        self.assertIsNone(citation.doi)
        self.assertIsNone(citation.url)

    def test_citation_to_dict(self):
        """Test converting citation to dictionary."""
        citation = Citation(
            source_type=SourceType.JOURNAL,
            title="Sample Paper",
            authors=["John Doe"],
            publication_date=datetime(2023, 5, 15),
            doi="10.1234/example",
            is_peer_reviewed=True,
            citation_text="Doe, J. (2023). Sample paper. Journal, 1(1), 1-10."
        )

        citation_dict = citation.to_dict()

        self.assertIsInstance(citation_dict, dict)
        self.assertEqual(citation_dict["title"], "Sample Paper")
        self.assertEqual(citation_dict["authors"], ["John Doe"])
        self.assertEqual(citation_dict["source_type"], SourceType.JOURNAL.value)
        self.assertTrue(citation_dict["is_peer_reviewed"])
        self.assertEqual(citation_dict["doi"], "10.1234/example")
        self.assertIn("created_date", citation_dict)


if __name__ == '__main__':
    unittest.main()