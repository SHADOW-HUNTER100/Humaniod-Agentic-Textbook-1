import unittest
from datetime import datetime
from src.paper.models.research_paper import ResearchPaper, PaperStatus
from src.paper.models.author import Author
from src.paper.models.citation import Citation, SourceType


class TestResearchPaper(unittest.TestCase):
    """Unit tests for the ResearchPaper model"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.author = Author(
            first_name="John",
            last_name="Doe",
            institution="Example Institution",
            email="john.doe@example.com",
            orcid="0000-0000-0000-0000"
        )
        self.citation = Citation(
            source_type=SourceType.JOURNAL,
            title="Sample Paper",
            authors=["Author 1"],
            publication_date=datetime.now(),
            doi="10.1234/example",
            is_peer_reviewed=True,
            citation_text="Author, A. (2023). Sample paper. Journal, 1(1), 1-10."
        )

    def test_paper_creation_with_valid_data(self):
        """Test creating a paper with valid data."""
        paper = ResearchPaper(
            title="Test Paper",
            word_count=6000,
            authors=[self.author],
            citations=[self.citation]
        )

        self.assertEqual(paper.title, "Test Paper")
        self.assertEqual(paper.word_count, 6000)
        self.assertEqual(paper.status, PaperStatus.DRAFT)
        self.assertEqual(len(paper.authors), 1)
        self.assertEqual(len(paper.citations), 1)

    def test_word_count_validation(self):
        """Test that word count is validated correctly."""
        with self.assertRaises(ValueError):
            ResearchPaper(
                title="Test Paper",
                word_count=4000,  # Below minimum
                citations=[self.citation]
            )

        with self.assertRaises(ValueError):
            ResearchPaper(
                title="Test Paper",
                word_count=8000,  # Above maximum
                citations=[self.citation]
            )

    def test_citation_requirements(self):
        """Test that citation requirements are enforced."""
        # Create 14 citations (below minimum of 15)
        citations = []
        for i in range(14):
            citation = Citation(
                source_type=SourceType.JOURNAL,
                title=f"Paper {i}",
                authors=["Author"],
                publication_date=datetime.now(),
                doi=f"10.1234/example{i}",
                is_peer_reviewed=True,
                citation_text=f"Author, A. (2023). Paper {i}. Journal, 1(1), 1-{i+10}."
            )
            citations.append(citation)

        with self.assertRaises(ValueError):
            ResearchPaper(
                title="Test Paper",
                word_count=6000,
                citations=citations
            )

    def test_peer_reviewed_requirement(self):
        """Test that peer-reviewed requirement is enforced."""
        # Create 15 citations but only 6 are peer-reviewed (less than 50%)
        citations = []
        for i in range(15):
            is_peer = i < 6  # Only 6 out of 15 are peer-reviewed (40%)
            citation = Citation(
                source_type=SourceType.JOURNAL,
                title=f"Paper {i}",
                authors=["Author"],
                publication_date=datetime.now(),
                doi=f"10.1234/example{i}",
                is_peer_reviewed=is_peer,
                citation_text=f"Author, A. (2023). Paper {i}. Journal, 1(1), 1-{i+10}."
            )
            citations.append(citation)

        with self.assertRaises(ValueError):
            ResearchPaper(
                title="Test Paper",
                word_count=6000,
                citations=citations
            )

    def test_content_update(self):
        """Test updating paper content."""
        paper = ResearchPaper(
            title="Test Paper",
            word_count=6000,
            citations=[self.citation]
        )

        original_modified = paper.last_modified
        paper.update_content("This is a new content with more words than before")

        self.assertGreater(paper.word_count, 0)
        self.assertGreater(paper.last_modified, original_modified)

    def test_add_author(self):
        """Test adding authors to paper."""
        paper = ResearchPaper(
            title="Test Paper",
            word_count=6000,
            citations=[self.citation]
        )

        original_count = len(paper.authors)
        paper.add_author(self.author)

        self.assertEqual(len(paper.authors), original_count + 1)

    def test_add_citation(self):
        """Test adding citations to paper."""
        paper = ResearchPaper(
            title="Test Paper",
            word_count=6000,
            citations=[self.citation]
        )

        new_citation = Citation(
            source_type=SourceType.CONFERENCE,
            title="New Paper",
            authors=["New Author"],
            publication_date=datetime.now(),
            doi="10.1234/newpaper",
            is_peer_reviewed=True,
            citation_text="New Author. (2023). New paper. Conference, 1-5."
        )

        original_count = len(paper.citations)
        paper.add_citation(new_citation)

        self.assertEqual(len(paper.citations), original_count + 1)


if __name__ == '__main__':
    unittest.main()