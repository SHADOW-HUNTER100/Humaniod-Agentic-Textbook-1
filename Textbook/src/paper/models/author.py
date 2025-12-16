"""
Author Model for AI-Native Software Development & Physical AI project
Defines the structure and validation for paper authors
"""

import uuid
import re
from datetime import datetime
from typing import Dict, Optional


class Author:
    """
    Represents an author contributing to a research paper
    Validates author information according to academic standards
    """

    def __init__(self,
                 first_name: str,
                 last_name: str,
                 institution: str,
                 email: str,
                 orcid: Optional[str] = None):
        """
        Initialize an Author with validation

        Args:
            first_name: First name of the author
            last_name: Last name of the author
            institution: Institution affiliation of the author
            email: Email address of the author (must be valid format)
            orcid: ORCID identifier (optional)
        """
        self.id = str(uuid.uuid4())
        self.first_name = first_name
        self.last_name = last_name
        self.institution = institution
        self.email = email
        self.orcid = orcid
        self.created_date = datetime.now()
        self.last_modified = datetime.now()

        # Validate initial values
        self._validate_email_format()
        self._validate_institution()

    def _validate_email_format(self):
        """Validate that email is in a valid format"""
        # Regular expression for email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(email_pattern, self.email):
            raise ValueError(f"Email '{self.email}' is not in a valid format")

    def _validate_institution(self):
        """Validate that institution is not empty"""
        if not self.institution or not self.institution.strip():
            raise ValueError("Institution must be non-empty")

    def update_email(self, email: str):
        """Update email and validate format"""
        self.email = email
        self._validate_email_format()
        self.last_modified = datetime.now()

    def update_institution(self, institution: str):
        """Update institution and validate"""
        self.institution = institution
        self._validate_institution()
        self.last_modified = datetime.now()

    def update_orcid(self, orcid: Optional[str]):
        """Update ORCID identifier"""
        self.orcid = orcid
        self.last_modified = datetime.now()

    def to_dict(self) -> Dict:
        """Convert the author to a dictionary representation"""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "institution": self.institution,
            "email": self.email,
            "orcid": self.orcid,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat()
        }

    @property
    def full_name(self) -> str:
        """Get the full name of the author"""
        return f"{self.first_name} {self.last_name}"

    @property
    def display_name(self) -> str:
        """Get the display name with institution"""
        return f"{self.full_name} ({self.institution})"

    @property
    def initials(self) -> str:
        """Get author initials"""
        first_initial = self.first_name[0] if self.first_name else ""
        last_initial = self.last_name[0] if self.last_name else ""
        return f"{first_initial}{last_initial}".upper()