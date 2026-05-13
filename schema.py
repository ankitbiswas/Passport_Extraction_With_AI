"""Pydantic schema for passport extraction."""

from pydantic import BaseModel, Field


class Passport(BaseModel):
    """Fields to extract from a passport document."""

    first_name: str = Field(description="First/given name as printed on the passport")
    last_name: str = Field(description="Last/family name as printed on the passport")
    passport_number: str = Field(description="Passport document number")
    nationality: str = Field(description="Country of citizenship")
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")
    date_of_expiry: str = Field(description="Passport expiry date in YYYY-MM-DD format")
    issuing_country: str = Field(description="Country that issued the passport")
    issuing_authority: str = Field(description="Authority that issued the passport")
