from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):

    ENGLISH = "en"
    GERMAN = "de"


class PostType(str, Enum):

    FACTUAL_CLAIM = "Factual Claim"
    OPINION = "Opinion"
    QUESTION = "Question"
    PERSONAL_UPDATE = "Personal Update"
    PROMOTION = "Promotion"


class VeracityStatus(str, Enum):

    FACTUALLY_CORRECT = "Factually Correct"
    UNTRUTH = "Untruth"
    MISLEADING = "Misleading"
    UNVERIFIABLE = "Unverifiable"


class PoliticalTendency(str, Enum):

    LEFT = "Links"
    CENTER_LEFT = "Mitte-Links"
    CENTER = "Mitte"
    CENTER_RIGHT = "Mitte-Rechts"
    RIGHT = "Rechts"
    NEUTRAL = "Neutral"


class Intent(str, Enum):

    INFORMATIVE = "Informative"
    PERSUASIVE = "Persuasive"
    SATIRICAL = "Satirical"
    PROVOCATIVE = "Provocative"
    COMMERCIAL = "Commercial"
    ENTERTAINING = "Entertaining"


class SocialMediaPostRequest(BaseModel):

    post_id: str = Field(
        ...,
        description="Unique identifier for the post"
    )
    post_text: str = Field(
        ...,
        min_length=10,
        description="The text content of the post to evaluate"
    )
    language: Optional[str] = Field(
        "en",
        description="Language of the post (en/de)"
    )

    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        """Validate that language is one of the supported values."""
        if v not in ["en", "de"]:
            raise ValueError('Language must be either "en" or "de"')
        return v


class Source(BaseModel):

    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Text snippet from the source")


class PostAnalysis(BaseModel):

    post_type: PostType = Field(
        ...,
        description="Classification of the post type"
    )
    is_spam: bool = Field(
        ...,
        description="Whether the post is identified as spam"
    )


class VeracityAnalysis(BaseModel):

    status: VeracityStatus = Field(
        ...,
        description="Veracity status of the claim"
    )
    justification: str = Field(
        ...,
        description="AI-generated explanation of the conclusion"
    )
    verification_method: str = Field(
        ...,
        description="How the verification was performed"
    )
    sources: List[Source] = Field(
        ...,
        description="Sources used for verification"
    )


class PoliticalTendencyAnalysis(BaseModel):

    primary: PoliticalTendency = Field(
        ...,
        description="Primary political tendency"
    )
    scores: dict[str, float] = Field(
        ...,
        description="Confidence scores for all tendencies"
    )


class NuanceAnalysis(BaseModel):

    political_tendency: PoliticalTendencyAnalysis = Field(
        ...,
        description="Political tendency analysis"
    )
    detected_intents: List[Intent] = Field(
        ...,
        description="Detected intents in the post"
    )


class AdvancedEvaluationResponse(BaseModel):

    post_id: str = Field(..., description="Original post ID")
    analysis_timestamp: str = Field(
        ...,
        description="ISO timestamp of analysis"
    )
    language: str = Field(..., description="Language of the post")
    post_analysis: PostAnalysis = Field(
        ...,
        description="Post type and spam analysis"
    )
    veracity_analysis: Optional[VeracityAnalysis] = Field(
        None,
        description="Veracity analysis (only for claims)"
    )
    nuance_analysis: NuanceAnalysis = Field(
        ...,
        description="Nuance analysis results"
    )
    processing_error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )


class HealthResponse(BaseModel):

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Service version")
