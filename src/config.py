from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

    ANTHROPIC_API_KEY: str = Field(
        default="",
        description="Anthropic API key for Claude access"
    )
    CLAUDE_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Claude model to use for analysis"
    )

    ENABLE_ADVANCED_ANALYSIS: bool = True
    ENABLE_VERACITY_CHECK: bool = True
    ENABLE_NUANCE_ANALYSIS: bool = True

    EN_POST_TYPES: List[str] = [
        "Factual Claim", "Opinion", "Question",
        "Personal Update", "Promotion"
    ]
    DE_POST_TYPES: List[str] = [
        "Faktische Behauptung", "Meinungsäußerung", "Frage",
        "Persönliche Mitteilung", "Werbung / Spam"
    ]

    EN_POLITICAL_LABELS: List[str] = [
        "Left", "Center-Left", "Center",
        "Center-Right", "Right", "Neutral"
    ]
    DE_POLITICAL_LABELS: List[str] = [
        "Politisch Links", "Politisch Mitte-Links", "Politisch Mitte",
        "Politisch Mitte-Rechts", "Politisch Rechts", "Politisch Neutral"
    ]

    EN_INTENT_LABELS: List[str] = [
        "Informative", "Persuasive", "Satirical",
        "Provocative", "Commercial", "Entertaining"
    ]
    DE_INTENT_LABELS: List[str] = [
        "Informativ", "Überzeugend", "Satirisch",
        "Provozierend", "Kommerziell", "Unterhaltend"
    ]

    INTENT_CONFIDENCE_THRESHOLD: float = 0.3
    SPAM_CONFIDENCE_THRESHOLD: float = 0.7

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    USE_LOCAL_LLM: bool = False
    LOCAL_LLM_URL: str = ""
    LOCAL_LLM_MODEL: str = "llama3.2"
    LOCAL_LLM_TIMEOUT: int = 30


settings = Settings()
