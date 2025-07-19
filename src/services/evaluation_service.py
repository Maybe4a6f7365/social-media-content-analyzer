import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from ..config import settings
from ..models.api_models import (
    PostType, VeracityStatus, PoliticalTendency, Intent,
    PostAnalysis, VeracityAnalysis, PoliticalTendencyAnalysis, NuanceAnalysis
)
from .claude_service import ClaudeService

logger = logging.getLogger(__name__)
SPAM_LABELS = ["Werbung / Spam", "Promotion"]


class AdvancedEvaluationService:

    def __init__(self) -> None:
        self.claude_service = ClaudeService()
        logger.info("AdvancedEvaluationService initialized")

    async def perform_full_analysis(
        self,
        post_id: str,
        post_text: str,
        language: str = "en"
    ) -> Dict:

        processing_error = None

        try:
            post_type, is_spam = await self._run_triage(post_text, language)

            veracity_analysis = await self._get_veracity_analysis(
                post_type, is_spam, post_text, language
            )

            nuance_analysis = await self._get_nuance_analysis(
                is_spam, post_text, language
            )

        except Exception as e:
            logger.error(f"Analysis failed for post {post_id}: {e}")
            processing_error = f"Analysis failed: {str(e)}"
            return {
                "post_id": post_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "language": language,
                "post_analysis": PostAnalysis(
                    post_type=PostType.OPINION,
                    is_spam=False
                ),
                "veracity_analysis": None,
                "nuance_analysis": NuanceAnalysis(
                    political_tendency=PoliticalTendencyAnalysis(
                        primary=PoliticalTendency.NEUTRAL,
                        scores={"Neutral": 1.0}
                    ),
                    detected_intents=[Intent.INFORMATIVE]
                ),
                "processing_error": processing_error
            }

        return self._build_analysis_response(
            post_id, language, post_type, is_spam,
            veracity_analysis, nuance_analysis, processing_error
        )

    async def _run_triage(self, text: str, language: str) -> Tuple[PostType, bool]:

        primary_label, confidence, scores = await self.claude_service.classify_post_type(
            text, language
        )

        is_spam = self._detect_spam(primary_label, confidence)
        post_type = self._map_to_post_type(primary_label, language)

        logger.info(f"AI triage: {post_type}, spam: {is_spam}, confidence: {confidence:.2f}")
        return post_type, is_spam

    async def _get_veracity_analysis(
        self,
        post_type: PostType,
        is_spam: bool,
        post_text: str,
        language: str
    ) -> Optional[VeracityAnalysis]:

        if not self._should_perform_veracity_check(post_type, is_spam):
            return None

        status, justification, verification_method = await self.claude_service.analyze_veracity(
            post_text, language
        )

        return VeracityAnalysis(
            status=self._map_veracity_status(status),
            justification=justification,
            verification_method=verification_method,
            sources=[]
        )

    async def _get_nuance_analysis(
        self,
        is_spam: bool,
        post_text: str,
        language: str
    ) -> Optional[NuanceAnalysis]:

        if is_spam or not settings.ENABLE_NUANCE_ANALYSIS:
            return None

        political_scores = await self.claude_service.analyze_political_tendency(
            post_text, language
        )
        intent_scores = await self.claude_service.analyze_intents(
            post_text, language
        )

        political_analysis = self._build_political_analysis(political_scores, language)
        detected_intents = self._build_intent_list(intent_scores, language)

        return NuanceAnalysis(
            political_tendency=political_analysis,
            detected_intents=detected_intents
        )

    def _should_perform_veracity_check(
        self,
        post_type: PostType,
        is_spam: bool
    ) -> bool:
        return (
            post_type == PostType.FACTUAL_CLAIM
            and not is_spam
            and settings.ENABLE_VERACITY_CHECK
        )

    def _detect_spam(self, primary_label: str, confidence: float) -> bool:
        is_promotion_label = primary_label in SPAM_LABELS
        high_promotion_score = confidence > settings.SPAM_CONFIDENCE_THRESHOLD
        return is_promotion_label and high_promotion_score

    def _build_political_analysis(
        self,
        scores: Dict[str, float],
        language: str
    ) -> PoliticalTendencyAnalysis:
        primary_label = max(scores, key=scores.get)

        return PoliticalTendencyAnalysis(
            primary=self._map_to_political_tendency(primary_label, language),
            scores={k: round(v, 4) for k, v in scores.items()}
        )

    def _build_intent_list(
        self,
        scores: Dict[str, float],
        language: str
    ) -> List[Intent]:
        detected_intents = [
            intent for intent, score in scores.items()
            if score > settings.INTENT_CONFIDENCE_THRESHOLD
        ]

        return [
            self._map_to_intent(intent, language)
            for intent in detected_intents
        ]

    def _build_analysis_response(
        self,
        post_id: str,
        language: str,
        post_type: PostType,
        is_spam: bool,
        veracity_analysis: Optional[VeracityAnalysis],
        nuance_analysis: Optional[NuanceAnalysis],
        processing_error: Optional[str] = None
    ) -> Dict:

        return {
            "post_id": post_id,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "language": language,
            "post_analysis": PostAnalysis(
                post_type=post_type,
                is_spam=is_spam
            ),
            "veracity_analysis": veracity_analysis,
            "nuance_analysis": nuance_analysis,
            "processing_error": processing_error
        }

    def _map_veracity_status(self, status: str) -> VeracityStatus:
        mapping = {
            "Factually Correct": VeracityStatus.FACTUALLY_CORRECT,
            "Untruth": VeracityStatus.UNTRUTH,
            "Misleading": VeracityStatus.MISLEADING,
            "Unverifiable": VeracityStatus.UNVERIFIABLE
        }
        return mapping.get(status, VeracityStatus.UNVERIFIABLE)

    def _map_to_post_type(self, label: str, language: str) -> PostType:
        mapping = {
            "Factual Claim": PostType.FACTUAL_CLAIM,
            "Faktische Behauptung": PostType.FACTUAL_CLAIM,
            "Opinion": PostType.OPINION,
            "Meinungsäußerung": PostType.OPINION,
            "Question": PostType.QUESTION,
            "Frage": PostType.QUESTION,
            "Personal Update": PostType.PERSONAL_UPDATE,
            "Persönliche Mitteilung": PostType.PERSONAL_UPDATE,
            "Promotion": PostType.PROMOTION,
            "Werbung / Spam": PostType.PROMOTION
        }
        return mapping.get(label, PostType.OPINION)

    def _map_to_political_tendency(
        self,
        label: str,
        language: str
    ) -> PoliticalTendency:
        mapping = {
            "Left": PoliticalTendency.LEFT,
            "Politisch Links": PoliticalTendency.LEFT,
            "Center-Left": PoliticalTendency.CENTER_LEFT,
            "Politisch Mitte-Links": PoliticalTendency.CENTER_LEFT,
            "Center": PoliticalTendency.CENTER,
            "Politisch Mitte": PoliticalTendency.CENTER,
            "Center-Right": PoliticalTendency.CENTER_RIGHT,
            "Politisch Mitte-Rechts": PoliticalTendency.CENTER_RIGHT,
            "Right": PoliticalTendency.RIGHT,
            "Politisch Rechts": PoliticalTendency.RIGHT,
            "Neutral": PoliticalTendency.NEUTRAL,
            "Politisch Neutral": PoliticalTendency.NEUTRAL
        }
        return mapping.get(label, PoliticalTendency.NEUTRAL)

    def _map_to_intent(self, label: str, language: str) -> Intent:
        mapping = {
            "Informative": Intent.INFORMATIVE,
            "Informativ": Intent.INFORMATIVE,
            "Persuasive": Intent.PERSUASIVE,
            "Überzeugend": Intent.PERSUASIVE,
            "Satirical": Intent.SATIRICAL,
            "Satirisch": Intent.SATIRICAL,
            "Provocative": Intent.PROVOCATIVE,
            "Provozierend": Intent.PROVOCATIVE,
            "Commercial": Intent.COMMERCIAL,
            "Kommerziell": Intent.COMMERCIAL,
            "Entertaining": Intent.ENTERTAINING,
            "Unterhaltend": Intent.ENTERTAINING
        }
        return mapping.get(label, Intent.INFORMATIVE)
