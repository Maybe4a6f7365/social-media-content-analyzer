import json
import logging
import time
from typing import Dict, List, Tuple
from anthropic import Anthropic
from ..config import settings

logger = logging.getLogger(__name__)


class ClaudeService:

    def __init__(self) -> None:
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("No Anthropic API key provided. Claude analysis will be disabled.")
            self.client = None
            self.enabled = False
        else:
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.enabled = True
            logger.info("Claude service initialized successfully")

    async def classify_post_type(
        self,
        text: str,
        language: str = "en"
    ) -> Tuple[str, float, Dict[str, float]]:

        if not self.enabled:
            logger.warning("Claude service not enabled, using default classification")
            return self._get_default_classification(language)

        labels = self._get_post_type_labels(language)

        try:
            prompt = self._build_classification_prompt(text, labels, language)
            response = self._make_api_call(prompt)
            result = self._parse_classification_response(response, labels)
            return result

        except Exception as e:
            logger.error(f"Error in Claude classification: {e}")
            return self._get_default_classification(language)

    async def analyze_political_tendency(
        self,
        text: str,
        language: str = "en"
    ) -> Dict[str, float]:

        if not self.enabled:
            logger.warning("Claude service not enabled, using default political analysis")
            return self._get_default_political_analysis(language)

        labels = self._get_political_labels(language)

        try:
            prompt = self._build_political_analysis_prompt(text, labels, language)
            response = self._make_api_call(prompt)
            return self._parse_political_response(response, labels)

        except Exception as e:
            logger.error(f"Error in Claude political analysis: {e}")
            return self._get_default_political_analysis(language)

    async def analyze_intents(
        self,
        text: str,
        language: str = "en"
    ) -> Dict[str, float]:

        if not self.enabled:
            logger.warning("Claude service not enabled, using default intent analysis")
            return self._get_default_intent_analysis(language)

        labels = self._get_intent_labels(language)

        try:
            prompt = self._build_intent_analysis_prompt(text, labels, language)
            response = self._make_api_call(prompt)
            return self._parse_intent_response(response, labels)

        except Exception as e:
            logger.error(f"Error in Claude intent analysis: {e}")
            return self._get_default_intent_analysis(language)

    async def analyze_veracity(
        self,
        claim: str,
        language: str = "en"
    ) -> Tuple[str, str, str]:

        if not self.enabled:
            logger.warning("Claude service not enabled, using default veracity analysis")
            return self._get_default_veracity_analysis(claim, language)

        try:
            prompt = self._build_veracity_prompt(claim, language)
            response = self._make_api_call(prompt)
            return self._parse_veracity_response(response)

        except Exception as e:
            logger.error(f"Error in Claude veracity analysis: {e}")
            return self._get_default_veracity_analysis(claim, language)

    def _make_api_call(self, prompt: str, max_retries: int = 3) -> str:

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=settings.CLAUDE_MODEL,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e

    def _build_classification_prompt(
        self,
        text: str,
        labels: List[str],
        language: str
    ) -> str:
        if language == "de":
            return f"""
Klassifiziere den folgenden Text in eine der Kategorien: {', '.join(labels)}

Text: "{text}"

Antworte nur mit einem JSON-Objekt im folgenden Format:
{{
    "primary_label": "gewählte_kategorie",
    "confidence": 0.95,
    "scores": {{
        "kategorie1": 0.95,
        "kategorie2": 0.03,
        "kategorie3": 0.02
    }}
}}
"""
        else:
            return f"""
Classify the following text into one of these categories: {', '.join(labels)}

Text: "{text}"

Respond only with a JSON object in the following format:
{{
    "primary_label": "chosen_category",
    "confidence": 0.95,
    "scores": {{
        "category1": 0.95,
        "category2": 0.03,
        "category3": 0.02
    }}
}}
"""

    def _build_political_analysis_prompt(
        self,
        text: str,
        labels: List[str],
        language: str
    ) -> str:
        if language == "de":
            return f"""
Analysiere die politische Tendenz des folgenden Textes: {', '.join(labels)}

Text: "{text}"

Antworte nur mit einem JSON-Objekt im folgenden Format:
{{
    "scores": {{
        "kategorie1": 0.95,
        "kategorie2": 0.03,
        "kategorie3": 0.02
    }}
}}
"""
        else:
            return f"""
Analyze the political tendency of the following text: {', '.join(labels)}

Text: "{text}"

Respond only with a JSON object in the following format:
{{
    "scores": {{
        "category1": 0.95,
        "category2": 0.03,
        "category3": 0.02
    }}
}}
"""

    def _build_intent_analysis_prompt(
        self,
        text: str,
        labels: List[str],
        language: str
    ) -> str:
        if language == "de":
            return f"""
Analysiere die Absichten im folgenden Text: {', '.join(labels)}

Text: "{text}"

Antworte nur mit einem JSON-Objekt im folgenden Format:
{{
    "scores": {{
        "kategorie1": 0.95,
        "kategorie2": 0.03,
        "kategorie3": 0.02
    }}
}}
"""
        else:
            return f"""
Analyze the intents in the following text: {', '.join(labels)}

Text: "{text}"

Respond only with a JSON object in the following format:
{{
    "scores": {{
        "category1": 0.95,
        "category2": 0.03,
        "category3": 0.02
    }}
}}
"""

    def _build_veracity_prompt(
        self,
        claim: str,
        language: str
    ) -> str:
        if language == "de":
            return f"""
Du bist ein neutraler, unparteiischer deutscher Faktenchecker. Bewerte die folgende Behauptung basierend auf deinem Wissen.

Behauptung: "{claim}"

Antworte nur mit einem JSON-Objekt im folgenden Format:
{{
    "status": "Factually Correct|Untruth|Misleading|Unverifiable",
    "justification": "Kurze Begründung",
    "verification_method": "AI-basierte Analyse"
}}
"""
        else:
            return f"""
You are a neutral, impartial fact-checker. Evaluate the following claim based on your knowledge.

Claim: "{claim}"

Respond only with a JSON object in the following format:
{{
    "status": "Factually Correct|Untruth|Misleading|Unverifiable",
    "justification": "Brief justification",
    "verification_method": "AI-based analysis"
}}
"""

    def _parse_classification_response(
        self,
        response_text: str,
        labels: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        try:
            data = json.loads(response_text)
            primary_label = data.get("primary_label", labels[0])
            confidence = data.get("confidence", 0.5)
            scores = data.get("scores", {})

            for label in labels:
                if label not in scores:
                    scores[label] = 0.0

            return primary_label, confidence, scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing classification response: {e}")
            return labels[0], 0.5, {label: 1.0 / len(labels) for label in labels}

    def _parse_political_response(
        self,
        response_text: str,
        labels: List[str]
    ) -> Dict[str, float]:
        try:
            data = json.loads(response_text)
            scores = data.get("scores", {})

            for label in labels:
                if label not in scores:
                    scores[label] = 0.0

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing political response: {e}")
            return {label: 1.0 / len(labels) for label in labels}

    def _parse_intent_response(
        self,
        response_text: str,
        labels: List[str]
    ) -> Dict[str, float]:
        try:
            data = json.loads(response_text)
            scores = data.get("scores", {})

            for label in labels:
                if label not in scores:
                    scores[label] = 0.0

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing intent response: {e}")
            return {label: 1.0 / len(labels) for label in labels}

    def _parse_veracity_response(self, response_text: str) -> Tuple[str, str, str]:
        try:
            data = json.loads(response_text)
            status = data.get("status", "Unverifiable")
            justification = data.get("justification", "No analysis available")
            verification_method = data.get("verification_method", "No method specified")

            return status, justification, verification_method

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing veracity response: {e}")
            return "Unverifiable", "Error in analysis", "No method available"

    def _get_post_type_labels(self, language: str) -> List[str]:
        return settings.DE_POST_TYPES if language == "de" else settings.EN_POST_TYPES

    def _get_political_labels(self, language: str) -> List[str]:
        return (
            settings.DE_POLITICAL_LABELS
            if language == "de"
            else settings.EN_POLITICAL_LABELS
        )

    def _get_intent_labels(self, language: str) -> List[str]:
        return (
            settings.DE_INTENT_LABELS
            if language == "de"
            else settings.EN_INTENT_LABELS
        )

    def _get_default_classification(self, language: str) -> Tuple[str, float, Dict[str, float]]:
        labels = self._get_post_type_labels(language)
        scores = {label: 1.0 / len(labels) for label in labels}
        return labels[0], 1.0 / len(labels), scores

    def _get_default_political_analysis(self, language: str) -> Dict[str, float]:
        labels = self._get_political_labels(language)
        return {label: 1.0 / len(labels) for label in labels}

    def _get_default_intent_analysis(self, language: str) -> Dict[str, float]:
        labels = self._get_intent_labels(language)
        return {label: 1.0 / len(labels) for label in labels}

    def _get_default_veracity_analysis(
        self,
        claim: str,
        language: str
    ) -> Tuple[str, str, str]:
        return "Unverifiable", "Analysis not available", "Default method used"
