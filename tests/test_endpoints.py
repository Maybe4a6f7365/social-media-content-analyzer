import pytest
from fastapi.testclient import TestClient
from src.main import app, app_state
from src.services.evaluation_service import AdvancedEvaluationService

app_state["evaluation_service"] = AdvancedEvaluationService()

client = TestClient(app)


def test_health_endpoint_returns_operational_status():
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "timestamp" in data
    assert data["version"] == "2.0.0"


def test_config_endpoint_returns_service_configuration():
    response = client.get("/config")
    
    assert response.status_code == 200
    data = response.json()
    assert "enable_advanced_analysis" in data
    assert "enable_veracity_check" in data
    assert "enable_nuance_analysis" in data
    assert "claude_model" in data


def test_evaluate_english_content_returns_analysis():
    response = client.post(
        "/evaluate",
        json={
            "post_id": "test_001",
            "post_text": "The weather is nice today.",
            "language": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["post_id"] == "test_001"
    assert "post_analysis" in data
    assert "nuance_analysis" in data
    assert "post_type" in data["post_analysis"]
    assert "is_spam" in data["post_analysis"]


def test_evaluate_german_content_returns_analysis():
    response = client.post(
        "/evaluate",
        json={
            "post_id": "test_002",
            "post_text": "Das Wetter ist heute schön.",
            "language": "de"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "de"
    assert "post_analysis" in data
    assert "nuance_analysis" in data


def test_evaluate_longer_content_returns_analysis():
    response = client.post(
        "/evaluate",
        json={
            "post_id": "test_003",
            "post_text": "I think we should consider the environmental impact of our decisions.",
            "language": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "post_analysis" in data
    assert "nuance_analysis" in data


def test_evaluate_question_content_returns_analysis():
    response = client.post(
        "/evaluate",
        json={
            "post_id": "test_004",
            "post_text": "What time is the meeting?",
            "language": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "post_analysis" in data
    assert "nuance_analysis" in data


def test_evaluate_invalid_input_returns_validation_error():
    response = client.post(
        "/evaluate",
        json={"post_id": "invalid", "post_text": "Too short"}
    )
    
    assert response.status_code == 422


def test_evaluate_missing_required_fields_returns_error():
    response = client.post(
        "/evaluate",
        json={"post_id": "test_005"}
    )
    
    assert response.status_code == 422


def test_evaluate_empty_text_returns_error():
    response = client.post(
        "/evaluate",
        json={
            "post_id": "test_006",
            "post_text": "",
            "language": "en"
        }
    )
    
    assert response.status_code == 422


def test_evaluate_invalid_language_returns_error():
    response = client.post(
        "/evaluate",
        json={
            "post_id": "test_007",
            "post_text": "Test content",
            "language": "fr"
        }
    )
    
    assert response.status_code == 422
