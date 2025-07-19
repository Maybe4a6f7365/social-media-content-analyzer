# Social Media Content Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![AI](https://img.shields.io/badge/AI-LLM%20Powered-purple.svg)](https://anthropic.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-orange.svg)](https://github.com/your-repo/social-media-analyzer)

A microservice that analyzes social media content using large language models with a zero-shot approach. This project implements a multi-stage analysis pipeline for categorizing and evaluating social media posts without requiring training data or model fine-tuning.

## Overview

The service performs three-stage analysis on social media content:

1. **Content Classification** - Identifies post types (factual claims, opinions, questions, etc.)
2. **Spam Detection** - Filters promotional and spam content
3. **Veracity Analysis** - Evaluates factual claims using AI analysis
4. **Context Analysis** - Analyzes political tendencies and content intent

## Architecture

```
┌─────────────────────┐
│ Stage 1: Triage         │
│ • LLM Classification    │
│ • Spam Detection        │
│ • Post Type             │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Stage 2: Veracity       │
│ (Conditional)           │
│ • AI Verification       │
│ • LLM Analysis          │
│ • Justification         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Stage 3: Context        │
│ (Conditional)           │
│ • Political Analysis    │
│ • Intent Detection      │
│ • Nuance Scoring        │
└─────────────────────┘
```

## Features

- **Multi-language support (English/German)**
- **LLM-powered analysis (Anthropic)**
- **AI-based verification for factual claims**
- **Asynchronous processing with async/await**
- **Testing suite** 
- **Modular structure**

## Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key (for Claude)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd social-media-content-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Service

```bash
# Start the service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/

# Get configuration
curl http://localhost:8000/config
```

## API Usage

### Analyze Social Media Content

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": "post_123",
    "post_text": "NASA announced today that the Perseverance rover discovered organic molecules on Mars.",
    "language": "en"
  }'
```

### Example Response

```json
{
  "post_id": "post_123",
  "analysis_timestamp": "2024-01-15T10:30:00Z",
  "language": "en",
  "post_analysis": {
    "post_type": "Factual Claim",
    "is_spam": false
  },
  "veracity_analysis": {
    "status": "Factually Correct",
    "justification": "NASA confirmed this discovery in their official press release...",
    "verification_method": "AI-based analysis",
    "sources": []
  },
  "nuance_analysis": {
    "political_tendency": {
      "primary": "Neutral",
      "scores": {
        "Left": 0.1,
        "Center": 0.8,
        "Right": 0.1
      }
    },
    "detected_intents": [
      "Informative"
    ]
  }
}
```

## Testing

### Run Tests

```bash
# Run all tests
python test_integration.py

# Run with custom URL
python test_integration.py http://localhost:8000
```

### Run Unit Tests

```bash
# Run pytest tests
pytest tests/ -v

# Run specific test
pytest tests/test_endpoints.py::test_health_endpoint_returns_operational_status -v
```

## Project Structure

```
social-media-content-analyzer/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── models/
│   │   └── api_models.py       # Pydantic models
│   └── services/
│       ├── evaluation_service.py    # Main analysis logic
│       └── claude_service.py        # Claude API integration
├── tests/
│   └── test_endpoints.py       # Unit tests
├── test_integration.py         # Integration tests
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## Current Limitations

This proof-of-concept has several limitations:

- **LLM dependency** - Requires Claude API access
- **Limited languages** - Currently supports English and German
- **No persistence** - Results are not stored
- **No authentication** - No user management
- **Limited scalability** - Single-instance deployment

## License

This project is licensed under the MIT License, allowing for commercial use.
