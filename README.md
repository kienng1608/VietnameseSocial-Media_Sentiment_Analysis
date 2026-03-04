# Robust Sentiment Classification of Informal Vietnamese Social Media

This repository contains the source code, deployment scripts, and API for my paper:
**"Leveraging Large Language Models and Feature Engineering for Robust Sentiment Classification of Informal Vietnamese Social Media"**

A robust REST API for Vietnamese text sentiment classification built with FastAPI, Hugging Face Sentence Transformers, and Scikit-learn SVM. The system targets informal Facebook comments, specifically focusing on football domains (Lionel Messi and Cristiano Ronaldo), classifying them into Positive (+1), Neutral (0), or Negative (-1).

## Highlights & Paper Contributions
*   **LLM-Assisted Normalization:** Employs Gemini 2.0 Flash to build mapping dictionaries for "teen code" and abbreviations.
*   **LLM-Assisted Labeling:** Utilizes Gemini 2.5 Flash for automatic large-scale dataset annotation.
*   **Named Entity Removal:** Uses `NlpHUST/ner-vietnamese-electra-base` to eliminate irrelevant personal names.
*   **Advanced Embeddings vs Traditional ML:** Compares BoW, TF-IDF against PhoBERT, E5, and Vietnamese-Document-Embedding (VE).
*   **Best Model Performance:** The combination of **VE (Vietnamese Embedding) + SVM** with **Oversampling** achieves an accuracy and F1-score of **0.76**, outperforming even fine-tuned PhoBERT models on this highly informal dataset.

## Core Pipeline

```text
[Raw Informal Facebook Comment]
    ↓ NlpHUST/ner-vietnamese-electra-base (PER removal)
    ↓ LLM-assisted Teen-code Normalization & Cleaning
    ↓ VnCoreNLP Word Segmentation
[Preprocessed Text]  
    ↓ dangvantuan/vietnamese-document-embedding (VE)
[768-dim Semantic Embedding]
    ↓ Balanced Support Vector Machine (Oversampled SVM)
[Label: -1 / 0 / 1]
```

## Features

*   **FastAPI backend** with strict CORS support (configured for `http://localhost,http://localhost:3000` by default).
*   **Pydantic models** for robust input validation and structured responses.
*   **Health and system endpoints** for Docker container monitoring.
*   **Fully Containerized** with multi-stage Docker builds, non-root user execution, and robust health checks.

## Project Structure

```text
fastapi_docker_compose/
├── app/
│   ├── main.py                 # FastAPI app initialization, lifespan, CORS, router registration
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── routers/
│   │   └── api.py              # API endpoints (/predict, /health)
│   └── services/
│       ├── preprocessor.py     # Core logic for cleaning text and loading VnCoreNLP
│       └── predictor.py        # Core logic for loading models and running predictions
├── docker/
│   ├── Dockerfile              # Multi-stage Dockerfile adhering to best practices
│   └── docker-compose.yml      # Compose file linking the API with resources
├── data/                       # Directory for dataset files (gitignored)
├── models/                     # Directory for the pre-trained SVM model (gitignored)
├── scripts/
│   └── label_data.py           # Data labeling script using the Google Gemini API
├── .env.example                # Sample environment variables
├── .dockerignore               # Optimized Docker build context
├── .gitignore                  # Git exclusions for secrets and models
├── AVOIDANCE_TABLE.md          # Documentation on how 8 common deployment pitfalls were avoided
└── requirements.txt            # Pinned Python dependencies
```

## Requirements

*   **Docker & Docker Compose** (Recommended for deployment)
*   **Python:** 3.11+ (If running locally without Docker)
*   **Java Runtime (JRE/JDK):** Required for VnCoreNLP if running locally without Docker.

**Key Dependencies (`requirements.txt`):**
*   `fastapi`, `uvicorn`, `pydantic`
*   `py_vncorenlp`
*   `sentence-transformers`, `torch`, `transformers`
*   `scikit-learn`, `joblib`


## API Endpoints

### `GET /api/v1/health`
**Description:** Health check endpoint used by Docker to ensure the app, embedding model, and VnCoreNLP are fully loaded and operational.
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "vncore_loaded": true,
  "version": "1.0.0",
  "message": null
}
```

### `POST /api/v1/predict`
**Description:** Processes a Vietnamese sentence, applies text cleaning and word segmentation, generates embeddings, and predicts the sentiment.
**Request Body (`PredictRequest`):**
```json
{
  "text": "Messi đá hay quá, fan thực sự tự hào!"
}
```
**Response (`PredictResponse`):**
```json
{
  "text": "Messi đá hay quá, fan thực sự tự hào!",
  "text_preprocessed": "messi đá hay quá fan thực_sự tự_hào",
  "label": 1,
  "sentiment": "Tích cực",
  "emoji": "😊",
  "processed_at": "2026-03-04T17:15:23.123Z"
}
```
**Errors:**
*   `422 Unprocessable Entity`: Input is empty or exceeds the character limit.
*   `503 Service Unavailable`: Triggered if the models have not finished loading yet.

## Model Details

*   **Tokenizer/Segmentation:** `VnCoreNLP` (Wraps Java processes for accurate Vietnamese word boundary detection).
*   **Embedding Pipeline:** `sentence-transformers` loading `dangvantuan/vietnamese-document-embedding` (768-dimensional output). Extracted dynamically without backpropagation (`torch.no_grad()`).
*   **Classifier:** A scikit-learn Support Vector Machine (SVM) pipeline persisted via `joblib`.
*   **Outputs:**
    *   `1`: Positive (Tích cực)
    *   `0`: Neutral (Trung lập)
    *   `-1`: Negative (Tiêu cực)

Models are loaded at startup within the FastAPI lifespan context in `app/main.py` and managed via Singletons in `app/services/`.

## Quality Assurance & Best Practices

The repository adheres strictly to deployment best practices, actively preventing 8 common infrastructural errors. Please view `AVOIDANCE_TABLE.md` for full details, which includes:
1.  Using a pinned `python:3.11-slim` base image.
2.  Proper `.gitignore` configuration for `.env` files and large model binaries.
3.  Implementing restart policies inside Docker compose.
4.  Avoiding hardcoded credentials.
5.  Implementing robust Docker `HEALTHCHECK`.
6.  Running containers using a non-root `appuser`.
7.  Restricting wildcard CORS arrays.
8.  Pinning all production dependency versions accurately in `requirements.txt`.
