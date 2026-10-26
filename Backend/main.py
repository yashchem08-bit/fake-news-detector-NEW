"""
Member C — Backend Developer
==============================
FastAPI backend that serves the trained model (Member B) to the
frontend (Member D).

Endpoints:
    GET  /health        - simple health check
    POST /predict        - predict from raw article text
    POST /predict-url    - fetch an article from a URL, then predict

Run from the project root (not from inside backend/) so the model/
and data_processing/ imports resolve correctly:
    uvicorn backend.main:app --reload
"""

import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Allow importing the model/ and data_processing/ packages from the
# project root even when this file is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.predict import predict_news
from Data_processing.scrape_url import fetch_article_text

app = FastAPI(
    title="Fake News Detector API",
    description="Backend API for the AI fake news detector project.",
    version="1.0.0",
)

# Allow the frontend (served from a different port/origin) to call this API.
# Tighten allow_origins to your actual frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v


class UrlRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: TextRequest):
    start = time.time()
    try:
        result = predict_news(request.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["elapsed_seconds"] = round(time.time() - start, 3)
    return result


@app.post("/predict-url")
def predict_url(request: UrlRequest):
    start = time.time()
    try:
        article_text = fetch_article_text(request.url)
        result = predict_news(article_text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # network errors, parsing failures, etc.
        raise HTTPException(status_code=422, detail=f"Could not process URL: {e}")

    result["elapsed_seconds"] = round(time.time() - start, 3)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

