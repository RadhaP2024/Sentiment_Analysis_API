# Sentiment Analysis API

## Project Overview

This project provides a REST API for sentiment analysis using DistilBERT.

## Tech Stack

- Python
- FastAPI
- Hugging Face Transformers
- DistilBERT
- Docker
- AWS Elastic Beanstalk
- AWS S3

## Features

- Predict Positive, Neutral, Negative sentiment
- Interactive Swagger documentation
- Docker-ready deployment

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit:

http://127.0.0.1:8000/docs

## API

POST `/predict`

Example:

```json
{
    "text":"I love this product!"
}
```

Example response:

```json
{
    "prediction":"Positive",
    "confidence":0.998
}
```