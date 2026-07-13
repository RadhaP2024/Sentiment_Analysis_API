# 🚀 Sentiment Analysis API using FastAPI, Transformers & Docker

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

A production-ready **Sentiment Analysis REST API** built using **FastAPI** and **Hugging Face Transformers**, containerized with **Docker**, and deployable to cloud platforms such as **AWS Elastic Beanstalk**, **Render**, or **Railway**.

The API predicts whether a sentence is **Positive**, **Neutral**, or **Negative** along with the prediction confidence.

---

# 📌 Features

- ✅ Fine-tuned DistilBERT model
- ✅ REST API using FastAPI
- ✅ Interactive Swagger UI
- ✅ Dockerized application
- ✅ Cloud Deployment Ready
- ✅ JSON Response
- ✅ High-performance inference
- ✅ Production-ready project structure

---

# 🏗 Project Architecture

```text
                   +----------------------+
                   |      Client/User     |
                   | Browser / Postman    |
                   +----------+-----------+
                              |
                              |
                       HTTP Request
                              |
                              ▼
                   +----------------------+
                   |      FastAPI API     |
                   |    (Uvicorn Server)  |
                   +----------+-----------+
                              |
                              |
                              ▼
                  +-----------------------+
                  | Prediction Module     |
                  | predict.py            |
                  +----------+------------+
                             |
                             |
                             ▼
                +----------------------------+
                | HuggingFace Tokenizer      |
                | DistilBERT Sentiment Model |
                +------------+---------------+
                             |
                             ▼
                    Sentiment Prediction
                             |
                             ▼
                  JSON Response Returned
```

---

# 📂 Project Structure

```text
Sentiment-Analysis-API/
│
├── app/
│   ├── main.py
│   ├── predict.py
│
├── model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── model.safetensors
│   └── special_tokens_map.json
│
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
└── Procfile
```

---

# ⚙ Tech Stack

- Python
- FastAPI
- Uvicorn
- Hugging Face Transformers
- PyTorch
- Docker
- Git & GitHub

---

# 🚀 API Workflow

```text
Input Text
      │
      ▼
Tokenizer
      │
      ▼
DistilBERT Model
      │
      ▼
Softmax
      │
      ▼
Highest Probability
      │
      ▼
Prediction + Confidence
```

---

# 🔥 REST API Endpoints

## Home

```
GET /
```

Returns

```json
{
  "message":"Sentiment Analysis API Running"
}
```

---

## Predict Sentiment

```
POST /predict
```

Request

```json
{
    "text":"I love this product!"
}
```

Response

```json
{
    "text":"I love this product!",
    "prediction":"Positive",
    "confidence":0.9981
}
```

---

# 📦 Docker Deployment

## Build Image

```bash
docker build -t sentiment-api .
```

## Run Container

```bash
docker run -p 8000:8000 sentiment-api
```

API available at

```
http://localhost:8000
```

Swagger UI

```
http://localhost:8000/docs
```

---

# ☁ Cloud Deployment

This project can be deployed on

- AWS Elastic Beanstalk
- AWS ECS
- Railway
- Render
- Northflank
- Azure App Service
- Google Cloud Run

---

# 🧠 Machine Learning Model

Model Used

- DistilBERT

Framework

- Hugging Face Transformers

Task

- Text Classification

Classes

- Positive
- Neutral
- Negative

---

# 📈 Sample Prediction

Input

```
The movie was amazing!
```

Output

```json
{
    "prediction":"Positive",
    "confidence":0.9974
}
```

---

# ⚡ Installation

Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Sentiment_Analysis_API.git
```

Move into project

```bash
cd Sentiment_Analysis_API
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install Packages

```bash
pip install -r requirements.txt
```

Run API

```bash
uvicorn app.main:app --reload
```
---
# 🌐 Live Demo

API
https://sentiment-analysis-api-lpko.onrender.com/

Swagger UI
https://sentiment-analysis-api-lpko.onrender.com/docs#/
--- 
# 📊 Performance

- Fast inference
- REST architecture
- Lightweight API
- Dockerized deployment
- Cloud-ready

---

# 🔮 Future Improvements

- Streamlit Frontend
- Batch Prediction
- User Authentication
- Model Versioning
- Kubernetes Deployment
- CI/CD using GitHub Actions
- AWS S3 Model Storage
- Monitoring using Prometheus & Grafana

---

# 👨‍💻 Author

**Radha Prasad G**

GitHub

https://github.com/RadhaP2024

---
