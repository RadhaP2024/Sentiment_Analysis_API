from fastapi import FastAPI
from app.schemas import TextRequest
from app.predict import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running!"
    }

@app.post("/predict")
def predict(data: TextRequest):
    return predict_sentiment(data.text)