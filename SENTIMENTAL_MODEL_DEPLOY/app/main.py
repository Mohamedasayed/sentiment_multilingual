from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load model
classifier = pipeline("text-classification",
                      model="tabularisai/multilingual-sentiment-analysis")

app = FastAPI()


class TextInput(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "🚀 Sentiment Analysis API running"}


@app.post("/predict/")
def predict_sentiment(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "label": result["label"],
        "confidence": round(result["score"], 4)
    }
