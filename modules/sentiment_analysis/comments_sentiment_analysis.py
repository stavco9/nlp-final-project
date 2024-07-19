import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to get sentiment score
def get_sentiment_score(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get positive and negative scores
    negative_score = probabilities[0][0].item()
    positive_score = probabilities[0][1].item()

    # Calculate sentiment score on a scale from -10 to 10
    sentiment_score = (positive_score - negative_score) * 10

    return sentiment_score

def multi_sentiment_score(texts):
    sentiment_scores = []
    for text in texts:
        sentiment_scores.append(get_sentiment_score(text))
    return sentiment_scores

