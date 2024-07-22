import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def multi_sentiment_score(texts):
    sentiment_scores = []
    if len(texts) == 0:
        print("No texts to analyze")
        return [0]
    batch_size = min(64, len(texts))

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        batch_scores = (probabilities[:, 1] - probabilities[:, 0])
        sentiment_scores.extend(batch_scores.tolist())

    return sentiment_scores

