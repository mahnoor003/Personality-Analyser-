from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Minej/bert-base-personality")
model = AutoModelForSequenceClassification.from_pretrained("Minej/bert-base-personality")

def predict_personality(text):
    if not isinstance(text, str):
        text = str(text)
    if text.strip() == "" or text.lower() == "nan":
        return {trait: 0.0 for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).numpy()[0]
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    return dict(zip(traits, [float(np.round(p, 2)) for p in probabilities]))

def predict_personality_batch(texts):
    results = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        if text.strip() == "" or text.lower() == "nan":
            results.append({trait: 0.0 for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]})
            continue
        try:
            result = predict_personality(text)
            results.append(result)
        except Exception as e:
            print(f"⚠️ Skipping text due to error: {e}")
            results.append({trait: 0.0 for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]})
    return results