import pandas as pd
import os
from Evaluation import evaluate
import json

def evaluate_quac_sample(evaluate, quac_file, n_samples=5):
    with open(quac_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    sample_count = 0
    for article in data:
        paragraphs = article.get('paragraphs', [])
        if not paragraphs:
            # Optionally, use background as context if available
            context = article.get('background', '')
            # No QAs to evaluate if no paragraphs, so skip
            continue
        for paragraph in paragraphs:
            context = paragraph.get('context', paragraph.get('text', ''))
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                answer = qa.get('answers', [{}])[0].get('text', '') if qa.get('answers') else ''
                scores = evaluate(question, answer, context)
                print(f"Q: {question}")
                print(f"A: {answer}")
                print(f"Scores: {scores}")
                print("-" * 40)
                sample_count += 1
                if sample_count >= n_samples:
                    return

# Example usage
evaluate_quac_sample(evaluate, 'val_v0.2.json', 1)
