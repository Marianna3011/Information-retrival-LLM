import pandas as pd
import os
from Evaluation import evaluate
import json

def evaluate_quac_sample(evaluate, quac_file, n_samples=5):
    with open(quac_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    
    for article in data[:n_samples]:
        for paragraph in article['paragraphs']:
            context = paragraph.get('context', paragraph.get('text', ''))  # Try 'context', fallback to 'text'
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text'] if qa['answers'] else ''  # Handle empty answers
                scores = evaluate(question, answer, context)
                print(f"Q: {question}")
                print(f"A: {answer}")
                print(f"Scores: {scores}")
                print("-" * 40)

# Example usage
evaluate_quac_sample(evaluate, 'val_v0.2.json')
