import pandas as pd
import os
from Evaluation import (
    evaluate_vanilla,
    evaluate_cannotanswer_explicit,
    evaluate_shorter_scale,
    evaluate_explanation,
    evaluate_binary,
    evaluate_average,
    Evaluate as evaluate_hybrid_shorter_or_cannotanswer 
)
import json
import numpy as np

def evaluate_quac_sample(evaluate, quac_file, n_samples=5):
    with open(quac_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    sample_count = 0
    for article in data:
        paragraphs = article.get('paragraphs', [])
        if not paragraphs:
            context = article.get('background', '')
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

def evaluate_quac_matrix(quac_file, n_samples=5):
    eval_funcs = [
        ("vanilla", evaluate_vanilla),
        ("cannotanswer_explicit", evaluate_cannotanswer_explicit),
        ("shorter_scale", evaluate_shorter_scale),
        ("explanation", evaluate_explanation),
        ("binary", evaluate_binary),
        ("average", evaluate_average),
        ("hybrid_shorter_or_cannotanswer", evaluate_hybrid_shorter_or_cannotanswer),
    ]
    with open(quac_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    sample_count = 0
    print("Q/A/Context".ljust(60), end="")
    for name, _ in eval_funcs:
        print(f"{name:>25}", end="")
    print()
    print("-" * (60 + 25 * len(eval_funcs)))

    for article in data:
        paragraphs = article.get('paragraphs', [])
        if not paragraphs:
            continue
        for paragraph in paragraphs:
            context = paragraph.get('context', paragraph.get('text', ''))
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                answer = qa.get('answers', [{}])[0].get('text', '') if qa.get('answers') else ''
                row_label = f"Q: {question} | A: {answer}".ljust(60)
                print(row_label, end="")
                for _, func in eval_funcs:
                    try:
                        score = func(question, answer, context)
                    except Exception as e:
                        score = "ERR"
                    print(f"{str(score):>25}", end="")
                print()
                sample_count += 1
                if sample_count >= n_samples:
                    return

def evaluate_quac_sample_matrix(quac_file, n_samples=5):
    eval_funcs = [
        ("vanilla", evaluate_vanilla),
        ("cannotanswer_explicit", evaluate_cannotanswer_explicit),
        ("shorter_scale", evaluate_shorter_scale),
        ("explanation", evaluate_explanation),
        ("binary", evaluate_binary),
        ("average", evaluate_average),
        ("hybrid_shorter_or_cannotanswer", evaluate_hybrid_shorter_or_cannotanswer),
    ]
    with open(quac_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    samples = []
    for article in data:
        paragraphs = article.get('paragraphs', [])
        if not paragraphs:
            continue
        for paragraph in paragraphs:
            context = paragraph.get('context', paragraph.get('text', ''))
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                answer = qa.get('answers', [{}])[0].get('text', '') if qa.get('answers') else ''
                samples.append((question, answer, context))
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    print("Case".ljust(10), end="")
    print(f"{'target':>10}", end="")
    for name, _ in eval_funcs:
        print(f"{name:>25}", end="")
    print()
    print("-" * (10 + 10 + 25 * len(eval_funcs)))

    errors = [[] for _ in eval_funcs]
    for idx, (question, answer, context) in enumerate(samples):
        label = f"{idx+1}".ljust(10)
        print(label, end="")
        print(f"{'1':>10}", end="")
        for j, (_, func) in enumerate(eval_funcs):
            try:
                score = func(question, answer, context)
                error = abs(score - 1) if isinstance(score, (int, float)) else None
            except Exception:
                score = "ERR"
                error = None
            print(f"{str(score):>25}", end="")
            errors[j].append(error)
        print()


    print("\nMean Absolute Error (MAE) for each evaluation function:")
    for (name, _), err_list in zip(eval_funcs, errors):
        filtered = [e for e in err_list if e is not None]
        mae = np.mean(filtered) if filtered else "N/A"
        print(f"{name:>25}: {mae}")

evaluate_quac_sample_matrix('val_v0.2.json', n_samples=5)

test_cases = [
    {
        "question": "Who is the author of the book?",
        "answer": "The author of the book is John Doe.",
        "context": "George Orwell was the author of the book."
    },
    {
        "question": "Who is the author of the book?",
        "answer": "The author of the book is John Doe.",
        "context": "John Doe was the author of the book."
    },
    {
        "question": "Who is the author of the book?",
        "answer": "CANNOTANSWER",
        "context": "John Doe was the author of the book."
    },
    {
        "question": "Who is the author of the book?",
        "answer": "CANNOTANSWER",
        "context": "Icecream is a delicious treat."
    },
    {
        "question": "Who is the author of the book?",
        "answer": "Eve Smith icecream dough",
        "context": "John Doe was some random guy"
    }
]

eval_funcs = [
    ("vanilla", evaluate_vanilla),
    ("cannotanswer_explicit", evaluate_cannotanswer_explicit),
    ("shorter_scale", evaluate_shorter_scale),
    ("explanation", evaluate_explanation),
    ("binary", evaluate_binary),
    ("average", evaluate_average),
    ("hybrid_shorter_or_cannotanswer", evaluate_hybrid_shorter_or_cannotanswer),
]
targets = [-1, 1, 0, 1,-1]

print("Case".ljust(10), end="")
print(f"{'target':>10}", end="") 
for name, _ in eval_funcs:
    print(f"{name:>25}", end="")
print()
print("-" * (10 + 10 + 25 * len(eval_funcs)))

for idx, case in enumerate(test_cases):
    label = f"{idx+1}".ljust(10)
    print(label, end="")
    print(f"{str(targets[idx]):>10}", end="")
    for _, func in eval_funcs:
        try:
            score = func(case['question'], case['answer'], case['context'])
        except Exception as e:
            score = "ERR"
        print(f"{str(score):>25}", end="")
    print()

errors = [[] for _ in eval_funcs]

for idx, case in enumerate(test_cases):
    for j, (_, func) in enumerate(eval_funcs):
        try:
            score = func(case['question'], case['answer'], case['context'])

            error = abs(score - targets[idx]) if isinstance(score, (int, float)) else None
        except Exception:
            error = None
        errors[j].append(error)

print("\nMean Absolute Error (MAE) for each evaluation function:")
for (name, _), err_list in zip(eval_funcs, errors):

    filtered = [e for e in err_list if e is not None]
    mae = np.mean(filtered) if filtered else "N/A"
    print(f"{name:>25}: {mae}")

