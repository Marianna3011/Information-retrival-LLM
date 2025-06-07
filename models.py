from transformers import pipeline
import json
from Evaluation import Evaluate

qa_pipeline = pipeline('question-answering', model = 'deepset/roberta-base-squad2')
flan = pipeline('text2text-generation', model = 'google/flan-t5-base')
bert_pipeline = pipeline('question-answering', model = 'bert-large-uncased-whole-word-masking-finetuned-squad')

dataset = 'quac_simple_val.jsonl'
samples = []

with open(dataset, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        samples.append({
            'context': entry['context'],
            'question': entry['question'],
            'gold_answer': entry['answer']
        })
            
results = []
print('\n ----- ROBERTA -----\n')
for sample in samples[:10]:
    context = sample['context']
    question = sample['question']
    gold = sample['gold_answer']
    
    try:
        prediction = qa_pipeline(question = question, context = context)
        predicted_answer = prediction['answer']
    except Exception as e:
        predicted_answer = 'CANNOTANSWER'
        
    score = Evaluate(question, predicted_answer, context)
    
    results.append({
        'question': question,
        'predicted_answer': predicted_answer,
        'gold_answer': gold,
        'score': score
    })
    
for r in results:
    print("\n---")
    print(f"Q: {r['question']}")
    print(f"A (predicted): {r['predicted_answer']}")
    print(f"A (gold): {r['gold_answer']}")
    print(f"Faithfulness Score: {r['score']}")
    
results = []
print('\n ----- FLAN -----\n')
for sample in samples[10:20]:
    context = sample['context']
    question = sample['question']
    prompt = f'Based on the context: {context}; generate an answer to the question: {question}'
    gold = sample['gold_answer']
    
    try:
        predicted_answer = flan(prompt, max_new_tokens=128)[0]['generated_text']
    except Exception as e:
        print(e)
        predicted_answer = 'CANNOTANSWER'
    
    score = Evaluate(question, predicted_answer, context)
    results.append({
        'question': question,
        'predicted_answer': predicted_answer,
        'gold_answer': gold,
        'score': score
    })
    
for r in results:
    print("\n---")
    print(f"Q: {r['question']}")
    print(f"A (predicted): {r['predicted_answer']}")
    print(f"A (gold): {r['gold_answer']}")
    print(f"Faithfulness Score: {r['score']}")
    
results = []
print('\n ----- BERT -----\n')
for sample in samples[20:30]:
    context = sample['context']
    question = sample['question']
    gold = sample['gold_answer']
    
    try:
        prediction = bert_pipeline(question = question, context = context)
        predicted_answer = prediction['answer']
    except Exception as e:
        predicted_answer = 'CANNOTANSWER'
        
    score = Evaluate(question, predicted_answer, context)
    
    results.append({
        'question': question,
        'predicted_answer': predicted_answer,
        'gold_answer': gold,
        'score': score
    })
    
for r in results:
    print("\n---")
    print(f"Q: {r['question']}")
    print(f"A (predicted): {r['predicted_answer']}")
    print(f"A (gold): {r['gold_answer']}")
    print(f"Faithfulness Score: {r['score']}")