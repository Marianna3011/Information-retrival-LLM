from transformers import pipeline, AutoTokenizer
from Evaluation import Evaluate
from typing import Dict, List, Any, Tuple
# import jsnon

qa_pipeline = pipeline(
    'question-answering',
    model='deepset/roberta-base-squad2'
    )
flan = pipeline('text2text-generation', model='google/flan-t5-base')
bert_pipeline = pipeline(
    'question-answering',
    model='bert-large-uncased-whole-word-masking-finetuned-squad'
)

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

dataset = 'quac_simple_val.jsonl'
samples: List[Dict[str, Any]] = []


def printResults(results: List[Dict[str, Any]]) -> None:
    for r in results:
        print("\n---")
        print(f"Q: {r['question']}")
        print(f"A (predicted): {r['predicted_answer']}")
        print(f"A (gold): {r['gold_answer']}")
        print(f"Faithfulness Score: {r['score']}")


def truncate_context(context: str,
                     question: str,
                     max_length: int = 512) -> str:
    """
    Truncates the context to fit within
    the maximum length allowed by the model.
    """
    tokens = tokenizer.encode(question,
                              context,
                              truncation=True,
                              max_length=max_length)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    if decoded.startswith(question):
        return decoded[len(question):].strip()
    return decoded


def answer_question(context: str, question: str) -> Tuple[str, float]:
    """
    Function that takes a context and a question, and returns the best answer
    taken from 3 different models: RoBERTa, BERT, and FLAN-T5.
    Args:
        context (str): The context in which the question is asked.
        question (str): The question to be answered.
    Returns:
        Tuple[str, float]: The best answer and its faithfulness score.
    """
    truncated_context = truncate_context(context, question)
    try:
        prediction = qa_pipeline(question=question, context=truncated_context)
        predicted_answer = prediction['answer']
    except Exception:
        predicted_answer = 'CANNOTANSWER'
    best_answer = predicted_answer
    best_score = Evaluate(question, predicted_answer, truncated_context)
    try:
        flan_prompt = (
            f'''Based on the context: {truncated_context};
            generate an answer to the question: {question}'''
        )
        predicted_answer_flan = flan(flan_prompt,
                                     max_new_tokens=128)[0]['generated_text']
    except Exception:
        predicted_answer_flan = 'CANNOTANSWER'
    score_flan = Evaluate(question, predicted_answer_flan, truncated_context)
    if score_flan > best_score:
        best_answer = predicted_answer_flan
        best_score = score_flan
    try:
        bert_score = bert_pipeline(question=question,
                                   context=truncated_context)
        predicted_answer_bert = bert_score['answer']
    except Exception:
        predicted_answer_bert = 'CANNOTANSWER'
    score_bert = Evaluate(question, predicted_answer_bert, truncated_context)
    if score_bert > best_score:
        best_answer = predicted_answer_bert
        best_score = score_bert
    return best_answer, best_score


context = input("Enter context: ")
question = input("Enter question: ")

print('Best answer: ', answer_question(context, question))
# with open(dataset, 'r') as f:
#     for line in f:
#         entry = json.loads(line.strip())
#         samples.append({
#             'context': entry['context'],
#             'question': entry['question'],
#             'gold_answer': entry['answer']
#         })
# results = []
# roberta_score = 0
# for sample in samples[:30]:
#     context = sample['context']
#     question = sample['question']
#     gold = sample['gold_answer']
#     try:
#         prediction = qa_pipeline(question = question, context = context)
#         predicted_answer = prediction['answer']
#     except Exception as e:
#         predicted_answer = 'CANNOTANSWER'
#     score = Evaluate(question, predicted_answer, context)
#     roberta_score += score
#     results.append({
#         'question': question,
#         'predicted_answer': predicted_answer,
#         'gold_answer': gold,
#         'score': score
#     })
# print(f'Roberta Score: {roberta_score / len(results)}')
# results = []
# flan_score = 0
# for sample in samples[30:60]:
#     context = sample['context']
#     question = sample['question']
#     prompt = f'''Based on the context: {context};
#                generate an answer to the question: {question}'''
#     gold = sample['gold_answer']
#     try:
#         predicted_answer = flan(prompt,
#                                   max_new_tokens=128)[0]['generated_text']
#     except Exception as e:
#         print(e)
#         predicted_answer = 'CANNOTANSWER'
#     score = Evaluate(question, predicted_answer, context)
#     flan_score += score
#     results.append({
#         'question': question,
#         'predicted_answer': predicted_answer,
#         'gold_answer': gold,
#         'score': score
#     })
# print(f'Flan Score: {flan_score / len(results)}')
# results = []
# bert_score = 0
# for sample in samples[60:90]:
#     context = sample['context']
#     question = sample['question']
#     gold = sample['gold_answer']
#     try:
#         prediction = bert_pipeline(question = question, context = context)
#         predicted_answer = prediction['answer']
#     except Exception as e:
#         predicted_answer = 'CANNOTANSWER'
#     score = Evaluate(question, predicted_answer, context)
#     bert_score += score
#     results.append({
#         'question': question,
#         'predicted_answer': predicted_answer,
#         'gold_answer': gold,
#         'score': score
#     })
#
# print(f'BERT Score: {bert_score / len(results)}')
