from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faithfulness_model_1 = pipeline("text2text-generation", model="google/flan-t5-base")
faithfulness_model_2 = pipeline("text2text-generation", model="google/flan-t5-large")

def parse_llm_score(response):
    try:
        for token in response.split():
            try:
                val = float(token)
                if -1.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0

def evaluate(question, answer, context):
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    emb_q = embedding_model.encode(question, convert_to_tensor=True)
    emb_a = embedding_model.encode(answer, convert_to_tensor=True)
    emb_c = embedding_model.encode(context, convert_to_tensor=True)

    question_answer_sim = util.cos_sim(emb_q, emb_a).item()
    answer_context_sim = util.cos_sim(emb_a, emb_c).item()

    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "On scale from -1 (-1 being absolutly tragic) to 1 (1 being perfect), how good is this answer to the question based on the context? "
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=10)[0]['generated_text'].strip()

    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2

    return round(avg_faithfulness, 3)

def evaluate_vanilla(question, answer, context):
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "On scale from -1 (-1 being absolutely tragic) to 1 (1 being perfect), how good is this answer to the question based on the context?"
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2
    return round(avg_faithfulness, 3)

def evaluate_cannotanswer_explicit(question, answer, context):
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "On scale from -1 (absolutely tragic) to 1 (perfect), rate the answer's quality based on the context. "
        "If the answer is 'CANNOTANSWER', rate it highly if the context truly does not contain the answer."
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2
    return round(avg_faithfulness, 3)

def evaluate_shorter_scale(question, answer, context):
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "Rate the answer from -1 (bad) to 1 (good) based only on the information in the context."
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2
    return round(avg_faithfulness, 3)

def evaluate_explanation(question, answer, context):
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "First, explain if the answer is supported by the context. Then, rate from -1 (not supported) to 1 (fully supported)."
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=30)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=30)[0]['generated_text'].strip()
    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2
    return round(avg_faithfulness, 3)

def evaluate_binary(question, answer, context):
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "Is the answer correct based on the context? Reply with 1 for correct, -1 for incorrect."
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=5)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=5)[0]['generated_text'].strip()
    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2
    return round(avg_faithfulness, 3)

def evaluate_average(question, answer, context):
    scores = [
        evaluate_vanilla(question, answer, context),
        evaluate_cannotanswer_explicit(question, answer, context),
        evaluate_shorter_scale(question, answer, context),
        evaluate_explanation(question, answer, context),
        evaluate_binary(question, answer, context),
    ]
    avg = sum(scores) / len(scores)
    return round(avg, 3)

def Evaluate(question, answer, context):
    if answer.strip().upper() == "CANNOTANSWER":
        return evaluate_cannotanswer_explicit(question, answer, context)
    else:
        return evaluate_shorter_scale(question, answer, context)


