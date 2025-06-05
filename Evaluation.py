from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faithfulness_model_1 = pipeline("text2text-generation", model="google/flan-t5-base")
faithfulness_model_2 = pipeline("text2text-generation", model="google/flan-t5-large")  # You can change this model

def parse_llm_score(response):
    try:
        # Extract the first float between -1 and 1 from the response
        for token in response.split():
            try:
                val = float(token)
                if -1.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0  # Default if parsing fails

def evaluate(question, answer, context):
    emb_q = embedding_model.encode(question, convert_to_tensor=True)
    emb_a = embedding_model.encode(answer, convert_to_tensor=True)
    emb_c = embedding_model.encode(context, convert_to_tensor=True)

    question_answer_sim = util.cos_sim(emb_q, emb_a).item()
    answer_context_sim = util.cos_sim(emb_a, emb_c).item()

    # LLM-based faithfulness (scale -1 to 1)
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "On a scale from -1 (not supported at all) to 1 (fully supported), how well is the answer supported by the context? Respond with a single number between -1 and 1."
    )
    llm_response_1 = faithfulness_model_1(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    llm_response_2 = faithfulness_model_2(prompt, max_new_tokens=10)[0]['generated_text'].strip()

    score_1 = parse_llm_score(llm_response_1)
    score_2 = parse_llm_score(llm_response_2)
    avg_faithfulness = (score_1 + score_2) / 2

    return {
        "question_answer_similarity": round(question_answer_sim, 3),
        "answer_context_similarity": round(answer_context_sim, 3),
        "faithfulness_score_avg": round(avg_faithfulness, 3),
        "faithfulness_score_llm1": score_1,
        "faithfulness_score_llm2": score_2
    }
