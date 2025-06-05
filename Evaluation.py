from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faithfulness_model = pipeline("text2text-generation", model="google/flan-t5-base")

def evaluate(question, answer, context):
    emb_q = embedding_model.encode(question, convert_to_tensor=True)
    emb_a = embedding_model.encode(answer, convert_to_tensor=True)
    emb_c = embedding_model.encode(context, convert_to_tensor=True)

    question_answer_sim = util.cos_sim(emb_q, emb_a).item()
    answer_context_sim = util.cos_sim(emb_a, emb_c).item()

    # LLM-based faithfulness
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\nIs the answer supported by the context? Respond 'Yes' or 'No'."
    llm_response = faithfulness_model(prompt, max_new_tokens=10)[0]['generated_text'].strip().lower()

    faithfulness_score = 1.0 if "yes" in llm_response else 0.0

    # Combine
    return {
        "question_answer_similarity": round(question_answer_sim, 3),
        "answer_context_similarity": round(answer_context_sim, 3),
        "faithfulness_score": faithfulness_score
    }
