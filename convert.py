import json

def convert_quac_to_simple_format(quac_file, output_file, max_samples=None):
    """
    Converts QuAC dataset to a simple list of dicts with context, question, and answer.
    """
    with open(quac_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    simplified_data = []
    count = 0

    for article in data:
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', paragraph.get('text', ''))
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                is_impossible = qa.get('is_impossible', False)

                if is_impossible:
                    answer = "impossible"
                else:
                    answers = qa.get('answers', [])
                    if answers:
                        answer = answers[0].get('text', '')
                    else:
                        answer = "impossible"

                simplified_data.append({
                    "context": context,
                    "question": question,
                    "answer": answer
                })

                count += 1
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in simplified_data:
            out_f.write(json.dumps(item) + "\n")

    print(f"Converted {count} QA pairs to simplified format and saved to {output_file}")

# Example usage:
convert_quac_to_simple_format('val_v0.2.json', 'quac_simple_val.jsonl', max_samples=100)
