import json
import os
import random

def create_mock_data(filename, num_samples):
    os.makedirs('data/processed', exist_ok=True)
    filepath = os.path.join('data/processed', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Mocking data according to the dual-tower + hard negative requirements
            label = random.choice([0, 1])
            data = {
                "source_text": f"This is a major news event report containing factual details. News ID: {i}",
                "comment_text": f"This is a user comment responding to the news. Comment ID: {i}. It might agree or deviate.",
                "pos_comment": f"I completely agree with the news. The facts are correct. Pos ID: {i}",
                "neg_comment": f"This is fake news, the numbers are totally wrong. Neg ID: {i}",
                "label": label
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Successfully generated mock data: {filepath}")

if __name__ == "__main__":
    # Generate enough samples so that batch_size=64 will not crash
    create_mock_data('train.jsonl', 256)
    create_mock_data('val.jsonl', 128)
    create_mock_data('test.jsonl', 128)