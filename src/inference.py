import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data_processing import get_dataloaders
import re

if __name__ == "__main__":
    print("creating tokenizer")
    tokenizer =  AutoTokenizer.from_pretrained("t5-small")
    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    print("loading weigths from state dict")
    trained_file = torch.load('checkpoints/model_epoch_2.pth', map_location=torch.device('cpu'))
    model.load_state_dict(trained_file)

    model.eval()

    def generate_sql(question):
        request = "translate English to SQL: " + question
        inputs = tokenizer(request,return_tensors="pt")
        outputs = model.generate(**inputs, max_length=128,num_beams = 4)
        decoded_query = tokenizer.decode(outputs[0], skip_special_tokens = True )
        return decoded_query
    

    def extract_sql_from_string(text):

        match = re.search(r"'human_readable':\s*'(.*?)'", text)
        if match:
            return match.group(1)
        return "" # Return empty string if not found


    print("creationg data loaders")

    max_token_length = tokenizer.model_max_length
    _, _, test_dataloader = get_dataloaders('data/train.csv','data/validation.csv', 'data/test.csv', 8 ,tokenizer, max_token_length )

    LIMIT_BATCHES = 5
    total_samples = 0
    correct_predictions = 0
    num_batches = len(test_dataloader)
    print(f"Starting evaluation on {LIMIT_BATCHES} batches (out of {num_batches} total)...")
    for i, batch in enumerate(test_dataloader):
        if i >= LIMIT_BATCHES:
         print("Reached batch limit for this test run. Stopping.")
         break
        print(f"Processing batch {i + 1}/{LIMIT_BATCHES}...")
        nl_questions = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens = True )
        sqls = tokenizer.batch_decode(batch['labels'], skip_special_tokens = True)
        for question, sql in zip(nl_questions, sqls):
            generated_sql = generate_sql(question)
            expected_query = extract_sql_from_string(sql)
            generated_query = extract_sql_from_string(generated_sql)
            # ================== בלוק דיבוג ==================
            print("-" * 50)
            print(f"  QUESTION: {question}")   # השאלה המקורית
            print(f"  EXPECTED: {expected_query.lower()}")         # ה-SQL הנכון (ground truth)
            print(f" GENERATED: {generated_query.lower()}") # ה-SQL שהמודל יצר
            print(f"     MATCH? {expected_query.lower() == generated_query.lower()}") # האם יש התאמה מדויקת?
            print("-" * 50)
            # ===============================================
            if expected_query.lower() == generated_query.lower():
                 correct_predictions += 1
            total_samples += 1


    accuracy = (correct_predictions / total_samples) * 100

    print(f"Exact Match Accuracy: {accuracy:.2f}%")