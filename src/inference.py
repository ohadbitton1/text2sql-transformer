import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#לקאגל לא צריך src.
from data_processing import get_dataloaders
import re
import csv

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("creating tokenizer")
    tokenizer =  AutoTokenizer.from_pretrained("t5-small")
    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    model_path = 'checkpoints\model_epoch_3.pth'

    """
    # שינוי 1: נתיב מלא למודל החדש שאימנו בקאגל
    model_path = '/kaggle/working/model_checkpoints/model_epoch_3.pth'
    # =================================================================
    """

    print("loading weigths from state dict")
    trained_file = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(trained_file)
    model.to(device)
    model.eval()

    def generate_sql(question):
        request = "translate English to SQL: " + question
        inputs = tokenizer(request,return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=128,num_beams = 4)
        decoded_query = tokenizer.decode(outputs[0], skip_special_tokens = True )
        return decoded_query
    

    def extract_sql_from_string(text):

        match = re.search(r"'human_readable':\s*'(.*?)'", text)
        if match:
            return match.group(1)
        return "" # Return empty string if not found


    print("creationg data loaders")

    # =================================================================
    # שינוי 2: נתיבים מלאים לנתונים בסביבת קאגל
    train_path = "/kaggle/input/sql-data-new/train.csv"
    val_path = "/kaggle/input/sql-data-new/validation.csv"
    test_path = "/kaggle/input/sql-data-new/test.csv"
    # =================================================================

    max_token_length = tokenizer.model_max_length
    _, _, test_dataloader = get_dataloaders(train_path, val_path, test_path, 8 ,tokenizer, max_token_length )
    MAX_BATCH = 367
    total_samples = 0
    correct_predictions = 0
    num_batches = len(test_dataloader)
    # =================================================================
    # בשביל קאגל
    csv_output_path = '/kaggle/working/error_analysis.csv'
    # =================================================================
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Question', 'Expected_SQL', 'Generated_SQL', 'Batch_Loss'])
        print(f"Starting evaluation on  {MAX_BATCH} batches...")
        for i, batch in enumerate(test_dataloader):
            if i >= MAX_BATCH:
                break
            print(f"Processing batch {i + 1}/{MAX_BATCH}...")
            #to generate
            nl_questions = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens = True )
            sqls = tokenizer.batch_decode(batch['labels'], skip_special_tokens = True)
            #to claculate loss
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                batch_loss = outputs.loss.item()

            for question, sql in zip(nl_questions, sqls):
                generated_sql = generate_sql(question)
                expected_query = extract_sql_from_string(sql)
                generated_query = extract_sql_from_string(generated_sql)
                """
                # ================== בלוק דיבוג ==================
                print("-" * 50)
                print(f"  QUESTION: {question}")   # השאלה המקורית
                print(f"  EXPECTED: {expected_query.lower()}")         # ה-SQL הנכון (ground truth)
                print(f" GENERATED: {generated_query.lower()}") # ה-SQL שהמודל יצר
                print(f"     MATCH? {expected_query.lower() == generated_query.lower()}") # האם יש התאמה מדויקת?
                print("-" * 50)
                # ===============================================
                """
                is_match = (expected_query.lower() == generated_query.lower())
                if not is_match: 
                    csv_writer.writerow([question, expected_query, generated_query, batch_loss])   
                if is_match:
                    correct_predictions += 1
                total_samples += 1


        accuracy = (correct_predictions / total_samples) * 100

        print(f"Exact Match Accuracy: {accuracy:.2f}%")