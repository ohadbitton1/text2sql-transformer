from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

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

def extract_sql_from_string(text):

        match = re.search(r"'human_readable':\s*'(.*?)'", text)
        if match:
            return match.group(1)
        return "" # Return empty string if not found



app = Flask(__name__)
@app.route('/')
def home():
    message = {"message": "API is running!"}
    return jsonify(message)


@app.route('/generate-sql', methods = ['POST'])
def generate_sql():
    data = request.get_json()
    question_from_user = data["question"]
    request_for_model = "translate English to SQL: " + question_from_user
    inputs = tokenizer(request_for_model,return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128,num_beams = 4)
    decoded_query = tokenizer.decode(outputs[0], skip_special_tokens = True )
    clean_sql = extract_sql_from_string(decoded_query)
    
    return jsonify({"generated_sql": clean_sql})




if __name__ == "__main__":
    app.run(debug= True)
