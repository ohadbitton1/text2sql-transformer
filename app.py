from flask import Flask, jsonify, request
app = Flask(__name__)
@app.route('/')
def home():
    message = {"message": "API is running!"}
    return jsonify(message)


@app.route('/generate-sql', methods = ['POST'])
def generate_sql():
    data = request.get_json()
    question_from_user = data["question"]
    
    return jsonify({"recived question": question_from_user})




if __name__ == "__main__":
    app.run(debug= True)
