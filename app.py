from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import re
# from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import OneHotEncoder
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import numpy as np
from model.data import labels, all_queries
app = Flask(__name__)
CORS(app)

# Placeholder order data
orders = {
    "123": "Your order is being processed.",
    "124": "Your order has been shipped.",
    "125": "Your order has been delivered."
}


def save_response_feedback_df(file_path='response_feedback.csv'):
    global response_feedback_df
    response_feedback_df.to_csv(file_path, index=False)


def load_response_feedback_df(file_path='response_feedback.csv'):
    global response_feedback_df
    try:
        response_feedback_df = pd.read_csv(file_path)
    except FileNotFoundError:
        response_feedback_df = pd.DataFrame(columns=['user_id', 'bot_response', 'feedback'])


load_response_feedback_df()

# Return policies information
return_policies = {
    "general_return_policies": "You can return most items within 30 days of purchase for a full refund or exchange. Items must be in their original condition, with all tags and packaging intact. Please bring your receipt or proof of purchase when returning items.",
    "exceptions_return_policies": "Certain items such as clearance merchandise, perishable goods, and personal care items are non-returnable. Please check the product description or ask a store associate for more details.",
    "refund_return_policies": "Refunds will be issued to the original form of payment. If you paid by credit card, the refund will be credited to your card. If you paid by cash or check, you will receive a cash refund."
}


def extract_order_id(text):
    # Extract potential order ID from the text
    match = re.search(r'\b\d{3}\b', text)
    if match:
        return match.group()
    return None

user_states = {}
def extract_contact_info(user_input):
  """
  Extracts name, email, and phone number from user input using spaCy NER.
  """
  # Combine email and phone number extraction into a single regular expression
  email_pattern = r"[a-z0-9\.\-+_]+@[a-z0-9\-]+\.[a-z]+"
  phone_pattern = r'\+?\d[\d\s\-]+'

  emails = re.findall(email_pattern,user_input,re.I)
  phones = re.findall(phone_pattern,user_input)
  import spacy

  nlp = spacy.load("en_core_web_sm")  # Load the English NER model
  doc = nlp(user_input)  # Process the text
  names = []
  for ent in doc.ents:
      if ent.label_ == "PERSON":
          names.append(ent.text)  # Append names to a list

  return {
      "Full Name": names,  # List of names
      "Email": emails,
      "Phone": phones
  }

def get_user_feedback():
    feedback = input("Was this information helpful? (yes/no): ").strip().lower()
    if feedback in ['yes', 'no']:
        return feedback
    else:
        print("Please respond with 'yes' or 'no'.")
        return get_user_feedback()


@app.route('/')
def index():
    return render_template('index.html')


model = load_model("model/model.h5")
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the pretrained model
# Initialize and fit the encoder on the labels
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(labels).reshape(-1, 1))



@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message").lower()
    user_id = request.json.get("user_id")

    if user_id not in user_states:
        user_states[user_id] = {}

    if user_states[user_id].get("waiting_for_order_id"):
        order_id = extract_order_id(user_input)
        if order_id:
            order_status = orders.get(order_id, "Order ID not found.")
            user_states[user_id]["waiting_for_order_id"] = False
            user_states[user_id]["last_response"] = order_status
            return jsonify({"reply": order_status, "feedback_request": True})
        else:
            return jsonify({"reply": "Please provide your order ID."})

    if user_states[user_id].get("waiting_for_contact_info"):
        contact_info = extract_contact_info(user_input)
        if all(contact_info.values()):
            contact_info_df = pd.DataFrame([contact_info])
            contact_info_df.to_csv('contact_info.csv', mode='a', header=False, index=False)
            user_states[user_id]["waiting_for_contact_info"] = False
            response = "Your request has been submitted. A human representative will contact you soon."
            user_states[user_id]["last_response"] = response
            return jsonify({"reply": response, "feedback_request": True})
        else:
            return jsonify({"reply": "Didn't succeed to recognize contact info. Please provide your full name, email, and phone number."})

    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_padded = pad_sequences(input_sequence, maxlen=20)
    prediction = model.predict(input_padded)
    # confidence = np.max(prediction)
    query_label = encoder.inverse_transform(prediction)[0][0]

    if query_label == "order":
        order_id = extract_order_id(user_input)
        if order_id:
            order_status = orders.get(order_id, "Order ID not found.")
            user_states[user_id]["last_response"] = order_status
            return jsonify({"reply": order_status, "feedback_request": True})
        else:
            user_states[user_id]["waiting_for_order_id"] = True
            return jsonify({"reply": "Please provide your order ID."})

    if query_label == "human":
        user_states[user_id]["waiting_for_contact_info"] = True
        response = "Please provide your full name, email, and phone number."
        user_states[user_id]["last_response"] = response
        return jsonify({"reply": response, "feedback_request": False})

    if query_label == "general":
        response = return_policies["general_return_policies"]
        user_states[user_id]["last_response"] = response
        return jsonify({"reply": response, "feedback_request": True})

    if query_label == "refund":
        response = return_policies["refund_return_policies"]
        user_states[user_id]["last_response"] = response
        return jsonify({"reply": response, "feedback_request": True})

    if query_label == "exceptions":
        response = return_policies["exceptions_return_policies"]
        user_states[user_id]["last_response"] = response
        return jsonify({"reply": response, "feedback_request": True})

    if query_label == "nonsense":
        response = "Sorry, I don't understand you:("
        return jsonify({"reply": response, "feedback_request": False})

@app.route('/feedback', methods=['POST'])
def feedback():
    user_id = request.json.get("user_id")
    feedback = request.json.get("feedback")

    if feedback and feedback.lower() in ['yes','no']:
        last_response = user_states[user_id].get(
            "last_response","No previous response"
            )
        store_response_feedback(user_id,last_response,feedback)
        user_states.pop(user_id,None)
        return jsonify({"reply": "Thank you for your feedback!"})
    else:
        return jsonify({"reply": "Please respond with 'yes' or 'no'."})


def store_response_feedback(user_id, bot_response, feedback=None, file_path='response_feedback.csv'):
    global response_feedback_df
    new_row = pd.DataFrame([{
        'user_id': user_id,
        'bot_response': bot_response,
        'feedback': feedback
    }])
    response_feedback_df = pd.concat([response_feedback_df, new_row], ignore_index=True)
    save_response_feedback_df(file_path)

def evaluate_feedback():
    load_response_feedback_df()
    positive_feedback = response_feedback_df['feedback'].str.lower().value_counts().get('yes', 0)
    negative_feedback = response_feedback_df['feedback'].str.lower().value_counts().get('no', 0)
    total_feedback = positive_feedback + negative_feedback
    if total_feedback > 0:
        satisfaction_rate = (positive_feedback / total_feedback) * 100
    else:
        satisfaction_rate = 0
    return {
        'total_feedback': total_feedback,
        'positive_feedback': positive_feedback,
        'negative_feedback': negative_feedback,
        'satisfaction_rate': satisfaction_rate
    }




if __name__ == '__main__':
    app.run(debug=True)

    # to read feedback uncomment this line
    # print(evaluate_feedback())

