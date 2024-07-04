# 🤖 Ecommerce Agent

A smart, AI-powered chatbot to assist customers with order status updates, connecting to human representatives, and answering questions about return and refund policies.

## 📋 Table of Contents
- [Features](##Features)
- [Installation](##installation)
- [Usage](#usage)
- [Source Code](#source-code)
- [Documentation](#documentation)
- [Evaluation Report](#evaluation-report)

## ✨ Features
- **Order Status Updates:** Get real-time updates on your order status.
- **Human Representative:** Easily connect with a human representative for more detailed queries.
- **Return & Refund Policies:** Quickly access information on return and refund policies.
- **Garbage Query Handling:** Automatically detects and responds to irrelevant queries with "I don't understand you."

## 🛠️ Installation
Follow these steps to set up and run the chatbot locally.

### Prerequisites
- Python 3.7+
- TensorFlow
- Flask

### Clone the Repository
```bash
git clone https://github.com/morganizzzm/EcommerceAgent.git
cd EcommerceAgent
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download NLTK Data
```python
import nltk
nltk.download('wordnet')
```

### Usage 
## Start the Flask Server
```bash
python app.py
```
## Interact with the Chatbot
Open your browser and go to http://127.0.0.1:5000

## Examples of answers 

# Ask for refund
<img width="648" alt="Screenshot 2024-07-04 at 12 22 06" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/ec4d3edc-c547-4f22-8de6-76513045256d">


# Ask for human representative
<img width="635" alt="Screenshot 2024-07-04 at 12 09 41" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/92c26605-bfa1-4929-a11b-73d33785e2bf">

### 💻 Source Code

The source code for the agent, including deployment and conversation handling, can be found in the repository. 
> Key files include:

> * app.py: The main Flask application.
> * model/data.py: Contains the training data.
> * model/model.h5: The trained model.
> * model/tokenizer.pickle: The tokenizer used for text preprocessing.
> * template/index.html: The frontend code for the bot.

### 📄 Documentation

> Running and Testing the Agent
> * Run the Flask Server: Start the server using the command above.
> * Interact via Browser: Open the provided URL to interact with the chatbot.
> * Test Predefined Dialogues: Use a predefined set of queries to test the chatbot's responses.

### 📊 Evaluation Report

## Performance Metrics
> * *Accuracy*: Evaluated by comparing the chatbot's responses to predefined dialogues. The data for the dialogues is generated with the help of ChatGPT and can be found in model/data.py.
The accuracy is evaluated according to CrossEntropyLoss.
> * *Response Relevance*: Assessed by the relevance of the responses to the input queries.
> * *User Satisfaction*: Measured based on user feedback collected during interactions. 
<img width="628" alt="Screenshot 2024-07-04 at 12 15 57" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/777a2944-c503-4d01-aa7e-d626a1895a75">


## Predefined Dialogues
Query: "What is the return policy for items purchased at your store?"
Expected Response: Information on the return policy.
Query: "I want to talk to a human representative."
Expected Response: Request for contact details.
Query: "Buy me a chocolate cookie"
Expected Response: "I don't understand you."

## Performance Summary

Accuracy | Response Relevance | User Satisfaction
 ------------ | ------------- 
97.8% | High | 90%





