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
> * *Accuracy*: In the context of this project, accuracy measures how often the chatbot's predicted response matches the expected response from a predefined set of queries and responses.
> * > Evaluation Method: The chatbot's accuracy is calculated by using a test set of dialogues. Each response generated by the chatbot is compared to the expected response, and the proportion of correct responses over the total number of responses is computed.
> * > Data Source: The predefined dialogues used for accuracy evaluation are stored in model/data.py. These dialogues include a variety of typical customer inquiries, such as questions about order status, return policies, and requests for human assistance.
> * > Loss Function: The accuracy is evaluated according to the CrossEntropyLoss function. This function calculates the difference between the predicted probability distribution and the actual distribution, providing a measure of the model's performance. A lower cross-entropy loss indicates better performance.
> * > Calculation: During training and evaluation, the CrossEntropyLoss is calculated for each prediction. The overall accuracy is derived by converting the probability distribution output by the softmax layer of the model into a discrete class prediction and comparing it to the actual class.
> * > Example Calculation: Suppose the chatbot correctly predicts 85 out of 100 test queries. The accuracy would be calculated as (85/100) * 100 = 85%.
> * *Response Relevance*: Assessed by the relevance of the responses to the input queries.
> * *User Satisfaction*: Measured based on user feedback collected during interactions. 
<img width="628" alt="Screenshot 2024-07-04 at 12 15 57" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/777a2944-c503-4d01-aa7e-d626a1895a75">


## Predefined Dialogues

# Order satus
- Query: "What is my order status?"
- Expected Response: Ask for order id.
<img width="740" alt="Screenshot 2024-07-04 at 12 05 51" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/17d97bf1-99e3-4122-bab4-36f2d3a2045f">


# Return policy 
- Query: "What is the return policy for items purchased at your store?"
- Expected Response: Information on the return policy.
<img width="638" alt="Screenshot 2024-07-04 at 12 41 27" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/65bc348c-cccc-4340-9468-7ca8d10d79de">


# Refund
- Query: "What is the return policy for items purchased at your store?"
- Expected Response: Information on the return policy.
# Ask for refund
<img width="648" alt="Screenshot 2024-07-04 at 12 22 06" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/ec4d3edc-c547-4f22-8de6-76513045256d">

# Non-returnable items
- Query: "Are there any items that cannot be returned under this policy?"
- Expected Response: Information on non-returnable items.
<img width="617" alt="Screenshot 2024-07-04 at 12 43 19" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/3b442d76-114d-4b30-b9da-46e056f045e7">

 
# Ask for human representative
- Query: "I want to talk to a human representative."
- Expected Response: Request for contact details.
<img width="635" alt="Screenshot 2024-07-04 at 12 09 41" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/92c26605-bfa1-4929-a11b-73d33785e2bf">

# Nonsense request 
- Query: "Buy me a chocolate cookie"
- Expected Response: "I don't understand you."
<img width="636" alt="Screenshot 2024-07-04 at 12 44 09" src="https://github.com/morganizzzm/EcommerceAgent/assets/89296464/d1a1629f-6ae3-4fe8-83e4-e14cac6c4efa">




## Performance Summary

Accuracy | Response Relevance | User Satisfaction
 ------- | ------------- | ------------- 
97.8% | High | 90%





