# from flask import Flask, render_template, request, jsonify
# import openai
# import requests
# import nlpSVM as nlp  # Import the custom nlp module
# import nltk
# import pandas as pd
# import LineNoti

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# app = Flask(__name__)

# # Set your OpenAI API key
# openai.api_key = 'sk-proj-5D7Q7CGd3tstFdVl9LCDT3BlbkFJ8dD3TJd4NPJPhbl48o8L'

# # Load and preprocess data for NLP model
# data_filepath = 'nlp.csv'  # Adjust this path to your CSV file
# df = nlp.load_and_preprocess_data(data_filepath)
# featuresets = nlp.prepare_features(df)
# classifier, vectorizer, accuracy = nlp.train_model(featuresets)

# # Initialize user and response logs
# User = []
# Chatgptresponse = []

# @app.route('/')
# def index():
#     return render_template('chat.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_message = request.json['message']
#         User.append(user_message)
#         print(User)
        
#         # Predict user's mood
#         predicted_emotion = nlp.predict_emotion(classifier, user_message, vectorizer)
#         emotion_message = f"Your mood is {predicted_emotion}."
#         print(emotion_message)
#         if predicted_emotion == "depressed":
#             print(
#                 " It seems you might be feeling down. Please consider reaching out to a healthcare professional for support."
#                 " Here are some contact details for your reference:\n"
#                 " - Dr. John Smith\n"
#                 " - Phone: +1234567890\n"
#                 " - Facebook: fb.com/drjohnsmith\n"
#                 " - Email: dr.johnsmith@example.com"
#             )
#             LineNoti.line_notify(predicted_emotion + " : " + user_message)
#         else:
#             print("I think you are feeling good.")

#         # Get response from OpenAI API
#         api_url = "https://api.openai.com/v1/chat/completions"
#         headers = {
#             'Content-Type': 'application/json',
#             'Authorization': f'Bearer {openai.api_key}',
#         }
#         body = {
#             "model": "gpt-4",
#             "max_tokens": 250,
#             "messages": [
#                 {"role": "user", "content": user_message},
#                 {"role": "system", "content": f"The user seems to be feeling {predicted_emotion}."}
#             ]
#         }

#         # response = requests.post(api_url, headers=headers, json=body)
#         # data = response.json()
#         # answer_from_openai = data['choices'][0]['message']['content']
        
#         # # Combine the mood prediction with the OpenAI response
#         # full_response = f"{answer_from_openai}"
#         full_response = "Verry goodbye"
#         if predicted_emotion == "depressed":
#             full_response += (
#                 "<br><br>It seems you might be feeling down. "
#                 "Please consider reaching out to a healthcare professional for support.<br>"
#                 "Here are some contact details for your reference:<br>"
#                 "Dr. Natthawadee Kaewwongsa<br>"
#                 "Phone: +1234567890<br>"
#                 "Facebook: fb.com/drjohnsmith<br>"
#                 "Email: dr.johnsmith@example.com"
#             )
            
#         Chatgptresponse.append(full_response)

#         return jsonify({"response": full_response})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import openai
import requests
import nlpSVM as nlp  # Import the custom nlp module
import nltk
import pandas as pd
import LineNoti

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = 'sk-proj-5D7Q7CGd3tstFdVl9LCDT3BlbkFJ8dD3TJd4NPJPhbl48o8L'

# Load the saved model and vectorizer
classifier, vectorizer = nlp.load_model_and_vectorizer()

# Initialize user and response logs
User = []
Chatgptresponse = []

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        User.append(user_message)
        print(User)
        
        # Predict user's mood
        predicted_emotion = nlp.predict_emotion(classifier, user_message, vectorizer)
        emotion_message = f"Your mood is {predicted_emotion}."
        print(emotion_message)
        if predicted_emotion == "depressed":
            print(
                "It seems you might be feeling down. Please consider reaching out to a healthcare professional for support."
                "Here are some contact details for your reference:\n"
                "- Dr. John Smith\n"
                "- Phone: +1234567890\n"
                "- Facebook: fb.com/drjohnsmith\n"
                "- Email: dr.johnsmith@example.com"
            )
            LineNoti.line_notify(predicted_emotion + " : " + user_message)

        # Get response from OpenAI API
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai.api_key}',
        }
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "system", "content": f"The user seems to be feeling {predicted_emotion}."}
            ]
        }

        response = requests.post(api_url, headers=headers, json=body)
        data = response.json()
        answer_from_openai = data['choices'][0]['message']['content']
        
        # Combine the mood prediction with the OpenAI response
        full_response = f"{answer_from_openai}"
        if predicted_emotion == "depressed":
            full_response += (
                "<br><br>It seems you might be feeling down. "
                "Please consider reaching out to a healthcare professional for support.<br>"
                "Here are some contact details for your reference:<br>"
                "Dr. Natthawadee Kaewwongsa<br>"
                "Phone: +1234567890<br>"
                "Facebook: fb.com/drjohnsmith<br>"
                "Email: dr.johnsmith@example.com"
            )
        Chatgptresponse.append(full_response)

        return jsonify({"response": full_response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
