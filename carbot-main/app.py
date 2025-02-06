from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
import threading
import time

app = Flask(__name__)

# Initialize the InferenceClient with your model and token directly
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    token="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",  # Replace with your actual token
)

# Comprehensive list of keywords related to automotive topics
KEYWORDS = [
    "car", "automobile", "vehicle", "engine", "transmission", "brakes", "battery", 
    "oil change", "maintenance", "tire", "diagnostic", "repair", "troubleshooting", 
    "check engine light", "dashboard warning", "AC", "cooling system", "exhaust", 
    "suspension", "alignment", "fuel system", "ignition", "steering", "air filter", 
    "spark plug", "radiator", "starter", "alternator", "belt", "hose", "emissions", 
    "safety inspection", "leak", "wiper", "headlights", "taillights", "mirrors", 
    "battery life", "brake fluid", "power steering", "coolant", "transmission fluid", 
    "fuel efficiency", "engine overheating", "jump start", "flat tire", "jack", "tool kit"
]

def is_relevant_to_auto(text):
    """
    Checks if the text contains any of the keywords related to automotive topics.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in KEYWORDS)

def generate_response(user_message, response_queue):
    """
    Generates a response using the model and checks if the response is relevant to automotive topics.
    """
    try:
        response = ""
        # Modified prompt to guide the model for automotive expertise
        prompt = f"You are a knowledgeable assistant focused on automobiles. Only answer questions related to cars, maintenance, repairs, troubleshooting, and general automotive advice. If the question is not related to these topics, politely ask the user to ask something else. Question: {user_message}"
        
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
            stream=True  # Use streaming
        ):
            if 'choices' in message and message['choices']:
                response += message['choices'][0]['delta']['content']
        
        # Check if the response is relevant to automotive topics
        if not is_relevant_to_auto(response):
            response = "The response generated was not relevant to automobiles. Please ask questions related to automotive topics or vehicle issues."

        response_queue.append(response)
    except Exception as e:
        print(f"Error in generate_response: {e}")
        response_queue.append("An error occurred while generating the response.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat requests.
    """
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    # Check if the user's question is relevant to automotive topics
    if not is_relevant_to_auto(user_message):
        return jsonify({'reply': 'Please ask questions related to automotive topics or vehicle issues.'})
    
    try:
        start_time = time.time()
        
        # Use a list as a thread-safe queue to collect the response
        response_queue = []
        # Run response generation in a separate thread to avoid blocking
        thread = threading.Thread(target=generate_response, args=(user_message, response_queue))
        thread.start()
        thread.join()  # Wait for the thread to complete
        
        response = response_queue[0] if response_queue else "No response received"

        # Log response time for performance monitoring
        elapsed_time = time.time() - start_time
        print(f"Response Time: {elapsed_time:.2f} seconds")

        return jsonify({'reply': response})
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable reloader for production
