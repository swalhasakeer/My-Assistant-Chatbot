from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from model import chatbot_reply

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_msg = request.args.get("msg")
    if not user_msg:
        return jsonify({"reply": "Please enter a message."})
    bot_reply = chatbot_reply(user_msg)  # No API key passed; loaded from .env in model.py
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)