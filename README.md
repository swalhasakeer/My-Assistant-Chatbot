# 🤖 Swalha's Assistant – AI Chatbot with OpenAI API Support

Swalha's Assistant is a smart web-based chatbot powered by **Python**, **Flask**, and **scikit-learn**, with optional integration to **OpenAI's GPT models**. It classifies user input into intents using a local ML model, and can optionally switch to GPT (like `gpt-3.5-turbo`) if an API key is provided.

> 💬 Designed and built by **Swalha Sakeer**, a Data Engineering student with a passion for AI and ML.

---

## ✨ Features

- 🤖 AI chatbot with OpenAI API integration
- 🧠 Local fallback model using Naive Bayes & TF-IDF
- 💬 Friendly HTML web chat interface
- 🔧 Intent classification via `intents.json`
- 📝 Smart preprocessing using NLTK (tokenize, lemmatize, stopwords)
- 🔐 Secure API key management using `.env`
- 📊 Model evaluation with accuracy logging
- ⚠️ Graceful fallback if OpenAI API fails

---

## 📁 Project Structure

```bash
chatbot-project/
│
├── app.py # Flask backend
├── model.py # Intent classifier & GPT fallback logic
├── intents.json # Training data for local model
├── .env # Environment variables (OpenAI API key)
├── model.pkl # Trained local ML model (auto-generated)
├── vectorizer.pkl # Trained vectorizer (auto-generated)
└── templates/
└── index.html # Web frontend
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/swalha-chatbot.git
cd swalha-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare NLTK Data
Run this once to download required NLTK resources:

```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🔐 OpenAI Integration (Optional)
To use GPT-3.5 or GPT-4 for smarter responses:

1. Create a .env file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```
2. It will be loaded automatically by the app. If the key is missing, the app will gracefully fall back to the local ML model.

You can get your API key from OpenAI's platform ( https://platform.openai.com/ )

## 🌐 Run the Web App

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000/
```

## 🖥 Web Interface
- ✅ Type your message in the input field.

- 🤖 See instant responses from the chatbot.

- 💡 Interface is responsive and easy to customize (index.html in templates/).
  
- Screenshot of Chatbot

  ![image](https://github.com/user-attachments/assets/b5ad1152-6e7a-4291-991d-bdd96986cdb7)


---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

See the [LICENSE](LICENSE) file for full license details.
