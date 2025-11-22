📄 InsightDoc: AI-Powered PDF Assistant

InsightDoc is a modern, RAG-based (Retrieval-Augmented Generation) chatbot that allows users to chat with their PDF documents. Built using Google Gemini 2.5 Flash, LangChain, and Streamlit, it provides accurate, context-aware answers with a sleek, chat-like interface.

🚀 Key Features

⚡ Powered by Gemini 2.5 Flash: Uses Google's latest, high-speed model for generating responses.

🧠 Local Embeddings: Uses HuggingFace (all-MiniLM-L6-v2) for free and unlimited document processing.

💬 Modern Chat UI: Features a ChatGPT-style interface with chat bubbles and history.

📂 Multi-PDF Support: Upload and query multiple documents simultaneously.

💾 Vector Search: Utilizes FAISS for efficient similarity search and information retrieval.

🛠️ Tech Stack

Frontend: Streamlit (Custom CSS styled)

LLM: Google Gemini 2.5 Flash

Embeddings: HuggingFace (Sentence Transformers)

Vector Store: FAISS

Orchestration: LangChain

📸 Demo




💻 Installation & Setup

Follow these steps to run the project locally:

Clone the Repository

git clone [https://github.com/Kalanadhakshitha/InsightDoc-RAG-Chatbot.git]
cd InsightDoc-RAG-Chatbot


Create a Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install Dependencies

pip install -r requirements.txt


(Note: First run might download the embedding model ~100MB)

Set up Environment Variables
Create a .env file in the root directory and add your Google API key:

GOOGLE_API_KEY=your_google_api_key_here


Run the Application

streamlit run app.py
