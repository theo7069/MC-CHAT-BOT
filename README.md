🎓 Montgomery College Admissions Chatbot

An AI-powered conversational assistant built with LangChain, OpenAI, ChromaDB, and Streamlit to answer questions about Montgomery College admissions, tuition, registration, and financial aid.

The chatbot uses retrieval-augmented generation (RAG) to fetch information directly from official MC web pages, embed them into a local Chroma vector database, and generate natural, conversational answers using OpenAI’s GPT models.

🚀 Features

💬 Conversational Memory: Maintains chat context across questions

🌐 Web Data Loader: Automatically scrapes and updates content from MC’s official pages

🧠 Vector Search: Uses ChromaDB + OpenAI embeddings for semantic retrieval

⚙️ Efficient Caching: Streamlit caching for faster reloads

🔍 Transparent Sources: Shows which page each answer came from

🔒 Local Persistence: Saves embeddings in chroma_db/ for offline reuse

🧩 Tech Stack

Python 3.11+

LangChain (Document Loading, RAG, Memory)

OpenAI API (GPT-4 / GPT-3.5 Turbo)

ChromaDB (Vector Store)

Streamlit (Frontend UI)

dotenv (Environment management)

🧰 Setup Instructions

Clone the repo:

git clone https://github.com/<yourusername>/mc-admissions-chatbot.git
cd mc-admissions-chatbot


Install dependencies:

pip install -r requirements.txt


Create .env file:

OPENAI_API_KEY=your_api_key_here


Run the app:

streamlit run app.py

🏫 Use Case

This project helps students, advisors, and staff quickly access accurate, up-to-date information about Montgomery College without navigating multiple pages — ideal for embedding on college websites, advising tools, or internal student support systems.

🌟 Future Improvements

Add PDF support for course catalogs

Integrate LangGraph for complex dialogue control

Add multilingual support (Spanish, Amharic, etc.)

Deploy via Streamlit Cloud or Azure App Service
