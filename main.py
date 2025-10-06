from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- Document Loading ---------------- #
def load_and_process_docs():
    urls = [
        # Tuition-focused pages (important for your use case)
        "https://www.montgomerycollege.edu/paying-for-college/tuition/index.html",
        "https://www.montgomerycollege.edu/paying-for-college/tuition/current-rates.html",

        # Other admissions/support pages
        "https://www.montgomerycollege.edu/admissions",
        "https://www.montgomerycollege.edu/academics/index.html",
        "https://www.montgomerycollege.edu/admissions-registration/registration/index.html",
        "https://www.montgomerycollege.edu/admissions-registration/financial-aid-scholarships.html",
        "https://www.montgomerycollege.edu/counseling-and-advising/index.html",
    ]

    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    # Split into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


# ---------------- Vector Store ---------------- #
@st.cache_resource
def setup_vectorstore():
    splits = load_and_process_docs()
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    return vectorstore


# ---------------- LLM ---------------- #
@st.cache_resource
def setup_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ---------------- Streamlit App ---------------- #
st.title("MC Admissions Chatbot 🎓")
st.write("Ask me anything about Montgomery College admissions, tuition, or financial aid!")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize vectorstore and llm
vectorstore = setup_vectorstore()
llm = setup_llm()

# Chat input
if prompt := st.chat_input("Ask about admissions:"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        # Conversational chain with memory
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory
        )
        result = qa.invoke({"question": prompt})
        answer = result["answer"]

    # Display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)



