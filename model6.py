import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
You are a compassionate psychologist. Use the context below to generate a thoughtful, empathetic, and conversational response. Synthesize and reflect on the user's feelings, providing understanding and support. Keep your answer concise and avoid unnecessary repetition.

Context: {context}
Question: {question}

Generate a supportive response below that reflects empathy and care.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Load the model
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

# QA Retrieval Chain
def create_qa_chain(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # or 'cuda' if using an Intel GPU
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    return create_qa_chain(llm, db)

# Output function
def final_result(query):
    qa_chain = st.session_state.chain
    response = qa_chain({'query': query})
    return response

# User Intent Recognition Function
def recognize_intent(message):
    message = message.lower()
    if "hi" in message or "hello" in message:
        return "greeting"
    elif "help" in message or "support" in message:
        return "request_support"
    return "general_query"

# Streamlit app
st.title("Conflict Resolution Assistant")
st.write("How can i help you?")
st.sidebar.title("WELCOME TO CONFLICT RESOLUTION ASSISTANT")
st.sidebar.write("A Conflict Resolution Assistant is typically a role or tool designed to help individuals or groups navigate and resolve disputes or disagreements effectively.")

if 'chain' not in st.session_state:
    try:
        st.session_state.chain = qa_bot()
    except Exception as e:
        st.error(f"Error initializing QA bot: {e}")
        st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.write(msg)

user_input = st.text_input("You: ", "")

if st.button("Send") and user_input:
    try:
        intent = recognize_intent(user_input)

        if intent == "greeting":
            response_text = "Hello! I'm here to listen. What's on your mind?"
        elif intent == "request_support":
            response_text = "I’m here to provide support. Please tell me more about what you need help with."
        else:
            response = final_result(user_input)
            answer = response["result"]
            
            sources = response.get("source_documents", [])
            if sources:
                answer += "\nSources: " + ", ".join([doc.metadata['source'] for doc in sources])
            else:
                answer += "\nNo sources found."
            
            response_text = answer

        st.session_state.messages.append(f"You: {user_input}")
        st.session_state.messages.append(f"Bot: {response_text}")
        st.rerun()  # Refresh the app after processing
    except Exception as e:
        st.error(f"An error occurred: {e}")
