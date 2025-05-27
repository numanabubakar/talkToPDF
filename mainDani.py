import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure page
st.set_page_config(page_title="Talk to Your Docter", page_icon="📚", layout="wide")

# Custom header
st.markdown(
    """
    <style>
        .main-header {
            font-size: 32px;
            color: #1a73e8;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            # border: 1px solid #ddd;
            # border-radius: 10px;
            padding: 10px;
            # background-color: #f9f9f9;
            margin-bottom: 10px;
            display: flex;
            flex-direction: column-reverse; /* Show latest messages at the bottom */
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            line-height: 1.5;
        }
        .user-message {
            # background-color: #e6f4ff;
            text-align: right;
        }
        .bot-message {
            # background-color: #;
            text-align: left;
        }
        .input-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .input-box {
            flex: 1;
        }
        .send-button {
            background-color: #1a73e8;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 16px;
        }
        .send-button:hover {
            background-color: #1558b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Talk To Your Docter</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF processing
# with st.sidebar:
    # pdf_file_path = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
    # if st.button("Process PDF"):
    #     with st.spinner("Processing PDF..."):
    #         if not os.path.exists(pdf_file_path):
    #             st.error(f"PDF file '{pdf_file_path}' not found in the root directory.")
    #         else:
    #             try:
    #                 text = ""
    #                 reader = PdfReader(pdf_file_path)
    #                 for page in reader.pages:
    #                     extracted_text = page.extract_text()
    #                     if extracted_text:
    #                         text += extracted_text

    #                 splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #                 chunks = splitter.split_text(text)

    #                 embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #                 vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    #                 vector_store.save_local("faiss_index")
    #                 st.success("PDF processed successfully!")
    #             except Exception as e:
    #                 st.error(f"Error processing PDF: {str(e)}")

# Function to handle sending user input
def handle_send():
    if st.session_state.user_input.strip():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})

        # Process bot response
        with st.spinner("Docter is Examining..."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = db.similarity_search(st.session_state.user_input, k=4)

                prompt_template = """
                You are a helpful and knowledgeable AI health Doctor. Use the medical information provided in the context below to answer the user's question.

                1. First, check relevant information about the disease, symptoms, causes, or treatments.
                2. Then, combine it with your own general medical knowledge to offer clear explanations, practical tips, and possible solutions
                

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
                
                model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key="AIzaSyBXlDdWc9X9F1WXpLyqmyfjihVz-dcS8_Y")
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                response = chain.invoke({"input_documents": docs, "question": st.session_state.user_input}, return_only_outputs=True)

                # Add bot response to chat history
                st.session_state.messages.append({"role": "bot", "content": response['output_text']})
            except Exception as e:
                st.session_state.messages.append({"role": "bot", "content": "Error: Could not process your question."})

        # Clear the input box
        st.session_state.user_input = ""

# Display chat history container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="message user-message"><b>You:</b> {message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="message bot-message"><b>Your Docter:</b> {message["content"]}</div>',
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

# Input container at the bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.text_input(
    "",
    key="user_input",
    placeholder="Ask anything to your Doctor...",
    label_visibility="collapsed",  # Hide the label
    on_change=handle_send,
    help="Type your question here.",
)
st.button("Ask", on_click=handle_send, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            color: #808080;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
    </style>
    <footer>
        Developed by <b>Asif Danyal</b> | © 2025 All Rights Reserved
    </footer>
    """,
    unsafe_allow_html=True,
)