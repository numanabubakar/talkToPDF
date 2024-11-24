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

# Configure page with custom theme
st.set_page_config(
    page_title="PDF Talking",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header with a new title and style
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
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #333;
        }
        .custom-input {
            font-size: 18px;
            margin-top: 15px;
        }
        .button-container {
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Talk To Your PDF</div>', unsafe_allow_html=True)

user_question = st.text_input(
    "What do you want to ask from your PDF?", placeholder="Type your question here...", key="user_question"
)
if st.button("Ask"):

    if user_question:
        with st.spinner("PDF is typing..."):
            status_placeholder = st.empty()

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_question)

            prompt_template = """
            Answer the question using the provided context. If not available, respond with "Answer not found in the context."\n\nContext:\n{context}\nQuestion:\n{question}\n\nAnswer:
            """
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            status_placeholder.empty()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.markdown(f"### Answer:\n{response['output_text']}")
    else:
        st.warning("Please enter a question before submitting!")

with st.sidebar:
   
    pdf_docs = st.file_uploader(
        "Upload one or more PDFs for processing:", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process PDF(s)"):
        with st.spinner("Extracting and processing text..."):
            text = ""
            for pdf in pdf_docs:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    text += page.extract_text()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            st.success("PDF(s) processed successfully!")
st.markdown(
    """
    <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            color: #fff;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            # border-top: 1px solid #e9ecef;
        }
    </style>
    <footer>
        Developed by <b>Numan Abubakar</b> | © 2024 All Rights Reserved
    </footer>
    """,
    unsafe_allow_html=True,
)
