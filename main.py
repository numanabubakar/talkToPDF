from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = FastAPI()

# Schema matching your frontend request
class ChatRequest(BaseModel):
    message: str
    userId: str  # you can later log or use this to store history per user

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # Load the vector index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(req.message, k=4)

        # Prompt template
        prompt_template = """
        You are a helpful and knowledgeable AI health Doctor. Use the medical information provided in the context below to answer the user's question.

        1. First, check relevant information about the disease, symptoms, causes, or treatments.
        2. Then, combine it with your own general medical knowledge to offer clear explanations, practical tips, and possible solutions.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        # AI model setup
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        # Ask the chain
        response = chain.invoke({"input_documents": docs, "question": req.message}, return_only_outputs=True)

        return {
            "userId": req.userId,
            "message": req.message,
            "answer": response["output_text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
