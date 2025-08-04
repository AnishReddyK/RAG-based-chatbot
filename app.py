import streamlit as st
from pypdf import PdfReader
import re
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Setup Gemini API
os.environ["GEMINI_API_KEY"] = 'AIzaSyA1TzXYCxBnJzIK52jQlyBPKXgbNMBM8bE'
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Streamlit UI
st.set_page_config(page_title="📄 PDF Chatbot")
st.title("📚 Ask Questions from a PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    reader = PdfReader("uploaded.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Custom embedding function using Gemini
    class GeminiEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [
                genai.embed_content(
                    model="models/embedding-001",
                    content=txt,
                    task_type="retrieval_document",
                    title="PDF Chatbot"
                )["embedding"]
                for txt in texts
            ]

        def embed_query(self, query: str) -> list[float]:
            return genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"  # ✅ title removed
            )["embedding"]


    # Initialize FAISS vector store
    embeddings = GeminiEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # Ask questions
    query = st.text_input("Ask something based on the PDF:")
    if query:
        results = vectorstore.similarity_search(query, k=3)
        context = "\n".join([r.page_content for r in results])

        # Gemini answer generation
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(f"Context:\n{context}\n\nQuestion:\n{query}")
        st.markdown(f"**Answer:** {response.text}")
