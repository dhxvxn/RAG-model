import streamlit as st
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from openai.error import RateLimitError
import fitz

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# ==========================
# LLMs
# ==========================
openai_llm = ChatOpenAI(
    api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0
)

groq_llm = ChatGroq(
    api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0
)

# ==========================
# Configuration
# ==========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)

# ==========================
# Functions
# ==========================
def load_pdf(file):
    """Load PDF and return Langchain Document objects."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return [Document(page_content=text)]

def split_text(documents):
    """Split a list of Document objects into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    contents = [doc.page_content for doc in documents]
    docs_to_split = [Document(page_content=content) for content in contents]
    return text_splitter.split_documents(docs_to_split)

def create_vectorstore(chunks):
    """Create a FAISS vector store from text chunks."""
    return FAISS.from_documents(chunks, embeddings)

def create_qa_chain(llm, vectorstore):
    """Create a RetrievalQA chain for a given LLM and vectorstore."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=True
    )

def ask_question(vectorstore, user_question):
    """
    Ask a question using OpenAI first, then silently fallback to Groq if needed.
    """
    qa_chain_openai = create_qa_chain(openai_llm, vectorstore)
    try:
        return qa_chain_openai({"query": user_question})
    except RateLimitError:
        qa_chain_groq = create_qa_chain(groq_llm, vectorstore)
        return qa_chain_groq({"query": user_question})

# ==========================
# Streamlit UI
# ==========================
st.title("📄 PDF Q&A App with LangChain + OpenAI + Groq + HuggingFace")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    with st.spinner("Processing PDF..."):
        documents = load_pdf(uploaded_file)
        chunks = split_text(documents)
        vectorstore = create_vectorstore(chunks)
    st.success("✅ PDF processed successfully!")

    user_question = st.text_input("Enter your question about the PDF:")
    if user_question:
        with st.spinner("Generating answer..."):
            result = ask_question(vectorstore, user_question)
            answer = result["result"]
            sources = result["source_documents"]

        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Source Documents:")
        for i, src in enumerate(sources):
            st.write(f"{i + 1}. {src.page_content[:200]}...")
