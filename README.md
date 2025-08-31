# 📄 PDF Q&A App (Streamlit + LangChain + OpenAI + Groq)

A Streamlit-based application that allows you to upload a PDF, process its content using FAISS and HuggingFace embeddings, and ask questions using OpenAI GPT and Groq as fallback.

---

## ✅ Features
- Upload PDF and extract text
- Split text into chunks for better retrieval
- Use **FAISS** for vector search
- **OpenAI GPT-3.5** for answers
- **Groq LLM** as fallback when OpenAI hits Rate Limits

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/pdf-qa-app.git
cd pdf-qa-app
