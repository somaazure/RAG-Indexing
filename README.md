```markdown
# 🧠 RAGify: A Minimal Retrieval-Augmented Generation (RAG) System with LangChain & OpenAI

A clean, modular RAG (Retrieval-Augmented Generation) pipeline built using LangChain, ChromaDB, and OpenAI's GPT — ideal for LLM-powered search, Q&A over private documents, and real-world GenAI prototypes.

---

## 🚀 Project Vision

Modern LLMs like ChatGPT struggle with up-to-date or domain-specific knowledge. This project solves that using **RAG (Retrieval-Augmented Generation)** — where we:

1. **Index local documents** (PDFs, `.txt`, etc.) as vector embeddings
2. **Query over them** using natural language
3. **Generate accurate answers** using OpenAI's `gpt-3.5-turbo`

📌 **Use Cases**:
- Chat with local files or docs
- Enterprise knowledge base search
- Legal/finance/healthcare domain-specific assistants
- LLM-backed intranet or internal wikis

---

## 🛠️ Tech Stack

| Layer               | Tool                          |
|--------------------|-------------------------------|
| LLM                 | OpenAI GPT (`gpt-3.5-turbo`) |
| Embeddings          | OpenAI Embeddings API        |
| Vector Store        | ChromaDB (local)             |
| Index Deduplication | LangChain `SQLRecordManager` |
| Query Interface     | CLI (extensible to web)      |
| Language            | Python 3.9+                  |

---

## 🧱 Features

- ✅ Local RAG setup — no cloud dependencies beyond OpenAI
- ✅ Efficient document indexing with deduplication
- ✅ One-line natural language Q&A
- ✅ Modular code: `Indexer`, `Query`, LLM interface
- ✅ SQLite-backed document change tracking
- ✅ Minimal and fast — perfect for learning or extension

---

## 📁 Folder Structure

```

ragify/
├── app.py              # Indexing script (creates vector store)
├── query\_rag.py        # CLI for asking questions
├── sample.txt          # Sample knowledge base file
├── .env                # Store your OpenAI API key
├── README.md           # This file
├── chroma\_store/       # Vector DB (auto-generated)
├── index\_log.db        # SQLite log for deduplication

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/ragify.git
cd ragify
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> You’ll need: `langchain`, `openai`, `chromadb`, `python-dotenv`

### 3️⃣ Add your OpenAI API key

Create a `.env` file:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4️⃣ Index your knowledge file

Replace `sample.txt` with your own `.txt` or document:

```bash
python app.py
```

### 5️⃣ Ask Questions!

```bash
python query_rag.py
```

---

## 💡 Example

> **User Input**:
> What is the capital city of India?

> **AI Response**:
> The capital city of India is New Delhi.

---

## 🧪 Test It Yourself

Try changing `sample.txt` to contain different information and re-run `app.py`. The system uses file hashes to avoid redundant re-indexing and assigns unique UUIDs to each chunk.

---

## 🌱 Roadmap

* [ ] Add PDF & DOCX support
* [ ] Add Streamlit/Gradio UI
* [ ] Add chunk filtering by metadata
* [ ] Add FAISS / PGVector as plug-and-play alternatives

This project demonstrates:

* 🔍 Understanding of vector databases & embeddings
* 🤖 Practical application of OpenAI GPT with local data
* 🧱 Modular Python architecture
* 🧠 End-to-end GenAI knowledge (RAG, dedup, prompt chaining)
* ⚙️ Tools like LangChain, ChromaDB, dotenv, and more

> ✨ I'm excited to apply these concepts in production-level AI/ML workflows. Let’s build something smarter together!

## 📜 License

MIT License — free to use and extend!


