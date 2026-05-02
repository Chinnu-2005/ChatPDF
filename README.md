# 📄 ChatPDF — Chat with Multiple PDFs using RAG

> Upload any number of PDF documents and ask natural language questions — get answers grounded strictly in your documents, with full conversation memory.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Classic-1C3C3C?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Description

**ChatPDF** is a Retrieval-Augmented Generation (RAG) application built with Streamlit that lets you have a multi-turn conversation with one or more PDF files. It extracts text from your PDFs, splits it into chunks, indexes it in a local FAISS vector store using HuggingFace embeddings, and answers your questions using a free LLM (Qwen2.5-7B) served via the HuggingFace Inference Router — all without any paid API.

**Key highlights:**
- ✅ 100% free — uses HuggingFace's free serverless inference router
- ✅ Multi-PDF support — upload and query multiple documents at once
- ✅ Conversation memory — follow-up questions work correctly via history-aware retrieval
- ✅ Strictly grounded answers — the LLM only answers from your document content
- ✅ Local vector store — FAISS runs fully on CPU, no external database needed

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                    Streamlit Frontend                           │
└────────────────────────────┬────────────────────────────────────┘
                             │  question + chat history
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE (LangChain)                    │
│                                                                 │
│   ┌─────────────────────┐      ┌──────────────────────────┐    │
│   │  History-Aware      │      │   Stuff Documents Chain  │    │
│   │  Retriever          │─────▶│   (Answer Generation)    │    │
│   │                     │      │                          │    │
│   │  1. Rephrase Q      │      │  System: Answer ONLY     │    │
│   │     using history   │      │  from provided context   │    │
│   │  2. Retrieve top-k  │      │                          │    │
│   │     chunks          │      └──────────┬───────────────┘    │
│   └──────────┬──────────┘                 │                    │
│              │                            │                    │
│              ▼                            ▼                    │
│   ┌──────────────────┐       ┌────────────────────────────┐   │
│   │  FAISS Vector DB  │       │  Qwen2.5-7B-Instruct (LLM) │   │
│   │  (local, CPU)     │       │  via HuggingFace Router    │   │
│   │                   │       │  router.huggingface.co/v1  │   │
│   │  BAAI/bge-base-   │       └────────────────────────────┘   │
│   │  en-v1.5 embeddings│                                       │
│   └──────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │  PDF Upload → Text → Chunks → Embed
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│   PDF Files                                                     │
│      │                                                          │
│      ▼                                                          │
│   PyPDF2 (text extraction per page)                            │
│      │                                                          │
│      ▼                                                          │
│   CharacterTextSplitter                                         │
│   chunk_size=1000 | chunk_overlap=50 | separator="\n"          │
│      │                                                          │
│      ▼                                                          │
│   HuggingFaceEmbeddings → BAAI/bge-base-en-v1.5               │
│   (normalized, CPU)                                             │
│      │                                                          │
│      ▼                                                          │
│   FAISS Vector Store (in-memory)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Library / Model | Role |
|---|---|---|
| **UI** | Streamlit | Browser-based chat interface |
| **PDF Parsing** | PyPDF2 | Extract raw text from uploaded PDFs |
| **Text Splitting** | LangChain `CharacterTextSplitter` | Split text into 1000-char chunks with 50-char overlap |
| **Embeddings** | `BAAI/bge-base-en-v1.5` (HuggingFace) | Convert chunks to dense vectors on CPU |
| **Vector Store** | FAISS (local) | Similarity search over embedded chunks |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` | Answer generation via HF Inference Router |
| **Retriever** | LangChain History-Aware Retriever | Rephrase + retrieve relevant chunks per turn |
| **Chain** | LangChain `create_retrieval_chain` | Orchestrate retrieval → answer pipeline |

---

## 📁 Project Structure

```
CHATPDF/
├── .venv/                  # Python virtual environment
├── .env                    # Environment variables (API token)
├── .gitignore              # Git ignore rules
├── app.py                  # Main Streamlit application
└── requirements.txt        # Python dependencies
```

---

## ⚙️ How It Works — Step by Step

### Phase 1: Ingestion (when you click "Process")
1. Each uploaded PDF is read page-by-page using `PyPDF2.PdfReader`
2. All extracted text is concatenated into one large string
3. `CharacterTextSplitter` breaks it into overlapping chunks of ~1000 characters
4. Each chunk is embedded using `BAAI/bge-base-en-v1.5` running locally on CPU
5. All embeddings are indexed into a FAISS vector store held in memory

### Phase 2: Retrieval & Answer (when you ask a question)
1. Your question + full chat history is passed to the **history-aware retriever**
2. The LLM first **rephrases** your question as a standalone query (accounting for context like "what about the second one?")
3. The rephrased query is used to **retrieve top-3 most relevant chunks** from FAISS via cosine similarity
4. The retrieved chunks + your question + chat history are passed to `Qwen2.5-7B-Instruct`
5. The LLM answers **strictly from the retrieved context** — if the answer isn't there, it says "I don't know"
6. The answer is displayed and the exchange is appended to `chat_history` for future turns

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10 or higher
- A free [HuggingFace account](https://huggingface.co) with an API token

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ChatPDF.git
cd ChatPDF
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your HuggingFace token

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
Make sure **"Make calls to Inference Providers"** is enabled on the token.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit
python-dotenv
PyPDF2
langchain
langchain-text-splitters
langchain-huggingface
langchain-community
langchain-classic
langchain-core
langchain-openai
faiss-cpu
sentence-transformers
```

Install all at once:
```bash
pip install streamlit python-dotenv PyPDF2 langchain langchain-text-splitters \
            langchain-huggingface langchain-community langchain-classic \
            langchain-core langchain-openai faiss-cpu sentence-transformers
```

---

## 🎮 Usage Guide

1. **Open the app** at `http://localhost:8501`
2. In the **sidebar**, click **"Browse files"** and upload one or more PDF files
3. Click **"Process"** and wait for the spinner to finish — this builds the vector index
4. Once you see **"Ready! Ask your questions."**, type your question in the input box
5. The app will retrieve relevant sections and answer from your documents
6. Ask follow-up questions freely — the conversation history is maintained

**Tips:**
- You can upload multiple PDFs at once; they are all indexed together
- If you upload new PDFs and click Process again, the old index is replaced
- Answers are grounded strictly in your documents — the model won't hallucinate outside them
- Follow-up questions like "tell me more about that" work correctly due to history-aware retrieval

---

## 🔒 Environment Variables

| Variable | Description |
|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | Your HuggingFace access token for the Inference Router |

---

## 🛠️ Configuration

You can tune these parameters directly in `app.py`:

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `chunk_size` | `get_text_chunks()` | `1000` | Larger = more context per chunk |
| `chunk_overlap` | `get_text_chunks()` | `50` | Higher = less info lost at boundaries |
| `k` | `as_retriever()` | `3` | Number of chunks retrieved per query |
| `max_tokens` | `ChatOpenAI()` | `512` | Max length of each LLM response |
| `temperature` | `ChatOpenAI()` | `0` | 0 = deterministic, higher = creative |
| `model` | `ChatOpenAI()` | `Qwen/Qwen2.5-7B-Instruct` | Swap for any HF chat-compatible model |

---

## 🔄 Swapping the LLM

Any chat-compatible model on HuggingFace's Inference Router works. Just change the `model` field:

```python
llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",   # or any other
    openai_api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    openai_api_base="https://router.huggingface.co/v1",
    ...
)
```

Some options:

| Model | Notes |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Default — fast, free, reliable |
| `HuggingFaceH4/zephyr-7b-beta` | Good instruction following |
| `meta-llama/Llama-3.1-8B-Instruct` | Requires HF access approval |
| `mistralai/Mistral-7B-Instruct-v0.3` | Works via the router (not direct serverless) |

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — RAG orchestration
- [HuggingFace](https://huggingface.co) — Embeddings model & free LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — Fast vector similarity search
- [Streamlit](https://streamlit.io) — Python web UI framework
