# Agentic Document RAG System

A multi‑agent Retrieval‑Augmented Generation (RAG) system that answers complex
questions about long PDF documents using a **local FAISS‑style vector store** and
a team of specialised agents:

- 🧠 **Planner Agent** – turns questions into focused hypotheticals
- 🔍 **Retrieval Service** – chunks PDFs and runs dense vector search (FAISS / NumPy)
- ✍️ **Synthesizer Agent** – calls an LLM (Gemini or OpenAI) to craft structured answers
- ✅ **Fact‑Checker Agent** – checks answers against context and flags hallucinations

This repo is intentionally tuned for clarity and learning — you can read the code
end‑to‑end and see how each agent collaborates in the pipeline.

---

## High‑level architecture

![Architecture](static/architecture.png)

1. **Frontend (Tailwind UI)**  
   Users paste a PDF URL and a list of questions. The UI shows an agent timeline
   and displays answers + fact‑check results.

2. **Planner Agent** (`app/services/one_planning_synthesis_agent.py`)  
   - Generates concise *hypothetical answers* that improve recall during retrieval  
   - Builds compact prompts for batch and single‑question answering  
   - Can talk to **Gemini** or **OpenAI** depending on configuration

3. **Retrieval Service** (`app/services/three_retrieval_service.py`)  
   - Downloads the PDF and extracts raw text with **PyMuPDF**  
   - Splits text into overlapping word‑level chunks  
   - Embeds chunks with **SentenceTransformers**  
   - Builds either:
     - a **FAISS** inner‑product index (if `faiss-cpu` is installed), or  
     - a pure **NumPy** cosine‑similarity fallback  
   - Optionally reranks candidates with a cross‑encoder

4. **Synthesizer Agent**  
   - Consumes the top‑ranked context chunks  
   - Asks the LLM to respond in a strict JSON schema  
   - Extracts the `answer` field and returns it to the caller

5. **Fact‑Checker Agent** (`app/services/fact_checker_agent.py`)  
   - Reads the original question, answer, and the retrieved context  
   - Returns a JSON payload with:
     - `verdict`: `supported`, `partially_supported`, or `unsupported`  
     - `risk_score`: 0–1 hallucination risk  
     - `notes`: short explanation

---

## Tech stack

| Layer          | Tech                                             |
|----------------|--------------------------------------------------|
| API backend    | FastAPI, Uvicorn                                |
| Vector search  | SentenceTransformers + FAISS / NumPy cosine sim |
| LLM providers  | Google Gemini (via `google-generativeai`), OpenAI (`openai`) |
| PDF parsing    | PyMuPDF (`fitz`)                                 |
| Reranking      | `CrossEncoder` (MS MARCO MiniLM)                 |
| Frontend       | Static HTML + Tailwind CSS + vanilla JS         |
| Container      | Docker (Python 3.11 slim)                        |

---

## Quickstart (local)

1. **Create a virtualenv**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create `.env`**

```bash
cp env.example .env  # or create manually
```

Fill in at least:

```env
LLM_PROVIDER=gemini           # or: openai
GOOGLE_API_KEY=your_gemini_key_here   # required if LLM_PROVIDER=gemini
OPENAI_API_KEY=your_openai_key_here   # required if LLM_PROVIDER=openai
HACKATHON_BEARER_TOKEN=changeme
```

4. **Run the API**

```bash
uvicorn app.main:app --reload
```

Then open: `http://127.0.0.1:8000` to see the **Agentic Doc RAG Studio** UI.

---

## Docker usage

Build the image:

```bash
docker build -t agentic-doc-rag .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env agentic-doc-rag
```

---

## Project structure

```text
.
├── app
│   ├── api
│   │   └── endpoints
│   │       └── run.py           # /api/v1/hackrx/run endpoint
│   ├── core
│   │   ├── config.py            # LLM + security configuration
│   │   └── security.py
│   ├── schemas
│   │   └── models.py            # Pydantic request/response models
│   └── services
│       ├── one_planning_synthesis_agent.py  # Planner + Synthesizer
│       ├── three_retrieval_service.py       # PDF ingestion + FAISS/NumPy vector store
│       └── fact_checker_agent.py            # Custom hallucination guardrail agent
├── static
│   ├── index.html               # New Tailwind UI
│   └── architecture.png         # Custom architecture diagram
├── requirements.txt
├── Dockerfile
└── run_pipeline.py              # Convenience runner used during development
```

---

## API shape

The main endpoint is:

```http
POST /api/v1/hackrx/run
Content-Type: application/json
Authorization: Bearer <HACKATHON_BEARER_TOKEN>
```

Example request body:

```json
{
  "documents": "https://example.com/policies/sample.pdf",
  "questions": [
    "What are the main exclusions?",
    "How is user data stored?"
  ]
}
```

Depending on configuration and UI, the backend can return either:

```json
{
  "answers": [
    "Short answer for Q1...",
    "Short answer for Q2..."
  ]
}
```

or an enriched format such as:

```json
[
  {
    "question": "What are the main exclusions?",
    "answer": "Short answer...",
    "fact_check": {
      "verdict": "supported",
      "risk_score": 0.12,
      "notes": "All claims directly supported by context chunks 3 and 7."
    }
  }
]
```

The provided UI (`static/index.html`) understands both formats.

---

## What makes this repo different?

- **No hosted vector DB** – everything runs locally with FAISS (or pure NumPy),
  which is perfect for demos, interviews, and offline experimentation.

- **Provider‑agnostic LLM layer** – flip between Gemini and OpenAI via one
  environment variable without touching the core pipeline.

- **Explicit Fact‑Checker agent** – showcases how to layer lightweight
  guardrails and introspection on top of a standard RAG system.

This makes it a great portfolio project to demonstrate practical MLOps,
retrieval, and multi‑agent orchestration skills.
