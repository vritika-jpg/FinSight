# 📈 FinSight — RAG-Powered Financial Intelligence Chatbot

An AI-powered chatbot that analyzes 10-K filings and other financial documents for publicly filed companies using Retrieval-Augmented Generation (RAG) and GPT-4o.

🔗 [Live Streamlit Demo](https://finsightrag.streamlit.app/)

---

## 👥 Team

| Name | Role |
|------|------|
| Vritika Narra | Code Owner — architecture, UI, RAG pipeline |
| Yuying Ding | Hallucination Testing |
| Megan Huber | Demo + Troubleshooting |
| Jialu Li | Question Formulation |
| Fahda Alajmi | DeepSeek vs GPT-4o vs Gemini Model Comparison |
| Zhiruo Zhao | Documentation |

---

## 🧠 Approach

**Model:** GPT-4o with a custom financial analyst system prompt that enforces source citation, fiscal year attribution, and no outside knowledge.

**Architecture:** Custom RAG pipeline with table-aware ingestion → FAISS vector store → OpenAI text-embedding-3-small embeddings → GPT-4o

**RAG Settings:**

| Parameter | Value |
|-----------|-------|
| PDF Loader | pdfplumber (text + table-aware) |
| Chunk Size | 1500 tokens (text only; tables are never split) |
| Chunk Overlap | 200 tokens |
| Top-K (single-company retrieval) | 15 |
| Top-K (multi-company retrieval) | 5 per company (up to 15 total) |
| Top-K (table retrieval for numeric queries) | 10 additional table chunks |
| Temperature (text queries) | 0.1 |
| Temperature (visual queries) | 0.0 |

**Note:** Two GPT-4o instances are used — a deterministic model (temperature 0.0) for visual JSON generation and a low-temperature model (0.1) for natural language answers. With the low temperature approach along with giving the RAG a strictly objective persona, we were able to almost eliminate hallucinations.

---

### Ingestion

We originally loaded PDFs using PyPDFLoader from LangChain, which is a text-only parser that cannot capture embedded images, charts, or figures. We upgraded to **pdfplumber**, which separates text and tables during extraction:

- **Text** is chunked at 1,500 tokens with 200-token overlap using `RecursiveCharacterTextSplitter`
- **Tables** are extracted into markdown format and stored as their own individual documents — they are **never split**, since a chunk boundary cutting through a financial table would make the numbers meaningless
- Every chunk is tagged with two metadata fields at ingestion time: **company** (from filename) and **content_type** (text or table)

---

### Retrieval Strategy

FinSight uses a two-layer query classifier:

**Layer 1 — Company scope:**
- **Single-company queries** retrieve the top **15 most relevant chunks** from the FAISS vector store
- **Multi-company queries** (e.g., "compare", "vs", "which company", "all three") trigger **balanced retrieval** — 5 chunks per company filtered by metadata, ensuring each firm's filing contributes equally to the context

**Layer 2 — Content type:**
- For financially numeric queries (containing keywords like "revenue", "margin", "operating income", etc.), an additional retrieval pass pulls the top **10 table chunks** from the index and appends them to the context
- This ensures the model is reading from actual structured financial tables when answering questions about specific figures, not just surrounding prose

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| PDF Parsing | pdfplumber |
| RAG Framework | LangChain (text splitting only) |
| Vector Database | FAISS |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o |
| Visualization | Plotly |
| Local Model Experiments | Ollama |

---

## Model and Embedding Comparison

To determine the most reliable architecture for financial document analysis, we evaluated both large language models and embedding approaches used within the RAG pipeline.

<details>
<summary><strong>⚖️ LLM Evaluation (Qualitative)</strong></summary>

We evaluated several language models for financial document question answering. The comparison focused on reliability when interpreting long-form regulatory filings such as 10-K reports.

| Model | Strengths | Weaknesses | Observations |
|------|------|------|------|
| **GPT-4o** | Highly reliable answers, strong reasoning over financial text, consistent citation alignment | Slightly slower than lightweight models | Produced the most grounded answers when paired with retrieved document chunks. Best overall performance for financial QA. |
| **DeepSeek** | Fast response time, good general reasoning | Occasionally introduced unsupported financial figures | Performed well for simple queries but struggled with strict citation grounding in some tests. |
| **Gemini** | Balanced speed and reasoning ability | Less consistent with financial terminology | Capable responses but sometimes missed key context from retrieved passages. |

**Final Model Choice:** GPT-4o

GPT-4o demonstrated the strongest ability to remain grounded in retrieved passages and maintain accurate financial references, making it the most suitable model for the FinSight RAG system.

</details>

<details>
<summary><strong>🔎 Embedding Model Evaluation</strong></summary>

Several embedding approaches were considered for indexing financial documents in the vector database.

| Embedding Model | Strengths | Weaknesses | Observations |
|------|------|------|------|
| **OpenAI text-embedding-3-small** | Strong semantic similarity performance, stable across long financial passages, widely supported in LangChain | Slightly higher API cost than local embeddings | Produced the most consistent retrieval results when querying financial filings. |
| **SentenceTransformers (MiniLM)** | Fast and lightweight, can run locally | Lower retrieval precision on financial language | Good for experimentation but occasionally retrieved loosely related passages. |
| **Instructor Embeddings** | Designed for task-specific embedding generation | More complex setup | Showed promise but did not significantly outperform text-embedding-3-small in our experiments. |

**Final Embedding Choice:** OpenAI text-embedding-3-small

text-embedding-3-small provided the most consistent retrieval quality when searching across multiple companies' 10-K filings, making it the best fit for the FAISS vector store used in this project.

</details>

---

## 🧩 System Architecture

FinSight uses a table-aware Retrieval-Augmented Generation pipeline:

```
10-K PDFs
  → pdfplumber (text + table extraction)
  → Text: RecursiveCharacterTextSplitter (1500 tokens, 200 overlap)
     Tables: stored whole as markdown documents (never split)
  → OpenAI text-embedding-3-small
  → FAISS Vector Store (tagged with company + content_type metadata)
  → Two-layer retrieval:
      Layer 1: single-company (k=15) or multi-company (k=5×n, metadata filtered)
      Layer 2: +10 table chunks for numeric/financial queries
  → Custom Prompt Construction (system prompt + context + chat history)
  → GPT-4o (temperature 0.0 for visuals / 0.1 for text)
  → Answer + Source Citation + Plotly Visualization
```

---

## 🧪 Chunk Size Trial

Tested with three queries: Amazon revenue, cross-company operating income, and Microsoft risk factors.

| Chunk Size | Strength | Weakness |
|------------|----------|----------|
| 500 | Got Alphabet's exact total ($112.4B) | Lost Microsoft's total entirely |
| 1000 | Consistent results across all three companies | Alphabet total missing; got segment deltas instead |
| **1500** | Richest cross-company narrative, full context | Risk factors drifted to IP/AI topics vs. core cybersecurity |

**We decided to go with the 1500 chunk size.** Reasoning: 500 chunk size was inconsistent across companies; 1000 chunk size was still not giving the model the entire context. Anything above 1500 was too noisy.

---

## 📊 Visual Reports

Add `chart`, `graph`, `plot`, `visualize`, or `diagram` to any question to generate an interactive Plotly chart directly from the documents.

| Example Query | Chart Type |
|---------------|------------|
| *"Bar chart of total revenue for all three companies in 2024"* | Bar |
| *"Line graph of Amazon's revenue from 2022 to 2024"* | Line |
| *"Pie chart of Microsoft's revenue segments"* | Pie |
| *"Bar chart of cloud revenue across Google, Amazon, and Microsoft"* | Bar |

The LLM extracts the relevant figures, selects the appropriate chart type, and generates a plain-English explanation in a single query.

---

## Conversational Context Abilities

FinSight supports short follow-up questions by including the previous Q&A pair in the prompt. The system keeps the last 3 exchanges to maintain conversational continuity while limiting prompt size.

---

## How to Run the App Locally

Complete the following steps:

```bash
git clone https://github.com/YOUR_USERNAME/finsight.git
cd finsight
pip install -r requirements.txt
```

Add your API key to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-key-here"
```

```bash
streamlit run finsight.py
```

Upload the 10-K PDFs on the left panel to activate the chatbot.

---

## 📚 Data Sources

- [Alphabet 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GOOGL&type=10-K)
- [Amazon 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AMZN&type=10-K)
- [Microsoft 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=MSFT&type=10-K)

*Built for the AI Essentials RAG Project — Johns Hopkins University*
