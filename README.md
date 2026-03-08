# 📈 FinSight — RAG-Powered Financial Intelligence Chatbot

> An AI-powered chatbot that analyzes 10-K filings for **Alphabet (Google)**, **Amazon**, and **Microsoft** using Retrieval-Augmented Generation (RAG) and GPT-4o. Ask questions in plain English — or request interactive visual reports.

---

## 🚀 Live Demo

🔗 [Launch FinSight on Streamlit Cloud](https://finsightrag.streamlit.app/)

---

## 👥 Team

| Name | Role |
|------|------|
| Vritika Narra | Code Owner — app architecture, UI, RAG pipeline, visual reports |
| *(Teammate 2)* | Question Designer — test question bank, pass/fail evaluation |
| *(Teammate 3)* | Hallucination Hunter — boundary testing, failure case documentation |
| *(Teammate 4)* | Chunk Size Experimenter — chunk size comparison (500 / 1000 / 1500) |
| Fahda | DeepSeek Model Comparison — GPT-4o vs DeepSeek evaluation |
| *(Teammate 6)* | README, Tech Note, Presentation Slides |

---

## 🧠 Architecture

```
User Query
    │
    ▼
Intent Detection ──► Visual Request? ──► Structured JSON Prompt
    │                                           │
    │ No                                        ▼
    ▼                                    Plotly Chart Render
FAISS Vector Store
    │
    ▼
Top-K Chunk Retrieval (k=4 text, k=12 visual)
    │
    ▼
GPT-4o with System Prompt
    │
    ▼
Streamed Response / Visual Report
```

**Stack:**
- **Frontend:** Streamlit
- **LLM:** GPT-4o (OpenAI)
- **Embeddings:** OpenAIEmbeddings (`text-embedding-ada-002`)
- **Vector Store:** FAISS (in-memory)
- **Document Loader:** PyPDFLoader (LangChain)
- **Orchestration:** LangChain Classic `RetrievalQA`
- **Visualization:** Plotly

---

## ⚙️ RAG Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Chunk Size | 1000 tokens | Balanced context vs. noise |
| Chunk Overlap | 100 tokens | Prevents context loss at boundaries |
| Retrieval Top-K (text) | 4 | Sufficient for factual Q&A |
| Retrieval Top-K (visual) | 12 | Higher to capture all 3 companies |
| Embedding Model | text-embedding-ada-002 | OpenAI hosted |
| LLM Temperature | 0.1 (text), 0.0 (visual) | Low for factual accuracy |

---

## ✨ Key Features

- 💬 **Conversational Q&A** — ask anything about the 10-K filings in plain English
- 📊 **Visual Reports** — say *"bar chart of..."* or *"line graph of..."* to get interactive Plotly charts generated directly from the documents
- 🏷️ **Source Attribution** — every answer cites the company and document section
- 💰 **Live Analytics** — tracks query count, token usage, cost, and avg response time
- 💾 **Export Chat** — download the full conversation as a `.txt` file
- 🔄 **Session Reset** — clear all documents, chat history, and analytics

---

## 📊 Visual Report Examples

Add any of these keywords to your question to trigger a chart:
`chart` · `graph` · `plot` · `visualize` · `diagram`

| Example Query | Chart Type |
|---------------|-----------|
| *"Bar chart of total revenue for all three companies in 2024"* | Bar |
| *"Line graph of Amazon's revenue from 2022 to 2024"* | Line |
| *"Pie chart of Microsoft's revenue segments"* | Pie |
| *"Bar chart of cloud revenue across Google, Amazon, and Microsoft"* | Bar |

---

## 🧪 Model & Settings Comparison

### GPT-4o vs DeepSeek
*(Results from Fahda's evaluation — see presentation for full details)*

| Metric | GPT-4o | DeepSeek |
|--------|--------|----------|
| Factual Accuracy | ✅ High | *(TBD)* |
| Response Speed | *(TBD)* | *(TBD)* |
| Hallucination Rate | *(TBD)* | *(TBD)* |
| Cost | Higher | Lower |

### Chunk Size Comparison
*(Results from chunk size experiments — see presentation for full details)*

| Chunk Size | Overlap | Observation |
|------------|---------|-------------|
| 500 | 50 | *(TBD)* |
| 1000 | 100 | ✅ Current setting — balanced |
| 1500 | 150 | *(TBD)* |

---

## 🔍 Hallucination Analysis

FinSight is designed to resist hallucination through:
- Explicit system prompt instruction: *"Never guess or use outside knowledge to fill gaps"*
- Source-grounded retrieval — answers only from uploaded documents
- Low temperature (0.1) to reduce creativity in factual responses

**Known boundary cases:**
- Azure standalone revenue (Microsoft does not disclose this separately — a common hallucination trigger)
- Questions referencing data outside the uploaded filing years
- Market share figures (never reported in 10-Ks)

*(See presentation for documented hallucination examples and analysis)*

---

## 🛠️ Running Locally

### Prerequisites
- Python 3.10+
- An OpenAI API key

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/finsight.git
cd finsight
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-openai-key-here"
```

Run the app:
```bash
streamlit run finsight.py
```

Then upload the 10-K PDFs for Alphabet, Amazon, and Microsoft via the sidebar.

---

## 📁 Repository Structure

```
finsight/
├── finsight.py          # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .streamlit/
    └── secrets.toml     # API keys (not committed — add to .gitignore)
```

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key packages:

```
streamlit · langchain · langchain-classic · langchain-openai
faiss-cpu · tiktoken · pypdf · plotly · openai
```

---

## ⚠️ Important Notes

- **API Key:** Never commit your OpenAI API key to a public repository. Use Streamlit secrets or environment variables.
- **10-K Files:** The PDFs are not included in this repository. Upload them directly via the app interface.
- **Session State:** All data (chat history, vector store, charts) is stored in-session and resets on page refresh.

---

## 📚 Data Sources

- [Alphabet 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GOOGL&type=10-K)
- [Amazon 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AMZN&type=10-K)
- [Microsoft 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=MSFT&type=10-K)

---

*Built for the AI Essentials RAG Project — Johns Hopkins University*
