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
| Zhiruo Zhao | UPDATE |

---

## 🧠 Approach

**Model:** GPT-4o with a custom financial analyst system prompt that enforces source citation, fiscal year attribution, and no outside knowledge.

**Architecture:** LangChain `RetrievalQA` → FAISS vector store → OpenAI Ada-002 embeddings → GPT-4o

**RAG Settings:**

| Parameter | Value |
|-----------|-------|
| Chunk Size | 1500 tokens |
| Chunk Overlap | 200 tokens |
| Top-K (text queries) | 8 |
| Top-K (visual queries) | 12 |
| Temperature | 0.1 (text), 0.0 (visual) |

With the low temperature approach along with giving the RAG a strictly objective persona, we were able to almost eliminate hallucinations. *The tradeoff was that not all questions got answered.*

---

## ⚙️ Tech Stack

| Component               | Tool           |
| ----------------------- | -------------- |
| UI                      | Streamlit      |
| RAG Framework           | LangChain      |
| Vector Database         | FAISS          |
| Embeddings              | OpenAI Ada-002 |
| LLM                     | GPT-4o         |
| Visualization           | Plotly         |
| Local Model Experiments | Ollama         |

---

## 🧪 Model and Embedding Evaluation

To determine the most reliable architecture for financial document analysis, we evaluated both large language models and embedding approaches used within the Retrieval-Augmented Generation (RAG) pipeline.

<details>
<summary><strong>🤖 LLM Evaluation (Qualitative)</strong></summary>

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
| **OpenAI Ada-002** | Strong semantic similarity performance, stable across long financial passages, widely supported in LangChain | Slightly higher API cost than local embeddings | Produced the most consistent retrieval results when querying financial filings. |
| **SentenceTransformers (MiniLM)** | Fast and lightweight, can run locally | Lower retrieval precision on financial language | Good for experimentation but occasionally retrieved loosely related passages. |
| **Instructor Embeddings** | Designed for task-specific embedding generation | More complex setup | Showed promise but did not significantly outperform Ada-002 in our experiments. |

**Final Embedding Choice:** OpenAI Ada-002  

Ada-002 provided the most consistent retrieval quality when searching across multiple companies' 10-K filings, making it the best fit for the FAISS vector store used in this project.

</details>

---

## 🧩 System Architecture

FinSight uses a Retrieval-Augmented Generation pipeline:

10-K PDFs → Text Chunking → OpenAI Ada Embeddings → FAISS Vector Store → LangChain RetrievalQA → GPT-4o → Answer + Source Citation + Plotly Visualization

---

## 🧪 Chunk Size Trial

Tested with three queries: Amazon revenue, cross-company operating income, and Microsoft risk factors.

| Chunk Size | Strength | Weakness |
|------------|----------|----------|
| 500 | Got Alphabet's exact total ($112.4B) | Lost Microsoft's total entirely |
| 1000 | Consistent results across all three companies | Alphabet total missing; got segment deltas instead |
| **1500** | Richest cross-company narrative, full context | Risk factors drifted to IP/AI topics vs. core cybersecurity |

**We decided to go with the 1500 chunk size.** Reasoning: 500 chunk size was inconsistent across companies; and 1000 chunk size was still not giving the model the entire context. Anything above 1500 was too noisy. 

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
