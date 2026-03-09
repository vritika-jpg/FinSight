# 📈 FinSight — RAG-Powered Financial Intelligence Chatbot

An AI-powered chatbot that analyzes 10-K filings and other financial documents for publicly filed companies using Retrieval-Augmented Generation (RAG) and GPT-4o.

🔗 [Live Streamlit Demo](https://finsightrag.streamlit.app/)

---

## 👥 Team

| Name | Role |
|------|------|
| Vritika Narra | Code Owner — app architecture, UI, RAG pipeline, visual reports, chunk size testing |
| Yuying Ding | UPDATE |
| Megan Huber | UPDATE |
| Jialu Li | UPDATE |
| Fahda Alajmi | DeepSeek vs GPT-4o Model Comparison |
| Zhiruo Zhao | UPDATE |

---

## 🧠 Approach

**Model:** GPT-4o with a custom financial analyst system prompt that enforces source citation, fiscal year attribution, and no outside knowledge.

**Architecture:** LangChain `RetrievalQA` → FAISS vector store → OpenAI Ada-002 embeddings → GPT-4o

**RAG Settings:**

| Parameter | Value |
|-----------|-------|
| Chunk Size | 1000 tokens |
| Chunk Overlap | 100 tokens |
| Top-K (text queries) | 4 |
| Top-K (visual queries) | 12 |
| Temperature | 0.1 (text), 0.0 (visual) |

With the low temperature approach along with giving the RAG a strictly objective persona, we were able to almost eliminate hallucinations. *The tradeoff was that not all questions got answered.*

---

## 🧪 Chunk Size Trial

Tested with three queries: Amazon revenue, cross-company operating income, and Microsoft risk factors.

| Chunk Size | Strength | Weakness |
|------------|----------|----------|
| 500 | Got Alphabet's exact total ($112.4B) | Lost Microsoft's total entirely |
| **1000** | Consistent results across all three companies | Alphabet total missing; got segment deltas instead |
| 1500 | Richest cross-company narrative | Risk factors drifted to IP/AI topics vs. core cybersecurity |

**We decided to go with the 1000 chunk size.** Reasoning: 500 chunk size was inconsistent across companies; and 1500 chunk size was too noisy for straightforward (and factual) queries.

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

## Notes on Running the App Locally

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

Upload the 10-K PDFs via the left panel to activate the chatbot.

---

## 📚 Data Sources

- [Alphabet 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GOOGL&type=10-K)
- [Amazon 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AMZN&type=10-K)
- [Microsoft 2024 10-K](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=MSFT&type=10-K)

*Built for the AI Essentials RAG Project — Johns Hopkins University*
