#  Multimodal PDF RAG

An AI-powered **Multimodal Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions about their content.

Unlike traditional text-only RAG systems, this project combines:

- 📄 PDF text
- 🖼️ Page images
- 📊 Diagrams and visual information
- 🔎 Semantic retrieval
- 🧠 CLIP-based multimodal embeddings
- 🗂️ FAISS vector search
- 🤖 NVIDIA multimodal LLM

The system retrieves the most relevant PDF pages and uses both textual and visual context to generate answers.

---

# 🚀 Project Overview

Traditional RAG systems mainly retrieve text chunks from documents.

This project extends the concept by treating each PDF page as a **multimodal unit** containing both text and visual information.

The overall workflow is:

```text
                    PDF Upload
                        │
                        ▼
                PDF Page Processing
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        Text Extraction       Page Rendering
             │                     │
             ▼                     ▼
       Text Embeddings        Image Embeddings
             │                     │
             └──────────┬──────────┘
                        ▼
                 FAISS Vector Store
                        │
                        ▼
                  User Question
                        │
                        ▼
                Semantic Retrieval
                        │
                        ▼
             Relevant PDF Pages
                        │
                  ┌─────┴─────┐
                  │           │
                  ▼           ▼
                 Text       Images
                  │           │
                  └─────┬─────┘
                        ▼
             Multimodal NVIDIA LLM
                        │
                        ▼
                  Final Answer
```

---

## 📂 Project Structure

```text
Multimodal_ground1.0/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── pipeline.py
│   ├── llm.py
│   └── main.py
│
├── cache/
└── README.md
```
---


## 🛠️ Tech Stack

| Component | Technology | Used For |
|---|---|---|
| **UI** | Streamlit | Building the interactive web interface for PDF upload and chat |
| **Language** | Python | Core programming language used to build the entire application |
| **PDF Processing** | PyMuPDF | Reading PDFs, extracting text, and processing pages |
| **Image Processing** | PIL | Loading, converting, and handling PDF page images |
| **Embeddings** | CLIP | Converting text and images into semantic vector representations |
| **Vector Store** | FAISS | Storing embeddings and performing similarity-based retrieval |
| **Deep Learning** | PyTorch | Running the CLIP model and performing tensor computations |
| **Model Framework** | Hugging Face Transformers | Loading and using pretrained CLIP models |
| **LLM** | NVIDIA Multimodal LLM | Understanding retrieved text/images and generating answers |
| **Caching** | Local Cache | Reusing processed PDF data and avoiding unnecessary reprocessing |
---
## 🆚 Improvements from Multimodal_ground0

- **Professional Streamlit Interface** — Modern and user-friendly UI.
- **Conversation History** — Maintains previous questions and answers.
- **Conversation Starters** — Provides suggested questions for users.
- **Processing Status & Statistics** — Displays processing progress and document information.
- **PDF & FAISS Caching** — Reuses processed data to reduce processing time.
- **Visual Context Handling** — Uses relevant page images for multimodal answers.
- **Improved Multimodal Retrieval** — Better retrieval of text and visual content.
- **Selective Image Submission** — Sends relevant images to reduce LLM token usage.
---
## ⚠️ Current Limitations

- Diagram retrieval can sometimes select a similar but incorrect page.
- Follow-up questions need better conversation-aware retrieval.
- Large PDFs require significant processing time.
- Sending too many page images can exceed multimodal model limits.
- CLIP may struggle with very small diagram labels or dense tables.

---
## ▶️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd Multimodal_ground1.0
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project root and add:

```env
NVIDIA_API_KEY=your_api_key
HF_TOKEN=your_token
```

---

## ▶️ Run the Application

### Streamlit Web Interface

```bash
streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

### Terminal Version

```bash
python -m src.main
```

---
## 💡 Example Questions

### 1. Document Overview

```text
What is this document about?
```

### 2. Summarization

```text
Summarize the main concepts.
```

### 3. Diagram Analysis

```text
Explain this diagram.
```

### 4. Figure Analysis

```text
What does the figure represent?
```

### 5. Key Points

```text
What are the key points?
```

### 6. Relationship Analysis

```text
Explain the relationship between these components.
```

---
## 📌 Project Status

- Functional Prototype
- The core multimodal pipeline is working with:
- PDF → CLIP → FAISS → Visual Context → NVIDIA LLM → Answer

The next major focus is improving diagram retrieval and conversation-aware questioning.

---
## 🔮 Future Improvements
- Conversation-aware query rewriting
- OCR for diagrams and images
- Hybrid text + image retrieval
- Better diagram detection
- Page-level citations
- Improved table understanding
- Retrieval similarity/debugging panel
- Performance optimization
- Production deployment
 ---
## 🔭 Final Vision

The ultimate objective is to evolve this prototype into a document intelligence assistant that can reliably understand the complete structure of a PDF rather than treating it as text alone.

                MULTIMODAL DOCUMENT
                        │
        ┌───────────────┼────────────────┐
        │               │                │
       TEXT           IMAGES           DIAGRAMS
        │               │                │
        └───────────────┼────────────────┘
                        │
                 MULTIMODAL RAG
                        │
                 ┌──────┴──────┐
                 │             │
             RETRIEVAL       VISION
                 │             │
                 └──────┬──────┘
                        │
                     LLM
                        │
                        ▼
                GROUNDED ANSWER

**Multimodal_ground1 establishes a foundation for reliable multimodal document understanding and conversational PDF analysis.**
