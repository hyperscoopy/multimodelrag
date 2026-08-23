import streamlit as st
import tempfile
import os

from src.pdf_processor import process_pdf
from src.vector_store import (
    create_vector_store,
    load_vector_store
)
from src.llm import initialize_llm
from src.pipeline import multimodal_pdf_rag_pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal PDF RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL APP BACKGROUND
       ========================================================= */

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(129, 140, 248, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 95% 10%,
                rgba(56, 189, 248, 0.14),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #eef2ff 0%,
                #f8f9ff 50%,
                #ecfeff 100%
            ) !important;

        color: #172554 !important;
    }


    /* =========================================================
       STREAMLIT HEADER
       ========================================================= */

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }


    /* =========================================================
       MAIN CONTENT WIDTH
       ========================================================= */

    [data-testid="stMainBlockContainer"] {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }


    /* =========================================================
       SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.97) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.12);
    }

    section[data-testid="stSidebar"] * {
        color: #172554 !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #172554 !important;
        font-weight: 750 !important;
    }


    /* =========================================================
       MAIN TITLE
       ========================================================= */

    h1 {
        color: #172554 !important;
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        line-height: 1.15 !important;
        margin-bottom: 0.35rem !important;
    }


    /* =========================================================
       SUBTITLE
       ========================================================= */

    .app-subtitle {
        color: #64748b !important;
        font-size: 1.08rem !important;
        line-height: 1.6 !important;
        margin-bottom: 2rem !important;
    }


    /* =========================================================
       WELCOME CARD
       ========================================================= */

    .info-card {
        background: rgba(255, 255, 255, 0.94) !important;
        color: #172554 !important;

        padding: 2rem !important;

        border-radius: 20px !important;

        border: 1px solid rgba(99, 102, 241, 0.10) !important;

        box-shadow:
            0 12px 40px rgba(30, 41, 59, 0.07) !important;

        margin: 1.2rem 0 !important;

        overflow: hidden !important;
    }

    .info-card h3 {
        color: #172554 !important;
        font-size: 1.7rem !important;
        font-weight: 750 !important;
        margin: 0 0 1rem 0 !important;
    }

    .info-card p {
        color: #475569 !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
        margin: 0.7rem 0 !important;
    }


    /* =========================================================
       FEATURE CARDS
       ========================================================= */

    .status-card {
        background: rgba(255, 255, 255, 0.95) !important;

        min-height: 115px !important;

        padding: 1.25rem !important;

        border-radius: 18px !important;

        border: 1px solid rgba(99, 102, 241, 0.10) !important;

        box-shadow:
            0 8px 28px rgba(30, 41, 59, 0.06) !important;

        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;

        overflow: hidden !important;
    }

    .status-title {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
        letter-spacing: 0.08em !important;

        margin-bottom: 0.5rem !important;
    }

    .status-value {
        color: #172554 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;

        white-space: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.35 !important;
    }


    /* =========================================================
       FILE UPLOADER
       ========================================================= */

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.96) !important;

        border: 1px solid rgba(99, 102, 241, 0.14) !important;

        border-radius: 16px !important;

        padding: 0.5rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #172554 !important;

        border: 1px solid #c7d2fe !important;
        border-radius: 10px !important;

        font-weight: 650 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #64748b !important;
    }


    /* =========================================================
       NORMAL BUTTONS
       ========================================================= */

    .stButton > button {
        background: #ffffff !important;

        color: #172554 !important;

        border: 1px solid rgba(99, 102, 241, 0.20) !important;

        border-radius: 10px !important;

        font-weight: 650 !important;

        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        border-color: #6366f1 !important;

        color: #4338ca !important;

        transform: translateY(-1px);
    }


    /* =========================================================
       PRIMARY BUTTON
       ========================================================= */

    .stButton > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #6366f1,
            #4f46e5
        ) !important;

        color: white !important;

        border: none !important;
    }


    /* =========================================================
       CHAT
       ========================================================= */

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.92) !important;

        border: 1px solid rgba(99, 102, 241, 0.10) !important;

        border-radius: 16px !important;

        margin-bottom: 12px !important;

        padding: 1rem !important;
    }

    [data-testid="stChatMessage"] p {
        color: #172554 !important;

        line-height: 1.7 !important;
    }


    /* =========================================================
       CHAT INPUT
       ========================================================= */

    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.96) !important;

        border: 1px solid rgba(99, 102, 241, 0.15) !important;

        border-radius: 14px !important;
    }


    /* =========================================================
       METRICS
       ========================================================= */

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.92) !important;

        border: 1px solid rgba(99, 102, 241, 0.10) !important;

        border-radius: 16px !important;

        padding: 1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 750 !important;
    }


    /* =========================================================
       DIVIDERS
       ========================================================= */

    hr {
        border-color: rgba(99, 102, 241, 0.12) !important;
    }


    /* =========================================================
       CAPTIONS
       ========================================================= */

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }


    /* =========================================================
       REMOVE UNWANTED CODE BACKGROUNDS
       ========================================================= */

    .stMarkdown pre,
    .stCode,
    [data-testid="stCode"] {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "vector_store": None,

    "image_data_store": None,

    "llm": None,

    "conversation_history": [],

    "pdf_name": None,

    "pdf_path": None,

    "processed": False,

    "page_count": 0,

    "embedding_count": 0,

    "image_count": 0
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📚 Multimodal RAG"
    )

    st.caption(
        "AI-powered PDF question answering"
    )

    st.divider()

    st.markdown(
        "### 📄 Document"
    )

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### ⚙️ System"
    )

    if st.session_state.processed:

        st.success(
            "● PDF Ready"
        )

        st.caption(
            f"Document: {st.session_state.pdf_name}"
        )

        st.caption(
            f"Pages: {st.session_state.page_count}"
        )

    else:

        st.info(
            "○ Waiting for PDF"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.conversation_history = []

        st.rerun()

    st.divider()

    st.caption(
        "Multimodal PDF RAG"
    )

    st.caption(
        "Text + Vision + Retrieval"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📚 Multimodal PDF RAG")

st.markdown(
    """
    <div class="app-subtitle">
        Ask questions and understand your PDF using
        text, images, diagrams and visual context.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NO PDF STATE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-card">

        <h3>👋 Welcome</h3>

        <p>
        Upload a PDF from the sidebar to start chatting
        with your document.
        </p>

        <p>
        The system can retrieve relevant pages and use
        both textual and visual information to generate
        answers.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
      [1, 1, 1],
      gap="large"
    )

    with col1:

        st.markdown(
            """
            <div class="status-card">

            <div class="status-title">
            RETRIEVAL
            </div>

            <div class="status-value">
            🔎 Semantic Search
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="status-card">

            <div class="status-title">
            VISION
            </div>

            <div class="status-value">
            👁️ Image Understanding
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="status-card">

            <div class="status-title">
            GENERATION
            </div>

            <div class="status-value">
            🤖 AI Answers
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# PDF INFORMATION
# ============================================================

st.markdown(
    f"""
    <div class="info-card">

    <strong>📄 {uploaded_file.name}</strong>

    <br>

    <span style="color:#6b7280;">
    Ready to process
    </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROCESS BUTTON
# ============================================================

if not st.session_state.processed:

    st.markdown(
        "### 🚀 Process Document"
    )

    st.write(
        "Process the PDF page-by-page and build the "
        "multimodal retrieval index."
    )

    if st.button(
        "⚡ Process PDF",
        type="primary",
        use_container_width=True
    ):

        # -----------------------------------------------
        # SAVE TEMP PDF
        # -----------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name

        st.session_state.pdf_path = pdf_path

        # -----------------------------------------------
        # PROCESS
        # -----------------------------------------------

        with st.status(
            "Processing PDF...",
            expanded=True
        ):

            try:

                st.write(
                    "📄 Reading PDF pages..."
                )

                (
                    all_docs,
                    all_embeddings,
                    image_data_store
                ) = process_pdf(
                    pdf_path
                )

                st.session_state.page_count = (
                    len(all_docs)
                )

                st.session_state.embedding_count = (
                    len(all_embeddings)
                )

                st.session_state.image_count = (
                    len(image_data_store)
                )

                st.write(
                    f"✅ Processed {len(all_docs)} pages"
                )

                # -----------------------------------------
                # FAISS
                # -----------------------------------------

                st.write(
                    "🔎 Preparing vector search..."
                )

                vector_store = load_vector_store(
                    pdf_path
                )

                if vector_store is None:

                    vector_store = create_vector_store(
                        all_docs,
                        all_embeddings,
                        pdf_path
                    )

                st.session_state.vector_store = (
                    vector_store
                )

                st.write(
                    "✅ FAISS retrieval ready"
                )

                # -----------------------------------------
                # LLM
                # -----------------------------------------

                st.write(
                    "🤖 Initializing multimodal LLM..."
                )

                if st.session_state.llm is None:

                    st.session_state.llm = (
                        initialize_llm()
                    )

                st.write(
                    "✅ AI model ready"
                )

                # -----------------------------------------
                # STORE IMAGES
                # -----------------------------------------

                st.session_state.image_data_store = (
                    image_data_store
                )

                st.session_state.pdf_name = (
                    uploaded_file.name
                )

                # -----------------------------------------
                # MARK PDF AS PROCESSED
                # -----------------------------------------

                st.session_state.processed = True

                st.session_state.conversation_history = []

                st.success(
                    "🎉 PDF is ready for questions!"
                )

            except Exception as e:

                st.error(
                    f"Processing failed: {e}"
                )

        # ---------------------------------------------
        # RERUN AFTER SUCCESSFUL PROCESSING
        # ---------------------------------------------

        if st.session_state.processed:

            st.rerun()

    st.stop()

# ============================================================
# DOCUMENT READY
# ============================================================

st.markdown(
    f"""
    <div class="info-card">

    <strong>🟢 Document Ready</strong>

    <br>

    <span style="color:#6b7280;">
    {st.session_state.pdf_name}
    </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "📄 Pages",
        st.session_state.page_count
    )

with col2:

    st.metric(
        "🔎 Embeddings",
        st.session_state.embedding_count
    )

with col3:

    st.metric(
        "🖼️ Page Images",
        st.session_state.image_count
    )


st.divider()


# ============================================================
# CHAT HEADER
# ============================================================

st.markdown(
    "### 💬 Chat with your PDF"
)

st.caption(
    "Ask about concepts, diagrams, tables, definitions "
    "or anything contained in the document."
)


# ============================================================
# CONVERSATION STARTERS
# ============================================================

if not st.session_state.conversation_history:

    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(99,102,241,0.12);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin: 1rem 0 1.5rem 0;
            box-shadow: 0 6px 22px rgba(30,41,59,0.05);
        ">
            <div style="
                color:#172554;
                font-size:1.05rem;
                font-weight:700;
                margin-bottom:0.35rem;
            ">
                ✨ Conversation Starters
            </div>

            <div style="
                color:#64748b;
                font-size:0.9rem;
            ">
                Try one of these questions to explore your document.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    starter_col1, starter_col2 = st.columns(2, gap="large")

    with starter_col1:

        starter_1 = st.button(
            "📖 Summarize this document",
            use_container_width=True
        )

        starter_2 = st.button(
            "🧠 Explain the main concepts",
            use_container_width=True
        )

    with starter_col2:

        starter_3 = st.button(
            "📊 Explain the important diagrams",
            use_container_width=True
        )

        starter_4 = st.button(
            "🔍 What are the key points?",
            use_container_width=True
        )


# ============================================================
# SELECT STARTER QUESTION
# ============================================================

starter_question = None

if not st.session_state.conversation_history:

    if starter_1:
        starter_question = "Summarize this document."

    elif starter_2:
        starter_question = (
            "Explain the main concepts covered in this document."
        )

    elif starter_3:
        starter_question = (
            "Explain the important diagrams in this document "
            "and describe what they represent."
        )

    elif starter_4:
        starter_question = (
            "What are the key points I should understand "
            "from this document?"
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.conversation_history:

    with st.chat_message(
        "user"
    ):

        st.write(
            message["question"]
        )

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            message["answer"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your PDF..."
)


# ============================================================
# USE STARTER OR USER QUESTION
# ============================================================

if question:

    final_question = question

elif starter_question:

    final_question = starter_question

else:

    final_question = None


# ============================================================
# GENERATE ANSWER
# ============================================================

if final_question:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user"
    ):

        st.write(
            final_question
        )

    # ========================================================
    # AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching the document and analyzing relevant content..."
        ):

            try:

                answer = multimodal_pdf_rag_pipeline(
                    final_question,
                    st.session_state.vector_store,
                    st.session_state.image_data_store,
                    st.session_state.llm,
                    k=5
                )

                # Display answer

                st.markdown(
                    answer
                )

                # Save conversation

                st.session_state.conversation_history.append({

                    "question": final_question,

                    "answer": answer

                })

            except Exception as e:

                st.error(
                    f"Unable to generate answer: {e}"
                )