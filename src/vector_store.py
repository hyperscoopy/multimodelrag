import os
import hashlib
import numpy as np

from langchain_community.vectorstores import FAISS


VECTORSTORE_ROOT = "vectorstore"


def get_vectorstore_path(pdf_path):

    # Get absolute PDF path
    absolute_path = os.path.abspath(
        pdf_path
    )

    # Create unique hash for this PDF path
    file_hash = hashlib.md5(
        absolute_path.encode()
    ).hexdigest()

    # Each PDF gets its own folder
    return os.path.join(
        VECTORSTORE_ROOT,
        file_hash
    )


def create_vector_store(
    all_docs,
    all_embeddings,
    pdf_path
):

    # Get PDF-specific path
    vectorstore_path = get_vectorstore_path(
        pdf_path
    )

    # Create directory
    os.makedirs(
        vectorstore_path,
        exist_ok=True
    )

    # Convert embeddings to NumPy array
    embeddings_array = np.array(
        all_embeddings
    )

    # Create FAISS vector store
    vector_store = FAISS.from_embeddings(

        text_embeddings=[
            (
                doc.page_content,
                embedding
            )

            for doc, embedding in zip(
                all_docs,
                embeddings_array
            )
        ],

        embedding=None,

        metadatas=[
            doc.metadata
            for doc in all_docs
        ]
    )

    # Save FAISS
    vector_store.save_local(
        vectorstore_path
    )

    print(
        f"FAISS saved for this PDF."
    )

    return vector_store


def load_vector_store(
    pdf_path
):

    # Get PDF-specific path
    vectorstore_path = get_vectorstore_path(
        pdf_path
    )

    # Check if FAISS exists
    if not os.path.exists(
        vectorstore_path
    ):

        return None

    print(
        "Loading FAISS vector store..."
    )

    vector_store = FAISS.load_local(

        vectorstore_path,

        embeddings=None,

        allow_dangerous_deserialization=True
    )

    print(
        "FAISS vector store loaded."
    )

    return vector_store