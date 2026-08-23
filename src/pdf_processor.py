import os
import io
import base64
import pickle
import hashlib

import fitz
from PIL import Image

from langchain_core.documents import Document

from .embeddings import embed_image


def get_cache_path(pdf_path):

    absolute_path = os.path.abspath(pdf_path)

    file_hash = hashlib.md5(
        absolute_path.encode()
    ).hexdigest()

    os.makedirs(
        "cache",
        exist_ok=True
    )

    return os.path.join(
        "cache",
        f"{file_hash}.pkl"
    )


def process_pdf(pdf_path):

    # ==================================================
    # CHECK CACHE
    # ==================================================

    cache_path = get_cache_path(
        pdf_path
    )

    if os.path.exists(cache_path):

        print(
            "\nLoading processed PDF from cache..."
        )

        with open(
            cache_path,
            "rb"
        ) as f:

            return pickle.load(f)

    # ==================================================
    # CHECK PDF
    # ==================================================

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # ==================================================
    # OPEN PDF
    # ==================================================

    pdf = fitz.open(
        pdf_path
    )

    total_pages = len(pdf)

    print(
        "\nPDF loaded successfully."
    )

    print(
        f"Total pages: {total_pages}"
    )

    # ==================================================
    # STORAGE
    # ==================================================

    all_docs = []

    all_embeddings = []

    image_data_store = {}

    # ==================================================
    # PAGE-BY-PAGE PROCESSING
    # ==================================================

    for page_number in range(
        total_pages
    ):

        page = pdf[
            page_number
        ]

        display_page = (
            page_number + 1
        )

        print(
            f"\nProcessing page "
            f"{display_page}/{total_pages}..."
        )

        # ==================================================
        # 1. EXTRACT TEXT
        # ==================================================

        text = page.get_text(
            "text"
        ).strip()

        if not text:

            text = (
                "No extractable text found "
                "on this page. Use the page image "
                "for visual information."
            )

        # ==================================================
        # 2. RENDER PAGE AS IMAGE
        # ==================================================

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(
                1.5,
                1.5
            ),
            alpha=False
        )

        image_bytes = pixmap.tobytes(
            "png"
        )

        # Convert bytes → PIL Image
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert(
            "RGB"
        )

        # ==================================================
        # 3. STORE IMAGE
        # ==================================================

        image_id = (
            f"page_{display_page}"
        )

        image_base64 = base64.b64encode(
            image_bytes
        ).decode(
            "utf-8"
        )

        image_data_store[
            image_id
        ] = image_base64

        # ==================================================
        # 4. CREATE DOCUMENT
        # ==================================================

        document = Document(

            page_content=text,

            metadata={

                "page": display_page,

                "type": "page",

                "image_id": image_id,

                "source": pdf_path

            }

        )

        all_docs.append(
            document
        )

        # ==================================================
        # 5. CREATE IMAGE EMBEDDING
        # ==================================================

        print(
            f"Generating image embedding "
            f"for page {display_page}..."
        )

        embedding = embed_image(
            image
        )

        all_embeddings.append(
            embedding
        )

        print(
            f"Page {display_page} completed."
        )

    pdf.close()

    # ==================================================
    # SAVE CACHE
    # ==================================================

    processed_data = (

        all_docs,

        all_embeddings,

        image_data_store

    )

    with open(
        cache_path,
        "wb"
    ) as f:

        pickle.dump(
            processed_data,
            f
        )

    print(
        "\nPDF processing completed."
    )

    print(
        "Processed data saved to cache."
    )

    return processed_data