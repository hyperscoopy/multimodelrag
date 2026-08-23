from .retrieval import retrieve_multimodal

from .message import (
    create_multimodal_message
)


def multimodal_pdf_rag_pipeline(
    query,
    vector_store,
    image_data_store,
    llm,
    conversation_history=None,
    k=5
):

    # ==================================================
    # RETRIEVE RELEVANT PAGES
    # ==================================================

    context_docs = retrieve_multimodal(
        query,
        vector_store,
        k=k
    )

    print(
        f"\nRetrieved "
        f"{len(context_docs)} relevant pages."
    )

    for doc in context_docs:

        print(
            f"  Page "
            f"{doc.metadata.get('page', '?')}"
        )

    # ==================================================
    # LIMIT IMAGES SENT TO LLM
    # ==================================================

    # Keep text from all retrieved pages,
    # but send only the first 2 page images.
    multimodal_docs = context_docs[:2]

    print(
        f"\nSending "
        f"{len(multimodal_docs)} page images "
        f"to the multimodal LLM."
    )

    # ==================================================
    # CREATE MULTIMODAL MESSAGE
    # ==================================================

    message = create_multimodal_message(

        query,

        multimodal_docs,

        image_data_store,

        conversation_history

    )

    # ==================================================
    # CALL NVIDIA MULTIMODAL LLM
    # ==================================================

    response = llm.invoke(
        [message]
    )

    # ==================================================
    # RETURN TEXT ANSWER
    # ==================================================

    return response.content