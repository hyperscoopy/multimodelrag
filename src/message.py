from langchain_core.messages import HumanMessage


def create_multimodal_message(
    query,
    retrieved_docs,
    image_data_store,
    conversation_history=None
):

    if conversation_history is None:

        conversation_history = []

    content = []

    # ==================================================
    # INSTRUCTION
    # ==================================================

    content.append({

        "type": "text",

        "text": f"""
You are a multimodal PDF question-answering assistant.

Answer the following question using the provided
PDF text and PDF page images.

Question:
{query}

Rules:

- Give a clear textual explanation.
- Use both textual and visual information.
- Carefully inspect diagrams, tables, charts and figures.
- Combine information from multiple pages when necessary.
- Do not mention FAISS, embeddings or retrieval.
- Do not output page markers.
- Do not output image labels.
- Do not invent information.
- If the answer is not present in the provided context,
  say that it could not be found.
"""
    })

    # ==================================================
    # CONVERSATION HISTORY
    # ==================================================

    if conversation_history:

        history_text = (
            "\nPrevious conversation:\n"
        )

        for item in conversation_history[-5:]:

            history_text += (

                f"\nUser: "
                f"{item['question']}\n"

                f"Assistant: "
                f"{item['answer']}\n"

            )

        content.append({

            "type": "text",

            "text": history_text

        })

    # ==================================================
    # PDF TEXT + IMAGES
    # ==================================================

    for doc in retrieved_docs:

        page = doc.metadata.get(
            "page",
            "?"
        )

        image_id = doc.metadata.get(
            "image_id"
        )

        # ------------------------------
        # TEXT
        # ------------------------------

        content.append({

            "type": "text",

            "text": (
                f"\nRelevant PDF content:\n"
                f"{doc.page_content}"
            )

        })

        # ------------------------------
        # IMAGE
        # ------------------------------

        if (
            image_id
            and image_id in image_data_store
        ):

            content.append({

                "type": "image_url",

                "image_url": {

                    "url": (
                        "data:image/png;base64,"
                        f"{image_data_store[image_id]}"
                    )

                }

            })

    # ==================================================
    # FINAL INSTRUCTION
    # ==================================================

    content.append({

        "type": "text",

        "text": """
Answer the user's question now.

Return only a natural-language explanation.
Do not mention page numbers or internal processing.
"""
    })

    return HumanMessage(
        content=content
    )