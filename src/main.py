from .pdf_processor import process_pdf

from .vector_store import (
    create_vector_store,
    load_vector_store
)

from .llm import initialize_llm

from .pipeline import multimodal_pdf_rag_pipeline


def main():

    print("=" * 70)
    print("              MULTIMODAL PDF RAG")
    print("=" * 70)

    # ------------------------------------------
    # 1. GET PDF PATH
    # ------------------------------------------

    pdf_path = input(
        "\nEnter PDF path: "
    ).strip()

    if not pdf_path:

        print(
            "No PDF path provided."
        )

        return

    # ------------------------------------------
    # 2. PROCESS PDF
    # ------------------------------------------

    print(
        "\nProcessing PDF..."
    )

    try:

        all_docs, all_embeddings, image_data_store = process_pdf(
            pdf_path
        )

    except Exception as e:

        print(
            "\nPDF processing failed:"
        )

        print(e)

        return

    print(
        f"\nProcessed pages: "
        f"{len(all_docs)}"
    )

    print(
        f"Generated embeddings: "
        f"{len(all_embeddings)}"
    )

    print(
        f"Stored page images: "
        f"{len(image_data_store)}"
    )

    # ------------------------------------------
    # 3. LOAD OR CREATE FAISS
    # ------------------------------------------

    print(
        "\nChecking FAISS vector store..."
    )

    try:

        vector_store = load_vector_store(
            pdf_path
        )

        if vector_store is not None:

            print(
                "Using existing FAISS vector store."
            )

        else:

            print(
                "No saved FAISS vector store found."
            )

            print(
                "Creating FAISS vector store..."
            )

            vector_store = create_vector_store(
                all_docs,
                all_embeddings,
                pdf_path
            )

            print(
                "FAISS vector store created."
            )

    except Exception as e:

        print(
            "\nFAISS operation failed:"
        )

        print(e)

        return

    # ------------------------------------------
    # 4. INITIALIZE LLM
    # ------------------------------------------

    print(
        "\nInitializing NVIDIA LLM..."
    )

    try:

        llm = initialize_llm()

    except Exception as e:

        print(
            "\nLLM initialization failed:"
        )

        print(e)

        return

    print(
        "LLM initialized."
    )

    # ------------------------------------------
    # 5. QUESTION LOOP
    # ------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PDF READY"
    )

    print(
        "=" * 70
    )

    print(
        "\nAsk questions about your PDF."
    )

    print(
        "Type 'exit' to quit."
    )

    # ------------------------------------------
    # CONVERSATION HISTORY
    # ------------------------------------------

    conversation_history = []

    while True:

        query = input(
            "\nAsk a question: "
        ).strip()

        # --------------------------------------
        # EXIT
        # --------------------------------------

        if query.lower() == "exit":

            print(
                "\nExiting..."
            )

            break

        # --------------------------------------
        # EMPTY QUESTION
        # --------------------------------------

        if not query:

            print(
                "Please enter a question."
            )

            continue

        # --------------------------------------
        # GENERATE ANSWER
        # --------------------------------------

        print(
            "\nGenerating answer..."
        )

        try:

            answer = multimodal_pdf_rag_pipeline(

                query,

                vector_store,

                image_data_store,

                llm,

                conversation_history

            )

            print(
                "\nAnswer:"
            )

            print(
                "-" * 70
            )

            print(
                answer
            )

            print(
                "-" * 70
            )

            # ----------------------------------
            # SAVE CONVERSATION
            # ----------------------------------

            conversation_history.append({

                "question": query,

                "answer": answer

            })

        except Exception as e:

            print(
                "\nError generating answer:"
            )

            print(e)


if __name__ == "__main__":

    main()