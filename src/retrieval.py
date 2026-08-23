from .embeddings import embed_text


def retrieve_multimodal(
    query,
    vector_store,
    k=5
):

    # ==================================================
    # CONVERT QUESTION TO CLIP TEXT EMBEDDING
    # ==================================================

    query_embedding = embed_text(
        query
    )

    # ==================================================
    # SEARCH IMAGE EMBEDDINGS
    # ==================================================

    results = (
        vector_store
        .similarity_search_by_vector(
            query_embedding,
            k=k
        )
    )

    return results