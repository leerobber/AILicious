from core import embeddings as embeddings_module


def test_cosine_similarity_identical_vectors_is_one():
    assert embeddings_module.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert embeddings_module.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert embeddings_module.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_zero_vector_is_zero_not_a_crash():
    assert embeddings_module.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_top_k_similar_orders_most_similar_first():
    query = [1.0, 0.0]
    candidates = [
        {"id": 1, "embedding": [0.0, 1.0]},  # orthogonal -- least similar
        {"id": 2, "embedding": [1.0, 0.0]},  # identical -- most similar
        {"id": 3, "embedding": [0.7, 0.7]},  # partial match -- middle
    ]

    result = embeddings_module.top_k_similar(query, candidates, k=3)

    assert [c["id"] for c in result] == [2, 3, 1]


def test_top_k_similar_respects_k():
    query = [1.0, 0.0]
    candidates = [{"id": i, "embedding": [1.0, 0.0]} for i in range(10)]

    result = embeddings_module.top_k_similar(query, candidates, k=3)

    assert len(result) == 3


def test_top_k_similar_strips_embedding_from_results():
    query = [1.0, 0.0]
    candidates = [{"id": 1, "role": "user", "content": "hi", "embedding": [1.0, 0.0]}]

    result = embeddings_module.top_k_similar(query, candidates, k=1)

    assert result == [{"id": 1, "role": "user", "content": "hi"}]


def test_top_k_similar_empty_candidates_returns_empty_list():
    assert embeddings_module.top_k_similar([1.0, 0.0], [], k=4) == []
