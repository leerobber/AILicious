import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k_similar(query_embedding: list[float], candidates: list[dict], k: int) -> list[dict]:
    """`candidates`: dicts each carrying an 'embedding' key plus whatever else the caller
    wants preserved (id/role/content). Returns the k most similar to `query_embedding`,
    most similar first, with 'embedding' stripped from each result -- callers need to know
    which messages matched, not the vectors themselves.
    """
    scored = [(cosine_similarity(query_embedding, c["embedding"]), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{key: value for key, value in c.items() if key != "embedding"} for _, c in scored[:k]]
