import os
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import OpenAI


INDEX_NAME = "rag-sai-satcharitra"

@lru_cache
def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )


@lru_cache
def get_openai_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    vector_query = VectorizableTextQuery(
        text=question,
        fields="text_vector",
        k_nearest_neighbors=top_k,
    )

    results = get_search_client().search(
        search_text=question,              # keyword search
        vector_queries=[vector_query],    # vector search
        select=["chunk_id", "chunk", "title"],
        top=top_k,
    )

    return [dict(item) for item in results]


def answer(question: str) -> dict:
    docs = retrieve(question)

    context = "\n\n".join(
        f"[Source {i}] {doc.get('title', '')}\n{doc['chunk']}"
        for i, doc in enumerate(docs, start=1)
    )

    prompt = f"""
You answer questions using only the supplied context.

If the context does not contain enough information, say so.
Do not invent facts.

CONTEXT
{context}

QUESTION
{question}
"""

    response = get_openai_client().responses.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        input=prompt,
        max_output_tokens=800,
    )

    return {
        "answer": response.output_text,
        "sources": [
            {
                "title": doc.get("title", ""),
                "chunk": doc.get("chunk", "")[:500],
            }
            for doc in docs
        ],
    }
