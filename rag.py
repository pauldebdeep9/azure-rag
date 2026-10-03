import logging
import os
from functools import lru_cache

from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import OpenAI


INDEX_NAME = "rag-sai-satcharitra"
COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

# The semantic ranker re-orders at most the top 50 matches, so when it is
# switched on we ask the vector search for that many candidates.
SEMANTIC_CANDIDATES = 50

logger = logging.getLogger(__name__)


@lru_cache
def get_credential() -> DefaultAzureCredential:
    # In Azure this uses the Container App's managed identity.
    # On a machine with `az login`, it uses your own account instead.
    return DefaultAzureCredential()


@lru_cache
def get_search_client() -> SearchClient:
    # SearchClient refreshes Entra tokens itself.
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=get_credential(),
    )


@lru_cache
def get_token_provider():
    # Returns a function that gives a valid bearer token,
    # refreshing it before it expires.
    return get_bearer_token_provider(get_credential(), COGNITIVE_SCOPE)


@lru_cache
def get_base_openai_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=get_token_provider()(),
    )


def get_openai_client() -> OpenAI:
    # Tokens expire after about an hour, so attach a current one per request.
    # with_options reuses the base client's connection pool.
    return get_base_openai_client().with_options(api_key=get_token_provider()())


def get_semantic_config() -> str:
    # Name of the index's semantic configuration. Set this environment
    # variable to turn semantic ranking on; leave it unset to turn it off.
    return os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "").strip()


def run_search(question: str, top_k: int, semantic_config: str) -> list[dict]:
    vector_query = VectorizableTextQuery(
        text=question,
        fields="text_vector",
        k_nearest_neighbors=SEMANTIC_CANDIDATES if semantic_config else top_k,
    )

    search_args = {
        "search_text": question,              # keyword search
        "vector_queries": [vector_query],     # vector search
        "select": ["chunk_id", "chunk", "title"],
        "top": top_k,
    }

    if semantic_config:
        # Re-rank the combined keyword and vector matches.
        search_args["query_type"] = "semantic"
        search_args["semantic_configuration_name"] = semantic_config

    results = get_search_client().search(**search_args)

    # The request is sent when the results are read, so read them here.
    return [dict(item) for item in results]


def retrieve(question: str, top_k: int = 5) -> tuple[list[dict], str]:
    semantic_config = get_semantic_config()

    if semantic_config:
        try:
            return run_search(question, top_k, semantic_config), "semantic"
        except HttpResponseError as exc:
            # For example, the free monthly allowance has been used up.
            logger.warning(
                "Semantic ranking failed, using hybrid search instead: %s", exc
            )

    return run_search(question, top_k, ""), "hybrid"


def answer(question: str) -> dict:
    docs, retrieval = retrieve(question)

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
        "retrieval": retrieval,
        "sources": [
            {
                "title": doc.get("title", ""),
                "reranker_score": doc.get("@search.reranker_score"),
                "chunk": doc.get("chunk", "")[:500],
            }
            for doc in docs
        ],
    }