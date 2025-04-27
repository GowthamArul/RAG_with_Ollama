from langchain.vectorstores.chroma import Chroma
from model.embedding_model import get_embeddings_func
from model.inst_model import get_model_response
from scripts.prompt_rendering import prompt_rendering
from fastapi.responses import StreamingResponse
from .DataModel import UserRequest


async def query_rag(request: UserRequest):
    # Prepare the DB.
    db = Chroma(persist_directory='database', embedding_function=get_embeddings_func())

    # Search the DB.
    results = db.similarity_search_with_score(request.query, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_rendering(context_text, request.query)

    # Get the streaming response from the model
    response_generator = get_model_response(results, prompt)

    # Return a StreamingResponse
    return StreamingResponse(response_generator, media_type="text/plain")
