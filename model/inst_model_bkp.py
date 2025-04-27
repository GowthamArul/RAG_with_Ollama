from langchain_community.llms.ollama import Ollama

async def get_model_response(results, prompt):
    """
    Purpose: Get the response from the model as a streaming response
    """
    model = Ollama(model="mistral")

    # Collect unique sources
    sources = list(set(doc.metadata.get("id", None) for doc, _score in results))

    accumulated_response = ""

    async for chunk in model._astream(prompt):
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        accumulated_response += chunk_text
        
        # Yield the accumulated response without sources
        formatted_response = f"Response: {accumulated_response}\n"
        yield formatted_response

    # After the streaming is complete, yield the unique sources
    formatted_sources = f"Sources: {', '.join(sources)}"
    yield formatted_sources
