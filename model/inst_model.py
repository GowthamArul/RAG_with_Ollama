from langchain_community.llms.ollama import Ollama

async def get_model_response(results, prompt):
    """
    Purpose: Get the response from the model as a streaming response
    """
    model = Ollama(model="mistral")

    # Collect unique metadata
    metadata_list = []
    for doc, _score in results:
        print("Doc Metadata:- \n",doc.metadata)
        metadata = {
            "filename": (str(doc.metadata.get("source", "unknown.pdf")).split("/")[-1]),
            "title": doc.metadata.get("title", "Untitled"),
            "pagenumber": doc.metadata.get("page", "unknown")
        }
        metadata_list.append(metadata)

    # Use a set to ensure unique metadata
    unique_metadata = {tuple(m.items()): m for m in metadata_list}.values()

    accumulated_response = ""

    async for chunk in model._astream(prompt):
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        accumulated_response += chunk_text
        
        # Yield the accumulated response without metadata
        formatted_response = f"Response: {accumulated_response}\n"
        yield formatted_response

    # After the streaming is complete, yield the unique metadata
    formatted_metadata = "\n".join(
        f"Filename: {m['filename']}, Title: {m['title']}, Page Number: {m['pagenumber']}" 
        for m in unique_metadata
    )
    yield f"Metadata:\n{formatted_metadata}"
