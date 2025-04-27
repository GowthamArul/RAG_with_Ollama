import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse
from services.DataModel import UserRequest
from services.api import query_rag
import os

current_path = os.getcwd()
app = FastAPI(
    title='RAG Application with Ollama',
    description="",
    version= "1.0.0"
)

@app.get("/")
async def root():
    """
    Welcome message on the application startup
    """
    return {"Welcome to the RAG with Ollama"}

@app.get("/get_status")
async def get_status():
    """
    Status of the Application
    """
    return JSONResponse({"status": "Status OK"})


@app.post("/query",
          tags=["Query"],
          summary="Query the text document")
async def query_api(request: UserRequest):
    """
    Purpose: Query the text document using the retriever and generator
    """
    response = await query_rag(request)
    
    return response

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)