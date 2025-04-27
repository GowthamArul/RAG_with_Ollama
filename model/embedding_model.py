from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embeddings_func():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings