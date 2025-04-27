import os
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from model.embedding_model import get_embeddings_func

def load_documets():
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    document = PyPDFDirectoryLoader(pdf_path).load()
    return document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    length_function=len,
                    is_separator_regex=False
                    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(

        persist_directory= "database", embedding_function=get_embeddings_func()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")
    
def data_loader_main():
    """
    Purpose: Main function to load, split, and add documents to Chroma
    """
    documents = load_documets()
    print(f"Number of documents loaded: {len(documents)}")
    
    chunks = split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")

    add_to_chroma(chunks)


if __name__ == "__main__":
    data_loader_main()