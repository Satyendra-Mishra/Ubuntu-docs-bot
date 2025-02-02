import os
import sys
import logging
import markdown
import argparse
import faiss
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import VectorDBQA
from src.vector_store import *


logger = logging.getLogger(__name__)

# Function to load and read all Markdown files recursively
def get_md_files(directory_path):
    """
    Recursively retrieves all Markdown (.md) files from a given directory.

    Parameters
    ----------
    directory_path : str
        The path to the directory to search for Markdown files.

    Returns
    -------
    list of Path
        A list of Path objects representing the Markdown files found.
    """
    md_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.md'):
                md_files.append(Path(root) / file)
    return md_files

# Function to load and split Markdown content
def process_markdown_file(file_path):
    """
    Reads a Markdown file and converts its content to HTML.

    Parameters
    ----------
    file_path : str or Path
        The path to the Markdown file.

    Returns
    -------
    str
        The HTML-converted content of the Markdown file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    html_content = markdown.markdown(md_content)  # Convert to HTML (or keep as plain text)
    return html_content


def chunk_text(text, chunk_size=512, overlap=32):
    """
    Splits a given text into smaller overlapping chunks.

    Parameters
    ----------
    text : str
        The input text to be split into chunks.
    chunk_size : int, optional
        The size of each text chunk (default is 512).
    overlap : int, optional
        The number of overlapping characters between consecutive chunks (default is 32).

    Returns
    -------
    list of str
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Get all files with a specific extension in a directory.")
    parser.add_argument("-i",'--data_dir', type=str, help="Path to the root directory.")
    parser.add_argument("-e", '--file_extension', type=str, help="File extension to search for (e.g., '.md').")

    # Parse arguments
    args = parser.parse_args()

    # check if directory exists
    if not os.path.isdir(args.data_dir):
        logger.error(f"Error: {args.data_dir} is not a valid directory.")
        sys.exit(1)
    # Ensure that the file extension starts with a dot
    if not args.file_extension.startswith('.'):
        logger.error("Error: File extension must start with a dot (e.g., '.md').")
        sys.exit(1)
    

    dense_embedding_func = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-large-en-v1.5",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )

    embd_dim = len(dense_embedding_func.embed_query("hello world"))
    M = 50
    index = faiss.IndexHNSWFlat(embd_dim, M)
    index.hnsw.efConstruction = 32
    index.hnsw.efSearch = 32

    faiss_db = initialize_faiss_db(
        embedding_func=dense_embedding_func,
        index=index  
    )

    files = get_md_files(args.data_dir)
    all_chunks = []
    all_metadata = []
    for file in files:
        source = str(file).split("/")[-1]
        file_content  = process_markdown_file(file)
        chunks = chunk_text(file_content)
        all_chunks.extend(chunks)
        all_metadata.extend([{"source": f"{source}"}]*len(chunks))
    
    chunked_docs = convert_to_documents(all_chunks, all_metadata)

    upload_documnents(vector_store=faiss_db, documents=chunked_docs)

    save_vector_db(vector_store=faiss_db, path="models/vector_store")

    logging.info("Documents ingested into the vector store")