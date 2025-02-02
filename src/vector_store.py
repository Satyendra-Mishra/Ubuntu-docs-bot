from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def initialize_faiss_db(embedding_func, index):
    """
    Initializes a FAISS vector store with an embedding function and index.

    Parameters
    ----------
    embedding_func : Callable
        The embedding function used to convert text into vector representations.
    index : faiss.Index
        The FAISS index structure used for storing and retrieving embeddings.

    Returns
    -------
    FAISS
        An instance of the FAISS vector store.
    """
    vector_store = FAISS(
        embedding_function=embedding_func,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store


def convert_to_documents(chunks: list[str], meta_data: list[dict]):
    """
    Converts text chunks and metadata into Document objects.

    Parameters
    ----------
    chunks : list of str
        A list of text chunks to be converted into documents.
    meta_data : list of dict
        A list of metadata dictionaries corresponding to each text chunk.

    Returns
    -------
    list of Document
        A list of Document objects containing text chunks and their metadata.
    """
    if isinstance(chunks, str):
        chunks = [chunks]
        meta_data = [meta_data]
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata=meta_data[i]
            )
        )
    return documents


def upload_documnents(vector_store, documents: list[Document]):
    """
    Uploads documents to the FAISS vector store.

    Parameters
    ----------
    vector_store : FAISS
        The FAISS vector store instance where documents will be added.
    documents : list of Document
        A list of Document objects to be stored.

    Returns
    -------
    None
    """
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)


def delete_documents(vector_store, uuids=[]):
    """
    Deletes documents from the FAISS vector store using their UUIDs.

    Parameters
    ----------
    vector_store : FAISS
        The FAISS vector store instance.
    uuids : list of str, optional
        A list of document UUIDs to be deleted (default is an empty list).

    Returns
    -------
    None
    """
    vector_store.delete(ids=uuids)


def retrieve_documents(query, topK, vector_store, filter=None):
    """
    Retrieves the top-K most similar documents from the FAISS vector store.

    Parameters
    ----------
    query : str
        The search query for retrieving similar documents.
    topK : int
        The number of top relevant documents to return.
    vector_store : FAISS
        The FAISS vector store instance.
    filter : dict, optional
        A filtering condition for retrieval (default is None).

    Returns
    -------
    list of tuple
        A list of tuples containing retrieved documents and their similarity scores.
    """
    results = vector_store.similarity_search_with_score(
        query,
        k=topK,
        filter=filter,
    )
    return results    

def save_vector_db(vector_store, path):
    """
    Saves the FAISS vector store to a local directory.

    Parameters
    ----------
    vector_store : FAISS
        The FAISS vector store instance to be saved.
    path : str
        The directory path where the vector store will be stored.

    Returns
    -------
    None
    """
    vector_store.save_local(path)


def load_faiss_local(path, embedding_func):
    """
    Loads a FAISS vector store from a local directory.

    Parameters
    ----------
    path : str
        The directory path where the FAISS vector store is stored.
    embedding_func : Callable
        The embedding function used to convert text into vector representations.

    Returns
    -------
    FAISS
        An instance of the loaded FAISS vector store.
    """
    return FAISS.load_local(
    path, embedding_func, allow_dangerous_deserialization=True
    )
    