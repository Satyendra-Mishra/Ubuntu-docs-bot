import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from chatbot import chat_completion_prompt, query_rewriting_prompt
from langchain_huggingface import HuggingFaceEmbeddings
from src.vector_store import load_faiss_local, retrieve_documents
from src.generate import intialize_groq_client, chat_completion
from src.chat_store import init_db, get_conversation_history, store_message
from uuid import uuid4


# Pydantic model for request data
class QueryRequest(BaseModel):
    query: str
    session_id: str


app = FastAPI()

client = intialize_groq_client(api_key=os.getenv("GROQ_API_KEY"))
embedding_func = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)

faiss_db = load_faiss_local(path="models/vector_store", embedding_func=embedding_func)
chat_store_name = "models/chat_store/conversation_history.db"

model="llama-3.3-70b-versatile"


# Route for health check
@app.get("/health")
async def read_root():
    return {"message": "Chatbot API Running"}

@app.post("/chat")
async def query_bot(query_request: QueryRequest):
    
    session_id = query_request.session_id if query_request.session_id else str(uuid4())
    user_query = query_request.query
    
    try:
        conversation_history = get_conversation_history(session_id, chat_store_name)
        
        # Rewriting user query to fit chat format
        rewritten_query = chat_completion(
            messages=query_rewriting_prompt(user_query, conversation_history),
            groq_client=client,
            model=model,
        )
        
        # store the user query
        store_message(session_id, "user", rewritten_query, chat_store_name)

        # Retrieve relevant documents similar to the query
        retrieved_docs = retrieve_documents(rewritten_query, topK=25, vector_store=faiss_db)

        # Generate the chatbot response
        gen_params = {"max_tokens": 1024, "temperature": 1.0}
        response = chat_completion(
            messages=chat_completion_prompt(rewritten_query, retrieved_docs),
            groq_client=client,
            model=model,
            **gen_params
        )

        # store the bot response
        store_message(session_id, "assistant", response, chat_store_name)

        # Return the bot's reponse
        return {"response": response, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")


if __name__ == "__main__":
    init_db(chat_store_name)
    os.system("uvicorn app:app --reload --host 0.0.0.0 --port 8080")