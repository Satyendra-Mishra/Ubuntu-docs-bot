import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from src.vector_store import retrieve_documents, load_faiss_local
from src.generate import intialize_groq_client, chat_completion, format_retrieved_docs


load_dotenv()

def query_rewriting_prompt(current_query, chat_history):
    sys_prompt = '''\
    You are an AI assistant that reformulates user question into standalone question when necessary, to improve the effectiveness of information retrieval from a vector database.
    If chat history is provided, then consider the context of the chat history for reformulating the question.

    - If the user provides a greeting (e.g. "hi", "hello", "how are you?" etc.) then ignore the chat history and do not reformulate the user input.
    - If the user's question is independent of chat history, return it as is.
    - If the user's question depends on chat history, rephrase it into a standalone question.

    Do not add more details than necessary to the standalone question. Only return the standalone alone question and no other explanation or text.
    '''
    chat_history_str = ""
    if len(chat_history) > 0:
        for item in chat_history:
            chat_history_str += f'{item['role']}:\n'
            chat_history_str += f'{item["content"]}\n'

    messages = [
        {
            "role": "system",
            "content": f"{sys_prompt}"
        },
        {
            "role": "user",
            "content": f"##Chat History:\n{chat_history_str}\n\n##User Question:\n{current_query}"
        },
        {
            "role": "assistant",
            "content": "##Standalone Question:\n"
        }
    ]

    return messages


def chat_completion_prompt(query, retrieved_docs):

    system_prompt = '''\
    You are a helpful assistant trained to answer questions based on a provided set of documentation. When responding to any user query, you should:

    1. **Use only the provided documentation**: All answers must be based on the context from the documentation that has been provided. If the documentation does not contain sufficient information to answer the question, kindly say "Sorry, I don't have enough information to answer that."

    2. **Be concise and clear**: Provide the answer in a concise manner. If necessary, summarize the relevant information from the documentation.

    3. **Provide reference sources to the documentation**: If applicable, refer to specific sections or paragraphs of the documentation when answering. For example: "As mentioned in Section 3.2 of the documentation...".

    4. **Stay on-topic**: Do not deviate from the context of the provided documentation. If the question is out of scope for the documentation, explain that the information is unavailable.

    5. **Context-aware answers**: Maintain awareness of the conversation history. Use the previous questions and answers to understand the context better and give relevant, coherent responses. If the question is related to previous topics, try to connect your answers to those topics.

    ### User Instructions:
    - You must respond using the information in the provided documentation.
    - If a question is not clear or the documentation doesn't contain enough details, respond politely with a clarification request or let the user know that the information is unavailable.
    \
    '''
    documentation = format_retrieved_docs(retrieved_docs)
    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}"
        },
        {
            "role": "user",
            "content": f"Documentation Context:\n{documentation}\n\nQuery:\n{query}"
        }
    ]

    return messages 



if __name__ == "__main__":

    client = intialize_groq_client(api_key=os.getenv("GROQ_API_KEY"))
    embedding_func = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-large-en-v1.5",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
    faiss_db = load_faiss_local(path="models/vector_store", embedding_func=embedding_func)

    model="llama-3.3-70b-versatile"
    # model = "mixtral-8x7b-32768"

    print("Chatbot: Hello! I'm your assistant. How can I help you today ?")
    
    conversation_history = [] 
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == 'stop':
            print("Chatbot: Goodbye! Take care!")
            break
        elif user_input.lower() == "start":
            print("--"*20)
            print("Starting a new conversation")
            conversation_history = []
            user_input = input("User: ").strip()
        
        rewritten_query = chat_completion(
            messages=query_rewriting_prompt(user_input, conversation_history),
            groq_client=client,
            model=model,
            )
        
        conversation_history.append({
            "role": "user",
            "content": f"{rewritten_query}"
        })

        retrieved_docs = retrieve_documents(rewritten_query, topK=25, vector_store=faiss_db)
        
        gen_params = {"max_tokens": 1024, "temperature": 1.0}
        response = chat_completion(
            messages=chat_completion_prompt(rewritten_query, retrieved_docs),
            groq_client=client,
            model=model,
            **gen_params
        )
        
        conversation_history.append({
            "role": "assistant",
            "content": f"{response}"
        })
        
        print(f"Chatbot: {response}")
