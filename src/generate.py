from groq import Groq

def intialize_groq_client(api_key):
    client = Groq(
        api_key=api_key,  # This is the default and can be omitted
    )
    return client


def chat_completion(messages, groq_client, model, **gen_params):
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model=model,
        **gen_params
    )
    return chat_completion.choices[0].message.content


def format_retrieved_docs(docs):
    string = ""
    page_content = [doc[0].page_content for doc in docs]
    source = [doc[0].metadata["source"] for doc in docs]

    for i, text in enumerate(page_content):
        string += f"source: {source[i]}+\n"
        string += f"content: {text}"
        string += "\n\n"
    
    return string