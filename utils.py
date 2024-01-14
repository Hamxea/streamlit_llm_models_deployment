import time

from huggingface_hub import InferenceClient

from implementing_rag.rag_chromadb_engine import RagChromaDbEngine

# Initialize debounce variables
last_call_time = 0
debounce_interval = 2  # Set the debounce interval (in seconds) to your desired value
rag_chromadb_engine = RagChromaDbEngine()


def rag_chromadb_engine(query, n_results=1):
    context = rag_chromadb_engine.generate_context(query=query, n_results=n_results)
    return context


# @timer()
# @torch.inference_model()
def debounce_huggingface_run(llm, prompt, max_len, temperature, top_p, API_TOKEN_HEADERS):
    global last_call_time
    print("last call time: ", last_call_time)

    # Get the current time
    current_time = time.time()

    # Calculate the time elapsed since the last call
    elapsed_time = current_time - last_call_time

    # Check if the elapsed time is less than the debounce interval
    if elapsed_time < debounce_interval:
        print("Debouncing")
        return "Hello! You are sending requests too fast. Please wait a few seconds before sending another request."

    # Update the last call time to the current time
    last_call_time = time.time()

    headers = {"Authorization": f"Bearer " + API_TOKEN_HEADERS,
               "Content-Type": "application/json", }

    context = rag_chromadb_engine.generate_context(query=prompt, n_results=1)

    system_prompt = """\
    You are a helpful AI assistant that can answer questions on activity for cerg. Answer based on the context provided. If you cannot find the correct answer, say I don't know. Be concise and just include the response.
    """

    user_prompt = f"""
                    Based on the context:
                    {context}
                    Answer the below query:
                    {prompt}
                """

    response = rag_chromadb_engine.chat_completion(url=llm, system_prompt=system_prompt, user_prompt=user_prompt, length=max_len)

    return response




