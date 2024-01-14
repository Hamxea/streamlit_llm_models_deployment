import time

from huggingface_hub import InferenceClient

from implementing_rag.rag_chromadb_engine import RagChromaDbEngine

# Initialize debounce variables
last_call_time = 0
debounce_interval = 2  # Set the debounce interval (in seconds) to your desired value


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

    # Streaming Client # token = 'Bearer ' + API_TOKEN_HEADERS
    huggingface_client = InferenceClient(model=llm, token=False, headers=headers)

    rag_chromadb_engine = RagChromaDbEngine()

    context = rag_chromadb_engine.generate_context(query=prompt, n_results=3)

    user_prompt = f"<s>[INST]<<SYS>>{context}<</SYS>>{prompt}[/INST]"

    response = rag_chromadb_engine.chat_completion(huggingface_client=huggingface_client, user_prompt=user_prompt,
                                                   top_p=top_p, temperature=temperature, max_len=max_len)

    return response
