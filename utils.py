import time

from huggingface_hub import InferenceClient

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

    # Streaming Client
    client = InferenceClient(llm, token='Bearer ' + API_TOKEN_HEADERS, headers=headers)

    # generation parameter
    gen_kwargs = dict(
        max_new_tokens=max_len,  # 512,
        top_k=30,
        top_p=top_p,  # 0.9,
        temperature=temperature,  # 0.2,
        repetition_penalty=1.02,
        stop_sequences=["User:", "\n User:", "\nUser:", "</s>"],
    )

    stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)

    # yield each generated token
    for r in stream:
        # skip special tokens
        if r.token.special:
            continue
        # stop if we encounter a stop sequence
        if r.token.text in gen_kwargs["stop_sequences"]:
            break
        # yield the generated token
        # return r.token.text
        yield r.token.text
