import chromadb
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


class RagChromaDbEngine:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma")
        self.collection = self.client.get_or_create_collection('news-media-and-publication')

    def text_embedding(self, text, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)
        return model.encode(text)

    def generate_context(self, query, n_results=5):
        vector = self.text_embedding(query).tolist()
        result = self.collection.query(
            query_embeddings=vector,
            n_results=n_results,
            include=['documents']
        )
        res = "\n".join(str(item) for item in result['documents'][0])
        return res

    def chat_completion(self, huggingface_client, user_prompt, top_p, temperature, max_len=512):

        # generation parameter
        gen_kwargs = dict(
            max_new_tokens=max_len,  # 512,
            top_k=30,
            top_p=top_p,  # 0.9,
            temperature=temperature,  # 0.2,
            repetition_penalty=1.02,
            stop_sequences=["User:", "\n User:</s>", "</s>\nUser:", "</s>"],
        )

        stream = huggingface_client.text_generation(prompt=user_prompt, stream=True, details=True, **gen_kwargs)

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

    def save_db(self):
        return
