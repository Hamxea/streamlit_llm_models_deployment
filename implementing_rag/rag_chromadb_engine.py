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

    def chat_completion(self, url, system_prompt, user_prompt, length=1000):
        final_prompt = f"""<s>[INST]<<SYS>>
            {system_prompt}
        <</SYS>>
            {user_prompt} 
        [/INST]"""
        huggingface_client = InferenceClient(model=url)
        return huggingface_client.text_generation(prompt=final_prompt, max_new_tokens=length).strip()

    def save_db(self):
        return

