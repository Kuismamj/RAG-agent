from sentence_transformers import SentenceTransformer
import faiss
import numpy as np  
from llama_cpp import Llama

# 🔹 Asetukset
CHUNK_SIZE = 400 

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)

with open("oma_tiedosto.txt", "r", encoding="utf-8") as f:
    teksti = f.read()

tekstipalat = [teksti[i:i+CHUNK_SIZE] for i in range(0, len(teksti), CHUNK_SIZE)]

malli = SentenceTransformer("all-MiniLM-L6-v2")
embeddingit = malli.encode(tekstipalat)

# FAISS-indeksi
dim = embeddingit.shape[1]
indeksi = faiss.IndexFlatL2(dim)
indeksi.add(embeddingit)

while True:
    kysymys = input("\n🟡 Kysy jotain dokumentista (tai 'exit' lopettaaksesi):\n> ")
    if kysymys.lower() in ["exit", "quit"]:
        break

    kysymys_vektori = malli.encode([kysymys])
    D, I = indeksi.search(np.array(kysymys_vektori), k=1)

    osuma = tekstipalat[I[0][0]]

    prompt = f"""[INST] Here is some background information:\n\"\"\"\n{osuma}\n\"\"\"\n\nThe user asks: {kysymys}\n\nAnswer clearly based on the given context. [/INST]"""

    vastaus = llm(prompt, max_tokens=300)
    print(f"\n Mistralin vastaus:\n{vastaus['choices'][0]['text'].strip()}")
