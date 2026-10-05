import ollama 
reply = ollama.chat(model="llama3.1:8b",
                    messages=[{"role": "user", "content": "what is a GPU? one sentence"}])
print (reply ["message"] ["content"])
emb = ollama.embed(model="nomic-embed-text", input= "hello world")  
print(len(emb ["embeddings"][0]))