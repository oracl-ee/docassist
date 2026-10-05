def chunk_text(text, size=800, overlap=150):
text = " ".join(text.split()) # collapse whitespace and newlines
chunks = []
start = 0
while start < len(text):
chunks.append(text[start:start + size])
start += size - overlap # move forward 650 characters each time
return chunks
def chunk_documents(docs, size=800, overlap=150):
out = []
for doc in docs:
for i, piece in enumerate(chunk_text(doc["text"], size, overlap)):
out.append({"source": doc["source"], "chunk_id": i, "text": piece})
return out