import os
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from flask import Flask, render_template, request, jsonify

from medical_keywords import general_medical_words

app = Flask(__name__)

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

csv_file = "Half_information_cleaned.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

df['combined_text'] = df.apply(lambda row: " ".join(row.astype(str)), axis=1)

docs = [
    Document(page_content=text, metadata={"row_index": idx})
    for idx, text in enumerate(df['combined_text'])
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Initialize the sentence transformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "BAAI/bge-base-en"
model = SentenceTransformer(model_id).to(device)

# Extract embeddings for the documents
document_texts = [doc.page_content for doc in documents]  # Get text content
embeddings = model.encode(document_texts, normalize_embeddings=True, convert_to_numpy=True)

# Create a FAISS index to store embeddings
dimension = embeddings.shape[1]  # The size of the embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # Add embeddings to the index

# Function to process the query using RAG and LLM
def process_query_with_rag_and_llm(query):
    if not any(keyword.lower() in query.lower() for keyword in general_medical_words):
        return "Sorry, I am here just to assist you about the doctors and services at Shifa International Hospital."

    # Step 1: Query the Vector Store (RAG)
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    k = 2  # Retrieve top 2 nearest neighbors
    dimension = embeddings.shape[1]  # Embedding size (the dimensionality of the embeddings)
    index = faiss.IndexFlatL2(dimension)  # This is a correct initialization for a flat L2 index

    # Step 2: Add embeddings to the index
    index.add(embeddings)  # Add the embeddings to the index

    # Step 3: Search the index using a query embedding
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)  # Your query encoding
    k = 2
    distances, indices = index.search(query_embedding, k)  # Perform the search


    relevant_docs = [document_texts[idx] for idx in indices[0]]

    # Step 2: Send the result of RAG to the LLM for further processing
    extracted_text = " ".join(relevant_docs) 
    groq_client = Groq(api_key="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")


    prompt = f"""
    Based on the following information about doctors at Shifa International Hospital, provide a clear, concise response in a way that a layperson can easily understand.
    The question may pertain to a doctor or a specialty (e.g., Orthopedic doctor).
    Do not return JSON or structured data; provide the response in normal text, as if you are explaining to a patient or general public.
    Format the response in bullet points for better readability.
    Information: {extracted_text}
    Response: Provide the answer in simple, understandable language.
    """

    # Get the LLM response
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    
    response_content = completion.choices[0].message.content.strip()
    return response_content


@app.route('/')
def index():
    return render_template('index.html')

import logging

logging.basicConfig(level=logging.DEBUG)
@app.route('/ask', methods=['POST'])
def ask_query():
    try:
        query = request.form['query']
        if not query:
            return render_template('index.html', error="Query parameter is required.")

        response = process_query_with_rag_and_llm(query)

        formatted_response = response.replace('\n', '<br>').replace('*', '<li>')
        formatted_response = f"<ul>{formatted_response}</ul>"

        return render_template('index.html', response=formatted_response)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


# if __name__ == '__main__':
#     app.run(debug=True)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no port is set
    app.run(debug=True, host="0.0.0.0", port=port)