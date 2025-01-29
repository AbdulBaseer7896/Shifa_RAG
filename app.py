import os
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from flask import Flask, render_template, request, jsonify

from medical_keywords import general_medical_words

# Initialize the Flask app
app = Flask(__name__)

# Define Document class to include metadata
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Load the CSV file (Assuming it's already cleaned)
csv_file = "Half_information_cleaned.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Combine all text fields into one (if needed)
df['combined_text'] = df.apply(lambda row: " ".join(row.astype(str)), axis=1)

# Convert rows into Document objects
docs = [
    Document(page_content=text, metadata={"row_index": idx})
    for idx, text in enumerate(df['combined_text'])
]

# Split the documents using a text splitter (helps chunk large documents)
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
    # index = some_function  # incorrect
    # Create the FAISS index
    # index = faiss.IndexFlatL2(dimension)
    # Step 1: Initialize the FAISS index
    dimension = embeddings.shape[1]  # Embedding size (the dimensionality of the embeddings)
    index = faiss.IndexFlatL2(dimension)  # This is a correct initialization for a flat L2 index

    # Step 2: Add embeddings to the index
    index.add(embeddings)  # Add the embeddings to the index

    # Step 3: Search the index using a query embedding
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)  # Your query encoding
    k = 2  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)  # Perform the search



    # distances, indices = index.search(query_embedding, k)

    # Get the relevant documents based on the search
    relevant_docs = [document_texts[idx] for idx in indices[0]]

    # Step 2: Send the result of RAG to the LLM for further processing
    extracted_text = " ".join(relevant_docs)  # Combine the retrieved documents into one context
    # groq_api_key = os.getenv('GROQ_API_KEY')  # Get the API key securely from environment variables

    # groq_client = Groq(api_key=groq_api_key)
    groq_client = Groq(api_key="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")


    # Create a message for the LLM (Groq)
    prompt = f"""
    Based on the following information about doctors at Shifa International Hospital, provide a clear, concise response in a way that a layperson can easily understand.
    The question may pertain to a doctor or a specialty (e.g., Orthopedic doctor).
    Do not return JSON or structured data; provide the response in normal text, as if you are explaining to a patient or general public.
    Format the response in bullet points for better readability.
    Information: {extracted_text}
    Response: Provide the answer in simple, understandable language.
    """

    # prompt = f"""
    # You are a helpful assistant specifically designed to answer questions about Shifa International Hospital, its doctors, and related medical information. 
    # If the query is not related to Shifa International Hospital, its doctors, or medical topics, respond with: 
    # "Sorry, I am here just to assist you about the doctors and services at Shifa International Hospital."
    # Otherwise, provide a clear and concise response based on the following information:
    # Information: {extracted_text}
    # Response: Provide the answer in simple, understandable language.
    # """


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




# def process_query_with_rag_and_llm(query):
#     # Basic keyword check to determine if the query is relevant
#     medical_keywords = [
#         "doctor", "hospital", "medicine", "specialist", "urology", "cardiology", 
#         "Shifa International", "health", "disease", "clinic", "surgery"
#     ]
#     if not any(keyword.lower() in query.lower() for keyword in medical_keywords):
#         return "Sorry, I am here just to assist you about the doctors and services at Shifa International Hospital."

#     # Step 1: Query the Vector Store (RAG)
#     query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
#     k = 2  # Retrieve top 2 nearest neighbors
#     distances, indices = index.search(query_embedding, k)

#     # Get the relevant documents based on the search
#     relevant_docs = [document_texts[idx] for idx in indices[0]]

#     # Step 2: Send the result of RAG to the LLM for further processing
#     extracted_text = " ".join(relevant_docs)  # Combine the retrieved documents into one context

#     # Create a message for the LLM (Groq)
#     prompt = f"""
#     You are a helpful assistant specifically designed to answer questions about Shifa International Hospital, its doctors, and related medical information. 
#     If the query is not related to Shifa International Hospital, its doctors, or medical topics, respond with: 
#     "Sorry, I am here just to assist you about the doctors and services at Shifa International Hospital."
#     Otherwise, provide a clear and concise response based on the following information:
#     Information: {extracted_text}
#     Response: Provide the answer in simple, understandable language.
#     """

#     # Get the LLM response
#     completion = groq_client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=1,
#         max_completion_tokens=1024,
#         top_p=1,
#         stream=False,
#     )

#     response_content = completion.choices[0].message.content.strip()
#     return response_content

# Flask route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

import logging

logging.basicConfig(level=logging.DEBUG)
@app.route('/ask', methods=['POST'])
def ask_query():
    try:
        # Get the query from the form
        query = request.form['query']
        if not query:
            return render_template('index.html', error="Query parameter is required.")

        # Process the query with the RAG + LLM pipeline
        response = process_query_with_rag_and_llm(query)

        # Format the response into proper HTML
        formatted_response = response.replace('\n', '<br>').replace('*', '<li>')
        formatted_response = f"<ul>{formatted_response}</ul>"

        # Return the result
        return render_template('index.html', response=formatted_response)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
