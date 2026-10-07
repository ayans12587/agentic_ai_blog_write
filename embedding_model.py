import os
from dotenv import load_dotenv
# 1. Notice the updated class name import path
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Look for your token in the parent folder setup
load_dotenv()

# Verify the token loaded safely from your environment
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Missing HF_TOKEN inside your .env configuration file!")

# 2. Use HuggingFaceEndpointEmbeddings for serverless cloud processing
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Changed parameter from model_name to model
    task="feature-extraction",                      # Explicitly define the vector generation task
    huggingfacehub_api_token=hf_token
)

text = "What is the capital of India?"
vector = embeddings.embed_query(text)

# Prints out the cloud-computed vector profile array dimensions
print(f"Vector Array (Total Dimensions: {len(vector)}):")
print(vector[:5]) # Prints just the first 5 floats to keep the terminal clean
