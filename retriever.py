import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import *
import streamlit as st

#----------------------------------
# LOAD SAVED FILES
#----------------------------------
@st.cache_resource
def load_index():
  return faiss.read_index(FAISS_PATH)

@st.cache_resource
def load_chunks():
  with open(CHUNKS_PATH,"rb") as f:
    return pickle.load(f)


@st.cache_resource
def load_embed_model():
  return SentenceTransformer(EMBED_MODEL)

index = load_index()
chunks = load_chunks()
embed_model = load_embed_model()

#----------------------------------
# HELPER FUNCTION
#---------------------------------- 
def retrieve_text(query,k=TOP_K):

  query_vector = embed_model.encode([query]).astype("float32")
  distances, indices = index.search(query_vector,k)

  results=[]

  for i in indices[0]:
    results.append(chunks[i])
  return results
  

