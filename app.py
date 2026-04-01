import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

#----------------------------------
# SETTINGS
#----------------------------------
FAISS_PATH = "index/faiss.index"
CHUNKS_PATH = "index/chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

#----------------------------------
# LOAD OPENAI KEY
#----------------------------------
client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

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
  chunked_docs = load_chunks()
  embed_model == load_embed_model()

#----------------------------------
# HELPER FUNCTIONS
#---------------------------------- 
def retrieve_text(query,k=5):

  query_vector = embedding_model.encode([query]).astype("float32")
  distances, indices = index.search(query_vector,k)

  results=[]

  for i in indices[0]:
    results.append(chunks[i])
  return results
  
  def get_context(query_result):
    context_parts = []
    for doc in query_result:
      source = doc["source"]
      page = doc["page"]
      context_parts.append(f"Source: {source}\nPage: {page} \n{doc["chunk_text"]}")
  return context_parts

def ask_query(query, k):
  
  # Retrieve relevant chunks from the FAISS index
  retrieved_chunks = retrieve_text(query,k)

  # Merge the content of retrieved documents into a single context string
  context_text = get_context(retrieved_chunks)

  # Define the prompt for the language model
  prompt = f"""
  You are a finanical risk assistant
  Only based on this context {context_text}
  Answer this question {query}
  Do not give a summary unless the question is asking for a summary

  Be conscise and to the point. Give a direct answer and short explanation

  Before answering, first determine if context directly answers the question. If not, return no information.

  Return:
  Just print answer to the question
  Provide KEY INSIGHTS in bullet points. Top 5 most relevant points.
  If relevant information is found, provide all the sources used to answer the question; Separate page numbers with ",". Add retreival Ranking
  If relevant information is not found, then for DOCUMENT SOURCES, write : None (no relevant context found). SKIP KEY INSIGHTS


  example:
  DOCUMENT SOURCES:
  Rank 1: ally-20241231.pdf | Page 34, 45



  Add a MODEL NOTE at the bottom that:
  Answer was generated using supporting documents only

  """

  # Call the LLM to generate a response
  model_response = client.responses.create(
      model = LLM_MODEL,
      input =prompt
  )

  return model_response.output_text,context_text 

#----------------------------------
# STREAMLIT UI
#----------------------------------

st.title("Financial Risk RAG Assistant")
st.write("Ask questions over annual reports and 10-K filings")

question = st.text_input("Enter your question:")

if st.button("Ask"):
  if question.strip():
    with st.spinner("Generating answer..."):
      answer = ask_query(question,k=3)
    st.text(answer)
    

