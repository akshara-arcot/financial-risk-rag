from config import *
from retriever import retrieve_text
from openai import OpenAI
import os

#----------------------------------
# LOAD OPENAI KEY
#----------------------------------
try:
  import streamlit as st
  api_key = os.getenv("OPEN_API_KEY") or st.secrets["OPEN_API_KEY"]
except:
  api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key = api_key)
#----------------------------------
# HELPER FUNCTIONS
#----------------------------------

def get_context(query_result):
  context_parts = []
  for doc in query_result:
    source = doc["source"]
    page = doc["page"]
    context_parts.append(f"Source: {source}\nPage: {page} \n{doc['chunk_text']}")
  return context_parts

def ask_query(query, k=TOP_K):
  
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
