import streamlit as st
from rag_pipeline import ask_query

#----------------------------------
# STREAMLIT UI
#----------------------------------

st.title("Financial Risk RAG Assistant")
st.write("Ask questions over annual reports and 10-K filings")

question = st.text_input("Enter your question:")

if st.button("Ask"):
  if question.strip():
    with st.spinner("Generating answer..."):
      answer, retrieved_chunks = ask_query(question)
    st.subheader("Answer")
    st.text(answer)
    

