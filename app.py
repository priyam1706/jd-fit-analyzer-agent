import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os, tempfile

st.set_page_config(page_title="JD Fit Analyzer Agent", page_icon="🤖")
st.title("🤖 JD Fit Analyzer Agent")
st.caption("Upload a Job Description → Get your fit score + gap analysis instantly")

api_key = st.text_input("Enter your Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
candidate_profile = st.text_area("Paste your skills/profile here",
    placeholder="e.g. I know Python, FastAPI, LangChain, React...")

if st.button("🔍 Analyze My Fit") and api_key and uploaded_file and candidate_profile:
    with st.spinner("Analyzing JD..."):
        os.environ["GROQ_API_KEY"] = api_key

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert JD Fit Analyzer Agent.
Use ONLY the job description context below to answer.
Context: {context}
Question: {question}
Give a clear structured answer.
"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )

        questions = {
            "📋 Required Skills": "What are ALL the required skills and technologies for this role?",
            "🎯 Fit Score": f"Candidate profile: {candidate_profile}. Give a fit score out of 10 with reasoning.",
            "❌ Skill Gaps": f"Candidate profile: {candidate_profile}. What skills are they missing?",
            "✉️ Cover Letter Tips": f"Candidate profile: {candidate_profile}. What should they highlight in their cover letter?"
        }

        for label, question in questions.items():
            st.subheader(label)
            result = chain.invoke(question)
            st.write(result)
