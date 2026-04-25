# 🤖 JD Fit Analyzer Agent

An agentic AI system that analyzes job descriptions and scores candidate fit.

## What it does
- Upload any Job Description (PDF)
- Paste your skills/profile
- Get instant: Required skills, Fit score out of 10, Skill gaps, Cover letter tips

## Tech Stack
Python · LangChain · ChromaDB · Groq (LLaMA 3.3) · HuggingFace Embeddings · Streamlit · RAG Pipeline

## How it works
PDF → Chunks → Embeddings → ChromaDB → Retrieved Context → LLaMA 3.3 via Groq → Structured Fit Report

## How to run
1. Clone the repo
2. Install dependencies: `pip install langchain langchain-community langchain-groq chromadb pypdf streamlit sentence-transformers langchain-text-splitters`
3. Get a free Groq API key from console.groq.com
4. Run: `streamlit run app.py`

## Example output
- 📋 Required Skills extracted from JD
- 🎯 Fit Score: 8/10 with reasoning
- ❌ Skill gaps identified
- ✉️ Personalized cover letter tips
