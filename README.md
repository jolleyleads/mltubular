# mltubular

## Overview
Production-style Machine Learning API scaffold with an LLM-ready
Retrieval-Augmented Generation (RAG) layer.

This project prioritizes **clean architecture**, **reproducibility**,
and **deployment realism** over notebooks-only experimentation.

## Architecture
Input → Retrieval (Embeddings) → Context Assembly → LLM Stub → API Output

## Why This Matters
• API-first ML systems  
• RAG over fine-tuning  
• Clear inference boundaries  
• Designed for scale and deployment  

## RAG Layer
This repo includes a minimal RAG implementation using:
• Vector embeddings (stubbed)
• In-memory vector store
• Retrieval before generation

## Modeling & Analysis
See /notebooks/toy_analysis.ipynb

Includes:
• Feature reasoning
• Train/test split rationale
• Metric selection (accuracy vs precision tradeoffs)
• Future model selection strategy

## Future Extensions
• Replace embedding stub with OpenAI / SentenceTransformers
• Add persistent vector DB (FAISS / Chroma)
• Streaming inference
• Evaluation harness

This repository is intentionally structured to mirror real-world ML
and LLM systems used in production.
