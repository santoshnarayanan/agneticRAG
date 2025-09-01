Purpose - ingestion.py
Creates a searchable knowledge base from web articles that can be queried to provide context-aware answers.
What It Does
1. Data Collection

Scrapes content from 3 AI/ML blog posts from Lilian Weng's website
Topics include: AI agents, prompt engineering, and adversarial attacks on LLMs

2. Document Processing

Flattens the loaded documents into a single list
Splits long articles into smaller 250-token chunks (with no overlap)
Uses tiktoken encoder for accurate token counting

3. Vector Database Creation

Converts text chunks into numerical vectors using OpenAI embeddings
Stores these vectors in a Chroma database (locally in ./chroma folder)
Each chunk becomes searchable by semantic similarity

4. Retrieval Setup

Creates a retriever interface that can find relevant chunks based on queries
The retriever will return the most semantically similar chunks to any question