# Project Context: Multi-Modal RAG Customer Service Agent
- **Goal:** Design and implement a multi-modal RAG (Retrieval-Augmented Generation) system for a customer service agent that can read product manuals (text and images) and answer user queries, returning both textual answers and relevant images.
- **Tech Stack:** Python, LangChain, Qdrant (Vector Database), OpenAI compatible LLM (Qwen/Qwen2.5-7B-Instruct), Vision Language Model (Qwen3-VL-8B-Instruct).

# Progress Log (2026-03-26)
## Database and Persistence
- Migrated vector database from ChromaDB to **Qdrant**.
- Set Qdrant to local persistent storage (commented out `delete_collection` on initialization) to avoid rebuilding the embedding database repeatedly and save API tokens.

## Architecture Overhaul: Image Retrieval
- **Old Method:** Relied on `<PIC>` text anchors and proximity matching, which was unreliable.
- **New Method (VLM Summarization):**
  - Passed each image alongside its corresponding manual text (as context) to a Vision Language Model (`Qwen3-VL-8B-Instruct`).
  - Instructed the VLM to generate a short, precise summary (under 50 words) focusing on the main object and its action/state, rather than a detailed list of all parts.
  - Stored these generated image summaries as standard text documents in Qdrant with `metadata={"type": "image_summary", ...}`.
  - When querying, if the semantic search matches an `image_summary` document, the system retrieves and returns the corresponding image path.

## Cross-Manual Contamination Fix
- Implemented a **Keyword-routed Retrieval** system (`hybrid/keyword router`).
- Before semantic search, the system extracts keywords (product names) from the user's query.
- It dynamically builds a Qdrant `rest.Filter` to lock the retrieval strictly to chunks where `metadata.manual` matches the recognized product (e.g., query about "洗碗机" only searches the "洗碗机手册").

## Current State & Next Steps
- Successfully tested the new VLM image summarization and keyword routing pipeline on a single manual (`VR头显手册.txt`) with a query about wearing/adjusting the VR headset. The generated summaries were highly accurate and concise.
- **Next Action:**
  1. Remove the temporary hardcoded filter in `_to_docs` (`manual_files = [p for p in manual_files if p.name == "VR头显手册.txt"]`).
  2. Rebuild the full knowledge base for all product manuals using the new VLM pipeline.
  3. Run comprehensive tests across different manuals.