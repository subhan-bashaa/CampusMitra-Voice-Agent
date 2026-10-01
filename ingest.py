import json
from pathlib import Path

from config import settings

try:
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker import HierarchicalChunker
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

import pypdf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def parse_with_docling(pdf_path: Path) -> list[dict]:
    """Parse document using Docling hierarchical chunker."""
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    chunker = HierarchicalChunker(max_characters=settings.chunk_size)
    chunks = []
    for c in chunker.chunk(result.document):
        headings = c.meta.headings or []
        content = c.text.strip()
        breadcrumb = " > ".join(headings)
        chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content
        chunks.append({
            "source": pdf_path.name,
            "headings": headings,
            "content": content,
            "chunk_text": chunk_text,
        })
    return chunks


def parse_with_pypdf(pdf_path: Path) -> list[dict]:
    """
    Fallback parser using pypdf for clean, fast extraction with section and heading awareness.
    Breaks content into meaningful semantic sections (headers, tables, FAQ, bullet points).
    """
    reader = pypdf.PdfReader(str(pdf_path))
    chunks = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        current_heading = ""
        current_block = []

        for line in lines:
            # Detect section titles/headers
            is_header = False
            if line.startswith("•") or line.startswith("-") or line.startswith("Q:"):
                is_header = False
            elif len(line) < 60 and not line.endswith(".") and not line.endswith(","):
                # Likely a heading or sub-heading
                is_header = True

            if is_header and len(line) > 3 and not line.startswith("http"):
                if current_block:
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        headings = [pdf_path.stem]
                        if current_heading:
                            headings.append(current_heading)
                        breadcrumb = " > ".join(headings)
                        chunks.append({
                            "source": pdf_path.name,
                            "page": page_idx + 1,
                            "headings": headings,
                            "content": block_content,
                            "chunk_text": f"{breadcrumb}\n\n{block_content}",
                        })
                    current_block = []
                current_heading = line
            else:
                current_block.append(line)

        if current_block:
            block_content = "\n".join(current_block).strip()
            if block_content:
                headings = [pdf_path.stem]
                if current_heading:
                    headings.append(current_heading)
                breadcrumb = " > ".join(headings)
                chunks.append({
                    "source": pdf_path.name,
                    "page": page_idx + 1,
                    "headings": headings,
                    "content": block_content,
                    "chunk_text": f"{breadcrumb}\n\n{block_content}",
                })

    return chunks


def load_and_chunk_pdf(pdf_path: Path) -> list[dict]:
    """Try Docling first, fall back to pypdf if Docling is unavailable or fails."""
    if HAS_DOCLING:
        try:
            print(f"  Attempting Docling parse for {pdf_path.name}...")
            return parse_with_docling(pdf_path)
        except Exception as e:
            print(f"  Docling parse failed ({e}), falling back to pypdf parser.")
    return parse_with_pypdf(pdf_path)


def ingest_documents(docs_dir: str) -> list[dict]:
    """
    CampusMitra Knowledge Ingestion Pipeline:
    1. Scan docs_dir for MVGR PDF files
    2. Parse and chunk each document with source metadata
    3. Generate embeddings with sentence-transformers
    4. Index in Qdrant vector database
    5. Cache structured chunks in JSON for high-performance retrieval
    """
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{docs_dir}'. Add MVGR knowledge PDFs and try again.")
        return []

    print(f"\nFound {len(pdf_files)} MVGR PDF(s) in '{docs_dir}':")
    for f in pdf_files:
        print(f"  - {f.name}")

    # Step 1: Parse and chunk all documents
    print("\n--- Step 1: Parsing & Chunking MVGR Documents ---")
    all_chunks = []

    for pdf_path in pdf_files:
        try:
            chunks = load_and_chunk_pdf(pdf_path)
            print(f"  [OK] {pdf_path.name} -> {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [FAIL] {pdf_path.name}: {e}")
            continue

    if not all_chunks:
        print("No chunks created. Check your documents and try again.")
        return []

    print(f"\nTotal MVGR Knowledge Chunks: {len(all_chunks)}")

    # Ensure data directory exists
    data_dir = Path(settings.qdrant_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save chunks cache to JSON for instant fallback & fast queries
    cache_file = data_dir / "mvgr_chunks.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved chunks cache to {cache_file}")

    # Step 2: Generate embeddings
    print("\n--- Step 2: Generating Sentence Embeddings ---")
    embedder = SentenceTransformer(settings.embedding_model)
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
    print(f"[OK] Embedding shape: {embeddings.shape}")

    # Step 3: Index in Qdrant
    print("\n--- Step 3: Indexing in Qdrant Vector DB ---")
    client = QdrantClient(path=settings.qdrant_path)
    dim = embedder.get_embedding_dimension()

    client.recreate_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "source": chunk.get("source", "MVGR Knowledge Base"),
                "headings": chunk.get("headings", []),
                "content": chunk.get("content", ""),
                "chunk_text": chunk.get("chunk_text", ""),
                "page": chunk.get("page", 1),
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    client.upsert(collection_name=settings.collection_name, points=points, wait=True)

    info = client.get_collection(settings.collection_name)
    print(f"[OK] Indexed {info.points_count} points in collection '{settings.collection_name}' (dim={info.config.params.vectors.size})")

    client.close()
    print("\n[DONE] CampusMitra knowledge ingestion complete. You can now start the voice agent.")
    return all_chunks


if __name__ == "__main__":
    print("CampusMitra — MVGR College Knowledge Ingestion Pipeline")
    print(f"  Docs folder:     {settings.docs_dir}")
    print(f"  Embedding model: {settings.embedding_model}")
    print(f"  Qdrant path:     {settings.qdrant_path}")
    print(f"  Collection:      {settings.collection_name}")
    ingest_documents(settings.docs_dir)
