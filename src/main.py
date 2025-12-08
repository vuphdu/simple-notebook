"""
Simple RAG System - Main Entry Point

This module provides the main CLI interface for the RAG system.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    settings, ensure_directories,
    DOCUMENTS_DIR, VECTORDB_DIR, PROCESS_INPUT_DIR, PROCESS_OUTPUT_DIR
)
from src.chunking import DocumentChunker, chunk_documents
from src.vectorization import TextVectorizer, VectorStore, get_vectorizer, get_vector_store
from src.search import SearchEngine, get_search_engine
from src.sequence_chart import SequenceChartProcessor, get_chart_processor


def process_documents(
    input_dir: Optional[Path] = None,
    include_charts: bool = True
):
    """
    Process all documents in the input directory.
    
    - Chunks documents
    - Extracts and processes sequence charts
    - Vectorizes content
    - Stores in vector database
    """
    input_dir = input_dir or DOCUMENTS_DIR
    input_dir = Path(input_dir)
    
    print(f"Processing documents from: {input_dir}")
    print(f"Vector DB location: {VECTORDB_DIR}")
    print("-" * 50)
    
    # Initialize components
    chunker = DocumentChunker()
    vectorizer = get_vectorizer()
    vector_store = get_vector_store()
    chart_processor = get_chart_processor()
    
    all_vectorized = []
    
    # 1. Process regular documents
    print("\n[1/3] Chunking documents...")
    chunks = chunker.chunk_directory(input_dir)
    print(f"Total chunks: {len(chunks)}")
    
    if chunks:
        print("\n[2/3] Vectorizing text chunks...")
        vectorized_chunks = vectorizer.vectorize_chunks(chunks)
        all_vectorized.extend(vectorized_chunks)
    
    # 2. Process sequence charts
    if include_charts:
        print("\n[2b] Processing sequence charts...")
        for md_file in input_dir.glob("**/*.md"):
            try:
                charts = chart_processor.process_file(md_file, export_images=True)
                if charts:
                    chart_data = chart_processor.get_vectorization_data(charts)
                    
                    # Vectorize chart descriptions
                    for data in chart_data:
                        embedding = vectorizer.vectorize(data["content"])[0].tolist()
                        data["embedding"] = embedding
                    
                    all_vectorized.extend(chart_data)
                    print(f"  Added {len(charts)} charts from {md_file.name}")
            except Exception as e:
                print(f"  Error processing {md_file}: {e}")
    
    # 3. Store in vector database
    if all_vectorized:
        print(f"\n[3/3] Storing {len(all_vectorized)} items in vector database...")
        vector_store.add_chunks(all_vectorized)
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Total items in database: {vector_store.count()}")


def search_documents(
    query: str,
    top_k: int = 5,
    save_results: bool = True
):
    """
    Search through vectorized documents.
    """
    print(f"Searching for: '{query}'")
    print("-" * 50)
    
    engine = get_search_engine()
    
    if save_results:
        results = engine.search_and_save(query, top_k=top_k)
    else:
        results = engine.search(query, top_k=top_k)
    
    # Display results
    print(engine.format_results(results, format_type="text"))
    
    return results


def show_stats():
    """Show vector database statistics."""
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    print("Vector Database Statistics")
    print("-" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple RAG System - Document Processing and Search"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process and vectorize documents"
    )
    process_parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DOCUMENTS_DIR,
        help=f"Input directory (default: {DOCUMENTS_DIR})"
    )
    process_parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip sequence chart processing"
    )
    
    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search through documents"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    search_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save search input/output"
    )
    
    # Stats command
    subparsers.add_parser(
        "stats",
        help="Show vector database statistics"
    )
    
    # Init command
    subparsers.add_parser(
        "init",
        help="Initialize directory structure"
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    if args.command == "process":
        process_documents(
            input_dir=args.input,
            include_charts=not args.no_charts
        )
    
    elif args.command == "search":
        search_documents(
            query=args.query,
            top_k=args.top_k,
            save_results=not args.no_save
        )
    
    elif args.command == "stats":
        show_stats()
    
    elif args.command == "init":
        print("Directory structure initialized:")
        print(f"  Documents: {DOCUMENTS_DIR}")
        print(f"  Vector DB: {VECTORDB_DIR}")
        print(f"  Process Input: {PROCESS_INPUT_DIR}")
        print(f"  Process Output: {PROCESS_OUTPUT_DIR}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
