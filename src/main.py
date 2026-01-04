"""
Simple RAG System - Main Entry Point

This module provides the main CLI interface for the RAG system.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    settings, ensure_directories,
    DOCUMENTS_DIR, INPUT_DOCS_DIR, EXTRACTED_IMAGES_DIR,
    VECTORDB_DIR, FAISS_INDEX_DIR, PROCESS_INPUT_DIR, PROCESS_OUTPUT_DIR, MODELS_DIR
)
from src.chunking import DocumentChunker, chunk_documents
from src.vectorization import TextVectorizer, VectorStore, get_vectorizer, get_vector_store
from src.search import SearchEngine, get_search_engine
from src.sequence_chart import SequenceChartProcessor, get_chart_processor
from src.image_extraction import ImageExtractor, get_image_extractor
from src.image_tag import add_tag


def process_documents(
    input_dir: Optional[Path] = None,
    include_charts: bool = True,
    extraction_mode: str = "pymu"
):
    """
    Process all documents in the input directory.
    
    - Chunks documents
    - Extracts and processes sequence charts
    - Vectorizes content
    - Stores in vector database
    
    Args:
        input_dir: Input directory path
        include_charts: Whether to process sequence charts
        extraction_mode: Image extraction mode - "pymu" or "fullpage"
    """
    input_dir = input_dir or DOCUMENTS_DIR
    input_dir = Path(input_dir)
    
    backend = settings.vectordb.backend
    print(f"Processing documents from: {input_dir}")
    print(f"Vector store backend: {backend}")
    if backend == "faiss":
        print(f"FAISS index location: {FAISS_INDEX_DIR}")
    else:
        print(f"ChromaDB location: {VECTORDB_DIR}")
    print("-" * 50)
    
    # Initialize components
    chunker = DocumentChunker()
    vectorizer = get_vectorizer()
    vector_store = get_vector_store()
    chart_processor = get_chart_processor()
    
    all_vectorized = []
    
    # 1. Process regular documents (Text files including PDF)
    print("\n[1/3] Chunking documents (txt, md, docx, pdf)...")
    # Include PDF in text chunking as requested
    chunks = chunker.chunk_directory(
        input_dir, 
        extensions=['.txt', '.md', '.docx', '.markdown', '.pdf']
    )
    print(f"Total text chunks: {len(chunks)}")
    
    if chunks:
        print("\n[2/3] Vectorizing text chunks...")
        vectorized_chunks = vectorizer.vectorize_chunks(chunks)
        all_vectorized.extend(vectorized_chunks)
    
    # 2. Process PDFs (Extract Images) - Collect first, batch vectorize later
    print(f"\n[2a] Extracting images from PDFs (mode: {extraction_mode})...")
    extractor = get_image_extractor(mode=extraction_mode)
    pdf_files = list(input_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    # Collect all image data first (extraction only, no vectorization yet)
    all_image_data = []
    for pdf_file in pdf_files:
        try:
            extracted_pages = extractor.extract_from_file(pdf_file)
            if extracted_pages:
                page_data = extractor.get_vectorization_data(extracted_pages)
                all_image_data.extend(page_data)
                print(f"  Extracted {len(extracted_pages)} images from {pdf_file.name}")
        except Exception as e:
            print(f"  Error processing PDF {pdf_file}: {e}")
    
    # Batch vectorize all image descriptions (single GPU call)
    if all_image_data:
        print(f"\n[2a] Vectorizing {len(all_image_data)} image descriptions (batch)...")
        image_contents = [d["content"] for d in all_image_data]
        image_embeddings = vectorizer.vectorize(image_contents)
        for data, emb in zip(all_image_data, image_embeddings):
            data["embedding"] = emb.tolist()
        all_vectorized.extend(all_image_data)

    # 3. Process sequence charts (if any MD files) - Collect first, batch vectorize later
    if include_charts:
        print("\n[2b] Processing sequence charts...")
        all_chart_data = []
        for md_file in input_dir.glob("**/*.md"):
            try:
                charts = chart_processor.process_file(md_file, export_images=True)
                if charts:
                    chart_data = chart_processor.get_vectorization_data(charts)
                    all_chart_data.extend(chart_data)
                    print(f"  Found {len(charts)} charts from {md_file.name}")
            except Exception as e:
                print(f"  Error processing {md_file}: {e}")
        
        # Batch vectorize all chart descriptions (single GPU call)
        if all_chart_data:
            print(f"\n[2b] Vectorizing {len(all_chart_data)} chart descriptions (batch)...")
            chart_contents = [d["content"] for d in all_chart_data]
            chart_embeddings = vectorizer.vectorize(chart_contents)
            for data, emb in zip(all_chart_data, chart_embeddings):
                data["embedding"] = emb.tolist()
            all_vectorized.extend(all_chart_data)
    
    # 4. Store in vector database
    if all_vectorized:
        print(f"\n[3/3] Storing {len(all_vectorized)} items in vector database...")
        vector_store.add_chunks(all_vectorized)
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Total items in database: {vector_store.count()}")


def search_documents(
    query: str,
    top_k: int = 5,
    save_results: bool = True,
    format_type: str = "text",
    quiet: bool = False,
    use_hybrid: bool = None
):
    """
    Search through vectorized documents.
    
    Args:
        query: Search query text.
        top_k: Number of results to return.
        save_results: Whether to save results to process directory.
        format_type: Output format ('text', 'json', 'compact', 'markdown').
        quiet: If True, suppress header messages (useful for AI tools).
        use_hybrid: Enable/disable hybrid search. None = use config default.
    """
    if not quiet:
        print(f"Searching for: '{query}'")
        print(f"Using backend: {settings.vectordb.backend}")
        hybrid_status = "enabled" if (use_hybrid if use_hybrid is not None else settings.search.use_hybrid) else "disabled"
        print(f"Hybrid search: {hybrid_status} (alpha={settings.search.hybrid_alpha})")
        print("-" * 50)
    
    engine = get_search_engine()
    
    if save_results:
        results = engine.search_and_save(query, top_k=top_k, use_hybrid=use_hybrid)
    else:
        results = engine.search(query, top_k=top_k, use_hybrid=use_hybrid)
    
    # Display results
    print(engine.format_results(results, format_type=format_type))
    
    return results


def show_stats():
    """Show vector database statistics."""
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    print("Vector Database Statistics")
    print("-" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def index_code(
    input_path: Path,
    project_tag: Optional[str] = None,
    extensions: Optional[list[str]] = None
):
    """
    Index source code files for RAG search.
    
    This is optimized for indexing large codebases like wpa_supplicant,
    Linux kernel modules, etc. for AI assistant retrieval.
    
    Args:
        input_path: Path to code directory or file.
        project_tag: Tag to identify this codebase (e.g., 'wpa_supplicant', 'wifi_driver').
        extensions: File extensions to include. Default: common code files.
    """
    input_path = Path(input_path)
    project_tag = project_tag or input_path.name
    
    print(f"📁 Indexing code from: {input_path}")
    print(f"🏷️  Project tag: {project_tag}")
    print("-" * 50)
    
    # Default code extensions
    if extensions is None:
        extensions = [
            '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx',  # C/C++
            '.py', '.pyi',  # Python
            '.java',  # Java
            '.js', '.jsx', '.ts', '.tsx',  # JavaScript/TypeScript
            '.go', '.rs', '.rb', '.php',  # Other languages
            '.sh', '.bash',  # Shell scripts
            '.yaml', '.yml', '.json', '.toml',  # Config files
        ]
    
    # Initialize components
    chunker = DocumentChunker()
    vectorizer = get_vectorizer()
    vector_store = get_vector_store()
    
    # Find and process files
    if input_path.is_file():
        files = [input_path]
    else:
        files = []
        for ext in extensions:
            files.extend(input_path.rglob(f"*{ext}"))
    
    print(f"Found {len(files)} code files")
    
    all_vectorized = []
    processed = 0
    
    for file_path in files:
        try:
            # Chunk the file
            chunks = chunker.chunk_file(file_path)
            
            # Add project metadata
            for chunk in chunks:
                chunk.metadata["project"] = project_tag
                chunk.metadata["type"] = "source_code"
                chunk.metadata["language"] = file_path.suffix.lstrip('.')
                chunk.metadata["relative_path"] = str(file_path.relative_to(input_path) if input_path.is_dir() else file_path.name)
            
            # Vectorize
            vectorized = vectorizer.vectorize_chunks(chunks)
            all_vectorized.extend(vectorized)
            processed += 1
            
            if processed % 50 == 0:
                print(f"  Processed {processed}/{len(files)} files...")
                
        except Exception as e:
            print(f"  ⚠️ Error processing {file_path.name}: {e}")
    
    # Store in vector database
    if all_vectorized:
        print(f"\n💾 Storing {len(all_vectorized)} code chunks...")
        vector_store.add_chunks(all_vectorized)
    
    print("\n" + "=" * 50)
    print(f"✅ Code indexing complete!")
    print(f"   Files processed: {processed}")
    print(f"   Chunks created: {len(all_vectorized)}")
    print(f"   Project tag: {project_tag}")
    print(f"\n💡 Search with: python -m src.main query 'wifi connection handling'")


def extract_images(
    input_path: Optional[Path] = None,
    vectorize: bool = True
):
    """
    Extract images from documents using PyMuPDF.
    
    Args:
        input_path: Path to file or directory.
        vectorize: Whether to vectorize and store extracted images.
    """
    input_path = input_path or DOCUMENTS_DIR
    input_path = Path(input_path)
    
    print(f"Extracting images from: {input_path}")
    print("-" * 50)
    
    extractor = get_image_extractor()
    
    # Extract images
    if input_path.is_file():
        extracted = extractor.extract_from_file(input_path)
    else:
        extracted = extractor.extract_from_directory(input_path)
    
    print(f"\nExtracted {len(extracted)} images")
    
    # Vectorize and store
    if vectorize and extracted:
        print("\nVectorizing extracted images...")
        vectorizer = get_vectorizer()
        vector_store = get_vector_store()
        
        # Prepare data for vectorization
        vectorization_data = extractor.get_vectorization_data(extracted)
        
        # Vectorize
        for data in vectorization_data:
            embedding = vectorizer.vectorize(data["content"])[0].tolist()
            data["embedding"] = embedding
        
        # Store
        vector_store.add_chunks(vectorization_data)
        print(f"Stored {len(vectorization_data)} image descriptions in vector database")
    
    print("\n" + "=" * 50)
    print("Image extraction complete!")
    
    # Show extracted images info
    for img in extracted[:5]:  # Show first 5
        print(f"  - {img.region_type}: {Path(img.image_path).name}")
    if len(extracted) > 5:
        print(f"  ... and {len(extracted) - 5} more")


def update_context(
    query: str,
    new_context: str,
    confirm: bool = True
):
    """
    Update context for a document found by query.
    
    Args:
        query: Search query to find the document.
        new_context: New context text to append.
        confirm: Whether to ask for confirmation.
    """
    print(f"Searching for document matching: '{query}'")
    
    # 1. Search for the item
    # We use the search engine directly to get raw results
    engine = get_search_engine()
    results = engine.search(query, top_k=1)
    
    if not results:
        print("No documents found matching the query.")
        return

    result = results[0]
    doc_id = result.chunk_id
    current_content = result.document
    current_metadata = result.metadata
    similarity = result.similarity
    
    print("-" * 50)
    print(f"Found item (ID: {doc_id})")
    print(f"Match score: {similarity:.4f}")
    print(f"Source: {current_metadata.get('source_file', 'Unknown')}")
    print(f"Current Content Preview:\n{current_content[:200]}...")
    print("-" * 50)
    
    if confirm:
        response = input("\nIs this the correct document to update? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return

    # 2. Update content
    # We append the new context to the existing content
    updated_content = f"{current_content}\n\n[User Note]: {new_context}"
    
    print("\nRe-vectorizing content...")
    
    # 3. Re-vectorize
    vectorizer = get_vectorizer()
    new_embedding = vectorizer.vectorize(updated_content)[0].tolist()
    
    # 4. Update Store
    vector_store = get_vector_store()
    
    # Update metadata to track modification
    updated_metadata = current_metadata.copy()
    updated_metadata['user_modified'] = True
    
    vector_store.update(
        ids=[doc_id],
        embeddings=[new_embedding],
        documents=[updated_content],
        metadatas=[updated_metadata]
    )
    
    print(f"\nSuccessfully added context to document {doc_id}!")
    print("New content is now searchable.")





def clean_data(include_models: bool = False, skip_confirm: bool = False):
    """
    Clean extracted data.
    
    Args:
        include_models: If True, also delete downloaded models.
        skip_confirm: If True, skip confirmation prompt.
    """
    import shutil
    
    dirs_to_clean = [
        (VECTORDB_DIR, "ChromaDB database"),
        (FAISS_INDEX_DIR, "FAISS index"),
        (EXTRACTED_IMAGES_DIR, "Extracted images"),
        (PROCESS_INPUT_DIR, "Process input"),
        (PROCESS_OUTPUT_DIR, "Process output"),
    ]
    
    if include_models:
        dirs_to_clean.append((MODELS_DIR, "Downloaded models"))
    
    # Show what will be deleted
    print("The following directories will be cleaned:")
    total_size = 0
    for dir_path, name in dirs_to_clean:
        if dir_path.exists():
            # Calculate size
            size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  {name}: {dir_path} ({size_mb:.2f} MB)")
        else:
            print(f"  {name}: {dir_path} (not exists)")
    
    print(f"\nTotal: {total_size / (1024 * 1024):.2f} MB")
    
    # Confirm
    if not skip_confirm:
        response = input("\nAre you sure you want to delete these? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Clean directories
    print("\nCleaning...")
    for dir_path, name in dirs_to_clean:
        if dir_path.exists():
            try:
                # Remove all contents but keep the directory
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print(f"  ✓ Cleaned {name}")
            except Exception as e:
                print(f"  ✗ Error cleaning {name}: {e}")
        else:
            print(f"  - {name} not exists, skipped")
    
    # Recreate directories
    ensure_directories()
    print("\nClean completed!")


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
        default=INPUT_DOCS_DIR,
        help=f"Input directory (default: {INPUT_DOCS_DIR})"
    )
    process_parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip sequence chart processing"
    )
    
    # Image extraction mode - mutually exclusive
    extract_group = process_parser.add_mutually_exclusive_group()
    extract_group.add_argument(
        "--pymu", "-pymu",
        action="store_true",
        help="Use PyMuPDF smart cropping for images (default)"
    )
    extract_group.add_argument(
        "--fp", "-fp",
        action="store_true",
        help="Full-page rendering at 150%% zoom when images detected"
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
    search_parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "json", "compact", "markdown"],
        default="text",
        help="Output format (default: text). Use 'compact' for AI assistants."
    )
    search_parser.add_argument(
        "--hybrid",
        action="store_true",
        default=None,
        help="Enable hybrid search (vector + BM25 keyword matching)"
    )
    search_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search, use pure vector similarity"
    )
    
    # Query command - Quick search for AI assistants (alias with sensible defaults)
    query_parser = subparsers.add_parser(
        "query",
        help="Quick search for AI assistants (compact output, no logging)"
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    query_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results (default: 10)"
    )
    query_parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "json", "compact", "markdown"],
        default="compact",
        help="Output format (default: compact)"
    )
    query_parser.add_argument(
        "--hybrid",
        action="store_true",
        default=None,
        help="Enable hybrid search (vector + BM25)"
    )
    query_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search"
    )
    
    # Index-code command - Index source code for RAG
    index_code_parser = subparsers.add_parser(
        "index-code",
        help="Index source code files for RAG search (for AI assistants)"
    )
    index_code_parser.add_argument(
        "path",
        type=Path,
        help="Path to code directory or file"
    )
    index_code_parser.add_argument(
        "--tag", "-t",
        type=str,
        default=None,
        help="Project tag for identification (default: directory name)"
    )
    index_code_parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to include (e.g., .c .h .py)"
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
    
    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean extracted data (vectordb, process input/output)"
    )
    clean_parser.add_argument(
        "--all",
        action="store_true",
        help="Also clean downloaded models"
    )
    clean_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Extract images command
    extract_parser = subparsers.add_parser(
        "extract-images",
        help="Extract images from documents using PyMuPDF"
    )
    extract_parser.add_argument(
        "--input", "-i",
        type=Path,
        default=INPUT_DOCS_DIR,
        help=f"Input file or directory (default: {INPUT_DOCS_DIR})"
    )
    extract_parser.add_argument(
        "--no-vectorize",
        action="store_true",
        help="Skip vectorization of extracted images"
    )

    # Update command
    update_parser = subparsers.add_parser(
        "update",
        help="Add context to a document/image"
    )
    update_parser.add_argument(
        "query",
        type=str,
        help="Query to find the document (e.g. 'page 100')"
    )
    update_parser.add_argument(
        "context",
        type=str,
        help="New context to add"
    )
    update_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation"
    )
    

    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Start timing
    start_time = time.time()
    command_name = args.command or "help"
    
    if args.command == "process":
        # Determine extraction mode from flags
        if args.fp:
            extraction_mode = "fullpage"
        else:  # default or --pymu explicitly set
            extraction_mode = "pymu"
        
        process_documents(
            input_dir=args.input,
            include_charts=not args.no_charts,
            extraction_mode=extraction_mode
        )
    
    elif args.command == "search":
        # Determine hybrid mode: --hybrid overrides, --no-hybrid disables, else use config
        use_hybrid = None
        if getattr(args, 'hybrid', False):
            use_hybrid = True
        elif getattr(args, 'no_hybrid', False):
            use_hybrid = False
        
        search_documents(
            query=args.query,
            top_k=args.top_k,
            save_results=not args.no_save,
            format_type=args.format,
            use_hybrid=use_hybrid
        )
    
    elif args.command == "query":
        # Determine hybrid mode
        use_hybrid = None
        if getattr(args, 'hybrid', False):
            use_hybrid = True
        elif getattr(args, 'no_hybrid', False):
            use_hybrid = False
        
        # Quick search for AI - quiet mode, no save, compact format
        search_documents(
            query=args.query,
            top_k=args.top_k,
            save_results=False,
            format_type=args.format,
            quiet=True,
            use_hybrid=use_hybrid
        )
    
    elif args.command == "index-code":
        index_code(
            input_path=args.path,
            project_tag=args.tag,
            extensions=args.ext
        )
    
    elif args.command == "stats":
        show_stats()
    
    elif args.command == "init":
        print("Directory structure initialized:")
        print(f"  Documents: {DOCUMENTS_DIR}")
        print(f"  Vector DB: {VECTORDB_DIR}")
        print(f"  Process Input: {PROCESS_INPUT_DIR}")
        print(f"  Process Output: {PROCESS_OUTPUT_DIR}")
    
    elif args.command == "clean":
        clean_data(
            include_models=args.all,
            skip_confirm=args.yes
        )
    
    elif args.command == "extract-images":
        extract_images(
            input_path=args.input,
            vectorize=not args.no_vectorize
        )

    elif args.command == "update":
        update_context(
            query=args.query,
            new_context=args.context,
            confirm=not args.yes
        )
    

    
    else:
        parser.print_help()
    
    # Show elapsed time for all commands (except help)
    if args.command:
        elapsed = time.time() - start_time
        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"\n⏱️  Completed in {minutes}m {seconds:.1f}s")
        else:
            print(f"\n⏱️  Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
