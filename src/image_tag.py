"""
Image Tag Management Module

This module provides functionality to add tags to images in the vector database.
"""
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorization import get_vectorizer, get_vector_store


def add_tag(
    image_id: str,
    tag: str,
    confirm: bool = True
) -> bool:
    """
    Add a tag/context to an image by its ID.
    
    Args:
        image_id: Image ID (e.g., 'HFP_v1.8_p92_crop_1' or 'img_abc12345')
        tag: Tag/context text to add (e.g., 'Established SCO connection')
        confirm: Whether to ask for confirmation.
    
    Returns:
        True if tag was added successfully, False otherwise.
    
    Example:
        python -m src.main addtag HFP_v1.8_p92_crop_1 "Established SCO connection"
    """
    # Get vector store
    vector_store = get_vector_store()
    
    # Search for the image by ID
    # FAISS store stores IDs as chunk_id which is "img_{hash}" format
    # But user might provide filename format like "HFP_v1.8_p92_crop_1"
    
    found_id = None
    
    # Check if ID exists directly (internal format)
    if image_id in vector_store._documents:
        found_id = image_id
    # Check with img_ prefix
    elif f"img_{image_id}" in vector_store._documents:
        found_id = f"img_{image_id}"
    else:
        # Search by filename pattern in metadata
        for doc_id, metadata in vector_store._metadata.items():
            if metadata.get('type') == 'extracted_image':
                img_path = metadata.get('image_path', '')
                # Match by filename (without .png)
                if image_id in img_path:
                    found_id = doc_id
                    break
    
    if not found_id:
        print(f"❌ Image '{image_id}' not found in database.")
        print("\nTip: Use image ID from search results, e.g.:")
        print("  HFP_v1.8_p14_crop_1")
        print("  img_abc12345")
        return False
    
    # Get current content
    found_document = vector_store._documents.get(found_id, "")
    found_metadata = vector_store._metadata.get(found_id, {})
    
    print(f"\n✅ Found image: {found_id}")
    print("-" * 50)
    print(f"📷 Image: {found_metadata.get('image_path', 'N/A')}")
    print(f"📄 Page: {found_metadata.get('page_number', 'N/A')}")
    print(f"🎯 Type: {found_metadata.get('region_type', 'N/A')}")
    print(f"\nCurrent Content:\n{found_document[:300]}...")
    print("-" * 50)
    print(f"\n🏷️  Tag to add: \"{tag}\"")
    
    if confirm:
        response = input("\nAdd this tag? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return False
    
    # Update content with tag
    # Format: append [Tag: ...] to the content
    updated_content = f"{found_document}\n\n[Tag]: {tag}"
    
    # Update metadata to track tags
    updated_metadata = found_metadata.copy()
    existing_tags = updated_metadata.get('tags', [])
    if isinstance(existing_tags, str):
        existing_tags = [existing_tags]
    existing_tags.append(tag)
    updated_metadata['tags'] = existing_tags
    updated_metadata['user_modified'] = True
    
    print("\n🔄 Re-vectorizing content...")
    
    # Re-vectorize
    vectorizer = get_vectorizer()
    new_embedding = vectorizer.vectorize(updated_content)[0].tolist()
    
    # Update store
    vector_store.update(
        ids=[found_id],
        embeddings=[new_embedding],
        documents=[updated_content],
        metadatas=[updated_metadata]
    )
    
    print(f"\n✅ Successfully added tag to {found_id}!")
    print(f"   Tag: \"{tag}\"")
    print("   The tag is now searchable.")
    return True


def list_image_tags(image_id: str = None):
    """
    List tags for an image or all tagged images.
    
    Args:
        image_id: Optional specific image ID to show tags for.
    """
    vector_store = get_vector_store()
    
    tagged_items = []
    
    for doc_id, metadata in vector_store._metadata.items():
        if metadata.get('type') == 'extracted_image':
            tags = metadata.get('tags', [])
            if tags:
                tagged_items.append({
                    'id': doc_id,
                    'path': metadata.get('image_path', 'N/A'),
                    'page': metadata.get('page_number', 'N/A'),
                    'tags': tags
                })
    
    if not tagged_items:
        print("No tagged images found.")
        return
    
    print(f"Found {len(tagged_items)} tagged images:\n")
    for item in tagged_items:
        path = Path(item['path']).name if item['path'] != 'N/A' else 'N/A'
        print(f"📷 {path} (Page {item['page']})")
        for tag in item['tags']:
            print(f"   🏷️  {tag}")
        print()
