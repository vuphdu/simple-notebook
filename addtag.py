#!/usr/bin/env python
"""
Shortcut command to add tags to images.

Usage:
    python addtag.py HFP_v1.8_p92_crop_1 "Established SCO connection"
    python addtag.py HFP_v1.8_p14_crop_1 "Signaling diagram conventions" -y
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.image_tag import add_tag, list_image_tags


def main():
    parser = argparse.ArgumentParser(
        description="Add tags to images in the RAG database",
        usage="python addtag.py IMAGE_ID \"TAG\" [-y]"
    )
    
    parser.add_argument(
        "image_id",
        type=str,
        nargs="?",
        help="Image ID from search results (e.g., 'HFP_v1.8_p14_crop_1')"
    )
    parser.add_argument(
        "tag",
        type=str,
        nargs="?",
        help="Tag/context to add (e.g., 'Established SCO connection')"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all tagged images"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_image_tags()
        return
    
    if not args.image_id or not args.tag:
        parser.print_help()
        print("\n" + "=" * 50)
        print("Examples:")
        print('  python addtag.py HFP_v1.8_p92_crop_1 "Established SCO connection"')
        print('  python addtag.py HFP_v1.8_p14_crop_1 "Signaling conventions" -y')
        print("  python addtag.py --list")
        return
    
    add_tag(
        image_id=args.image_id,
        tag=args.tag,
        confirm=not args.yes
    )


if __name__ == "__main__":
    main()
