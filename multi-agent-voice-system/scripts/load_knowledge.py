#!/usr/bin/env python3
"""
Load documents into the RAG knowledge base
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.knowledge_base import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(
        description="Load documents into the knowledge base"
    )
    parser.add_argument(
        '--agent',
        type=str,
        choices=['sales', 'research'],
        required=True,
        help='Agent type (sales or research)'
    )
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Directory containing text files to load'
    )
    parser.add_argument(
        '--initialize-sample',
        action='store_true',
        help='Initialize with sample data instead'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Knowledge Base Loader")
    print("=" * 60)
    print()
    
    # Initialize knowledge base
    print("📚 Initializing knowledge base...")
    kb = KnowledgeBase()
    print()
    
    if args.initialize_sample:
        print("🔧 Loading sample knowledge...")
        kb.initialize_with_sample_data()
    else:
        print(f"📂 Loading documents from: {args.directory}")
        print(f"🎯 Agent type: {args.agent}")
        print()
        
        # Check if directory exists
        data_path = Path(args.directory)
        if not data_path.exists():
            print(f"❌ Directory not found: {args.directory}")
            return
        
        # Count files
        txt_files = list(data_path.glob("*.txt"))
        if not txt_files:
            print(f"⚠️  No .txt files found in {args.directory}")
            return
        
        print(f"Found {len(txt_files)} text files")
        for file in txt_files:
            print(f"  - {file.name}")
        print()
        
        # Load documents
        kb.load_from_files(args.directory, args.agent)
    
    print()
    print("=" * 60)
    print("✅ Knowledge base updated successfully!")
    print("=" * 60)
    print()
    print("You can now run the main application:")
    print("  python main.py")


if __name__ == "__main__":
    main()
