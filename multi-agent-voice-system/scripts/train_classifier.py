#!/usr/bin/env python3
"""
Train the intent classification model with custom data
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.router_agent import IntentClassifier, SAMPLE_TRAINING_DATA


def load_training_data(filepath: str) -> list:
    """Load training data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def validate_data(data: list) -> bool:
    """Validate training data format"""
    required_keys = {'text', 'label'}
    
    for item in data:
        if not isinstance(item, dict):
            print(f"❌ Invalid item: {item}")
            return False
        
        if not required_keys.issubset(item.keys()):
            print(f"❌ Missing keys in item: {item}")
            return False
        
        if item['label'] not in ['sales', 'research', 'general']:
            print(f"❌ Invalid label: {item['label']}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train the intent classification model"
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data JSON file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/intent_classifier.pt',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Use sample training data'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Intent Classifier Training")
    print("=" * 60)
    print()
    
    # Load training data
    if args.use_sample or not args.data:
        print("📊 Using sample training data")
        training_data = SAMPLE_TRAINING_DATA
    else:
        print(f"📊 Loading training data from {args.data}")
        training_data = load_training_data(args.data)
    
    # Validate data
    print(f"✓ Loaded {len(training_data)} training examples")
    if not validate_data(training_data):
        print("❌ Invalid training data format")
        return
    
    print("✓ Training data validated")
    print()
    
    # Count examples per class
    label_counts = {}
    for item in training_data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Class distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} examples")
    print()
    
    # Initialize classifier
    print("🤖 Initializing intent classifier...")
    classifier = IntentClassifier()
    
    # Train the model
    print(f"🎯 Training for {args.epochs} epochs with lr={args.lr}")
    print()
    
    classifier.train(
        training_data=training_data,
        epochs=args.epochs,
        lr=args.lr
    )
    
    print()
    
    # Save the model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    classifier.save_model(str(output_path))
    print(f"✅ Model saved to {output_path}")
    print()
    
    # Test the model
    print("Testing model on sample inputs:")
    print("-" * 60)
    
    test_examples = [
        "How much does the pro plan cost?",
        "What do users think about feature X?",
        "Hello, what can you do?",
    ]
    
    import asyncio
    
    async def test_classifier():
        for example in test_examples:
            intent = await classifier.classify(example)
            print(f"Input: {example}")
            print(f"Predicted intent: {intent}")
            print()
    
    asyncio.run(test_classifier())
    
    print("=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
