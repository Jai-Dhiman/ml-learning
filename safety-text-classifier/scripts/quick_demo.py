#!/usr/bin/env python3
"""
Quick Safety Text Classifier Demo

Simple baseline demo using traditional ML while we resolve JAX/Flax compatibility.
"""

import sys
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

def quick_demo():
    """Quick demo of safety classification using baseline model"""
    
    print("🛡️ Safety Text Classifier - Quick Demo")
    print("=" * 45)
    print("📝 Using baseline model (resolving JAX/Flax compatibility)")
    print()
    
    # Quick training data for demo
    train_texts = [
        # Hate speech
        "I hate people from that group", "Those people are terrible", "They should not exist",
        # Self harm  
        "I want to hurt myself", "How to cut safely", "I want to end my life",
        # Dangerous advice
        "How to make bombs", "Poison recipe", "Make weapons at home", 
        # Harassment
        "I will find you", "You will pay for this", "I know where you live",
        # Safe
        "Beautiful weather today", "I love my family", "Cooking is fun", 
        "Reading books", "Exercise is healthy", "Work was good"
    ]
    
    # Labels: [hate_speech, self_harm, dangerous_advice, harassment]
    train_labels = [
        [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],  # hate
        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],  # self harm
        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],  # dangerous
        [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],  # harassment
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],  # safe
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],  # safe
    ]
    
    # Train baseline classifier
    print("🚀 Training baseline classifier...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    classifier = MultiOutputClassifier(LogisticRegression(max_iter=200))
    classifier.fit(X_train, train_labels)
    print("✅ Training completed!")
    print()
    
    # Safety categories
    categories = ['Hate Speech', 'Self-Harm', 'Dangerous Advice', 'Harassment']
    
    # Test cases
    test_cases = [
        "The weather is beautiful today, perfect for a walk.",
        "I hate all people from that group, they are terrible.",
        "Here's how to make dangerous explosives at home.",
        "I'm going to find where you live and hurt you.",
        "I feel sad and want to hurt myself badly.",
        "I love spending time with my family and friends.",
    ]
    
    print("🔍 Testing Safety Classification:")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        # Vectorize and predict
        X_test = vectorizer.transform([text])
        prediction = classifier.predict(X_test)[0]
        
        # Find flagged categories
        flagged = [categories[j] for j, flag in enumerate(prediction) if flag]
        
        if flagged:
            status = f"⚠️  UNSAFE: {', '.join(flagged)}"
        else:
            status = f"✅ SAFE"
        
        print(f"{i}. \"{text[:45]}{'...' if len(text) > 45 else ''}\"")
        print(f"   → {status}")
        print()
    
    print("=" * 50)
    print("🎉 Quick demo completed!")
    print("")
    print("📊 This is a BASELINE demonstration")
    print("   Full transformer model pending JAX/Flax fix")
    print("")
    print("🚀 Next steps:")
    print("   • Fix JAX/Flax compatibility")
    print("   • Train full transformer model") 
    print("   • Achieve 85%+ accuracy target")

if __name__ == "__main__":
    quick_demo()