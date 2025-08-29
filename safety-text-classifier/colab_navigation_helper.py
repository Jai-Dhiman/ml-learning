"""
Colab Navigation Helper for ml-learning Repository

This script helps users navigate to the correct directory structure
when working with the ml-learning repository in Google Colab.
"""

import os
from pathlib import Path


def find_and_navigate_to_project():
    """Find and navigate to the safety-text-classifier project directory."""
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Check if we're already in the right place
    if current_dir.name == "safety-text-classifier":
        if (current_dir / "src").exists() and (current_dir / "configs").exists():
            print("✅ Already in safety-text-classifier directory!")
            return str(current_dir)
    
    # Look for safety-text-classifier in current directory
    safety_path = current_dir / "safety-text-classifier"
    if safety_path.exists() and safety_path.is_dir():
        print(f"✅ Found safety-text-classifier at: {safety_path}")
        os.chdir(safety_path)
        print(f"📂 Changed to: {Path.cwd()}")
        return str(safety_path)
    
    # Look for ml-learning directory and navigate to safety-text-classifier
    ml_learning_path = current_dir / "ml-learning"
    if ml_learning_path.exists():
        safety_in_ml = ml_learning_path / "safety-text-classifier"
        if safety_in_ml.exists():
            print(f"✅ Found project at: {safety_in_ml}")
            os.chdir(safety_in_ml)
            print(f"📂 Changed to: {Path.cwd()}")
            return str(safety_in_ml)
    
    # Check if we're inside ml-learning already
    if "ml-learning" in str(current_dir):
        # Find the ml-learning root
        parts = current_dir.parts
        ml_index = None
        for i, part in enumerate(parts):
            if part == "ml-learning":
                ml_index = i
                break
        
        if ml_index is not None:
            ml_root = Path(*parts[:ml_index + 1])
            safety_path = ml_root / "safety-text-classifier"
            if safety_path.exists():
                print(f"✅ Found project at: {safety_path}")
                os.chdir(safety_path)
                print(f"📂 Changed to: {Path.cwd()}")
                return str(safety_path)
    
    print("❌ Could not find safety-text-classifier directory!")
    print("📋 Available directories:")
    for item in current_dir.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}")
    
    print("\n💡 Please ensure you've uploaded the ml-learning repository or navigate to the correct directory manually.")
    return None


def verify_project_structure():
    """Verify that we're in the correct project directory with all required files."""
    current_dir = Path.cwd()
    required_items = {
        "src": "directory",
        "configs": "directory", 
        "requirements.txt": "file",
        "requirements-colab.txt": "file"
    }
    
    print(f"\n🔍 Verifying project structure at: {current_dir}")
    
    all_good = True
    for item, item_type in required_items.items():
        item_path = current_dir / item
        if item_path.exists():
            if item_type == "directory" and item_path.is_dir():
                print(f"✅ {item}/ (directory)")
            elif item_type == "file" and item_path.is_file():
                print(f"✅ {item} (file)")
            else:
                print(f"❌ {item} (wrong type)")
                all_good = False
        else:
            if item == "requirements-colab.txt":
                print(f"⚠️  {item} (missing - will be created automatically)")
            else:
                print(f"❌ {item} (missing)")
                all_good = False
    
    return all_good


if __name__ == "__main__":
    print("🚀 ml-learning Colab Navigation Helper")
    print("=" * 50)
    
    # Find and navigate to project
    project_path = find_and_navigate_to_project()
    
    if project_path:
        # Verify structure
        structure_ok = verify_project_structure()
        
        if structure_ok:
            print("\n🎉 Project setup looks good!")
            print("🔄 You can now run the setup script.")
        else:
            print("\n⚠️  Some required files are missing.")
            print("Please check your project upload.")
    
    print(f"\n📍 Final directory: {Path.cwd()}")
