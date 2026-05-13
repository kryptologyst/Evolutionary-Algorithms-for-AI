"""Scripts for running experiments and demos."""

#!/usr/bin/env python3
"""Quick start script for evolutionary algorithms demo."""

import sys
import subprocess
from pathlib import Path

def main():
    """Main function for quick start."""
    print("🧬 Evolutionary Algorithms for AI - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("Choose an option:")
    print("1. Run comprehensive benchmark")
    print("2. Launch interactive demo")
    print("3. Run tests")
    print("4. Install dependencies")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Running comprehensive benchmark...")
        subprocess.run([
            sys.executable, "-m", "src.train", 
            "--config", "configs/default.yaml",
            "--output-dir", "results"
        ])
        
    elif choice == "2":
        print("\n🌐 Launching interactive demo...")
        print("The demo will open in your browser at http://localhost:8501")
        subprocess.run(["streamlit", "run", "demo/app.py"])
        
    elif choice == "3":
        print("\n🧪 Running tests...")
        subprocess.run(["pytest", "tests/", "-v"])
        
    elif choice == "4":
        print("\n📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
