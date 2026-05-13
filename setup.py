#!/usr/bin/env python3
"""Setup script for evolutionary algorithms project."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🧬 Evolutionary Algorithms for AI - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install black ruff mypy pytest", "Installing development tools"):
        print("⚠️  Development tools installation failed (optional)")
    
    # Create necessary directories
    directories = ['results', 'assets', 'data/raw', 'data/processed', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed (check dependencies)")
    
    # Run example
    if not run_command("python example.py", "Running example"):
        print("❌ Example failed to run")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run comprehensive benchmark:")
    print("   python -m src.train --config configs/default.yaml")
    print("\n2. Launch interactive demo:")
    print("   streamlit run demo/app.py")
    print("\n3. Run quick start script:")
    print("   python scripts/quick_start.py")
    
    print("\n📚 Documentation:")
    print("- README.md: Complete project documentation")
    print("- example.py: Simple usage example")
    print("- demo/app.py: Interactive Streamlit demo")
    
    print("\n⚠️  Safety Notice:")
    print("This project is for research and educational purposes only.")
    print("Not intended for production decisions or critical applications.")

if __name__ == "__main__":
    main()
