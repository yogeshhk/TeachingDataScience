#!/usr/bin/env python3
"""
Setup script to resolve dependency conflicts and install required packages.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Book QA System dependencies...")
    
    # Step 1: Uninstall problematic packages
    print("\n📦 Cleaning up existing packages...")
    packages_to_remove = [
        "bottleneck",
        "pandas",
        "numpy",
        "docling",
        "docling-core"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"Removing {package}")
    
    # Step 2: Install numpy first with specific version
    if not run_command("pip install 'numpy>=1.21.0,<1.25.0'", "Installing compatible numpy"):
        print("❌ Failed to install numpy. Exiting.")
        sys.exit(1)
    
    # Step 3: Install pandas with specific version
    if not run_command("pip install 'pandas>=2.0.0,<2.2.0'", "Installing compatible pandas"):
        print("❌ Failed to install pandas. Exiting.")
        sys.exit(1)
    
    # Step 4: Install bottleneck
    if not run_command("pip install 'bottleneck>=1.3.7'", "Installing bottleneck"):
        print("❌ Failed to install bottleneck. Exiting.")
        sys.exit(1)
    
    # Step 5: Install docling
    if not run_command("pip install 'docling>=1.0.0'", "Installing docling"):
        print("❌ Failed to install docling. Exiting.")
        sys.exit(1)
    
    # Step 6: Install remaining requirements
    if not run_command("pip install -r requirements.txt", "Installing remaining requirements"):
        print("❌ Failed to install requirements. Exiting.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("1. Make sure your local LLM server is running at http://localhost:1234")
    print("2. Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
