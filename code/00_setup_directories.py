"""
00_setup_directories.py
Create all necessary directories for the project
Run this first before any other scripts
"""

import os
import sys

def create_directories():
    """Create all required directories"""
    
    directories = [
        'logs',
        'data/raw',
        'data/processed',
        'models',
        'models/tuned',
        'models/features',
        'reports',
        'reports/figures',
        'reports/mlflow',
        'reports/tuning',
        'mlruns'
    ]
    
    print("="*60)
    print("CREATING PROJECT DIRECTORIES")
    print("="*60)
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")
    
    print("\n✅ All directories created successfully!")
    print("\nProject structure ready:")
    for directory in directories:
        print(f"  📁 {directory}/")

if __name__ == "__main__":
    create_directories()