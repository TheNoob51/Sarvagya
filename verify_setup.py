#!/usr/bin/env python3
"""
Verification script for the Intelligent Research Assistant project setup.
Run this script to verify that the project structure and configuration are correct.
"""

import os
import sys
from pathlib import Path

def check_directory_structure():
    """Check if all required directories exist"""
    required_dirs = [
        "agents",
        "utils", 
        "ui",
        "data"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    return missing_dirs

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        "requirements.txt",
        ".env.template",
        "config.py",
        "utils/config.py",
        "utils/logging_config.py",
        "agents/__init__.py",
        "utils/__init__.py", 
        "ui/__init__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def check_env_file():
    """Check if .env file exists and provide guidance"""
    env_exists = os.path.exists(".env")
    template_exists = os.path.exists(".env.template")
    
    return env_exists, template_exists

def main():
    """Main verification function"""
    print("🔍 Verifying Intelligent Research Assistant project setup...")
    print("=" * 60)
    
    # Check directory structure
    missing_dirs = check_directory_structure()
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_name in missing_dirs:
            print(f"   - {dir_name}")
    else:
        print("✅ All required directories present")
    
    # Check required files
    missing_files = check_required_files()
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    else:
        print("✅ All required files present")
    
    # Check environment configuration
    env_exists, template_exists = check_env_file()
    if not template_exists:
        print("❌ .env.template file missing")
    else:
        print("✅ .env.template file present")
    
    if not env_exists:
        print("⚠️  .env file not found")
        print("   Copy .env.template to .env and configure your API keys")
    else:
        print("✅ .env file present")
    
    # Test configuration loading
    print("\n🧪 Testing configuration loading...")
    try:
        from config import load_config, validate_config
        config = load_config()
        errors = validate_config(config)
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ Configuration loaded and validated successfully")
            
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    total_issues = len(missing_dirs) + len(missing_files)
    if not template_exists:
        total_issues += 1
    
    if total_issues == 0:
        print("🎉 Project setup verification completed successfully!")
        print("   You can now proceed to implement the agents.")
    else:
        print(f"⚠️  Found {total_issues} issues that need to be resolved.")
        print("   Please fix the above issues before proceeding.")
    
    if not env_exists and template_exists:
        print("\n📝 Next steps:")
        print("   1. Copy .env.template to .env")
        print("   2. Configure your API keys in .env")
        print("   3. Run this script again to verify")

if __name__ == "__main__":
    main()