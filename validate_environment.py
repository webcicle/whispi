#!/usr/bin/env python3
"""
Pi-Whispr Environment Validation
Quick check to ensure the testing environment is properly set up
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. Found: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_required_modules():
    """Check if required modules are available"""
    print("\n📦 Checking required modules...")
    
    required_modules = [
        'websockets',
        'docker',
        'asyncio',
        'json',
        'logging'
    ]
    
    optional_modules = [
        'pyaudio',
        'numpy', 
        'pynput',
        'pytest'
    ]
    
    all_good = True
    
    # Check required modules
    for module in required_modules:
        if importlib.util.find_spec(module) is not None:
            print(f"✅ {module}")
        else:
            print(f"❌ {module} (required)")
            all_good = False
    
    # Check optional modules (for client testing)
    print("\n📦 Checking optional modules (for client testing)...")
    optional_available = []
    
    for module in optional_modules:
        if importlib.util.find_spec(module) is not None:
            print(f"✅ {module}")
            optional_available.append(module)
        else:
            print(f"⚠️ {module} (optional - needed for client audio testing)")
    
    return all_good, optional_available

def check_docker():
    """Check if Docker is available and running"""
    print("\n🐳 Checking Docker...")
    
    try:
        # Check if docker command exists
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
        else:
            print("❌ Docker command failed")
            return False
            
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
        else:
            print("❌ Docker daemon not running")
            return False
            
        return True
        
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Docker command timed out")
        return False
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    print("\n🐚 Checking Docker Compose...")
    
    # Try docker-compose first
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ docker-compose: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    # Try docker compose (newer syntax)
    try:
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ docker compose: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Docker Compose not found")
    return False

def check_project_files():
    """Check if required project files exist"""
    print("\n📁 Checking project files...")
    
    required_files = [
        'docker-compose.yml',
        'test_connection.py',
        'client_runner.py', 
        'test_environment.py',
        'requirements_test.txt'
    ]
    
    all_present = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_present = False
    
    return all_present

def main():
    """Main validation function"""
    print("🧪 Pi-Whispr Environment Validation")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Modules", lambda: check_required_modules()[0]),
        ("Docker", check_docker),
        ("Docker Compose", check_docker_compose),
        ("Project Files", check_project_files)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Check optional modules separately for reporting
    try:
        _, optional_available = check_required_modules()
    except:
        optional_available = []
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 25)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Environment is ready for testing!")
        
        if 'pyaudio' in optional_available:
            print("🎤 Audio modules available - can test with real audio")
        else:
            print("⚠️ Audio modules missing - use mock mode for client testing")
            
        print("\nNext steps:")
        print("1. Run: ./run_tests.sh")
        print("2. Or run specific tests manually")
        
        return 0
    else:
        print("\n❌ Environment setup incomplete")
        print("\nTo fix issues:")
        print("1. Install missing Python modules: pip install -r requirements_test.txt")
        print("2. Install/start Docker if needed")
        print("3. Ensure you're in the project root directory")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 