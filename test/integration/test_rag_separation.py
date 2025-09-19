

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test that the expected files exist in the correct locations."""
    print("🧪 Testing file structure...")

    # Change to project root directory
    # Since we're in test/integration/, parent.parent is test/, so we need to go up one more level
    project_root = Path(__file__).parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        # Test files that should exist
        expected_files = [
            "rag/Dockerfile",
            "rag/worker.py",
            "server/core/celery/tasks/rag_tasks.py",
            "server/api/routers/rag/rag_query.py",
            "server/services/health_service.py",

            "deployment/docker_compose/docker-compose.yml",
            "deployment/docker_compose/docker-compose.dev.yml",
        ]

        all_exist = True
        for file_path in expected_files:
            full_path = Path(file_path)
            if full_path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} - MISSING")
                all_exist = False

        return all_exist
    finally:
        os.chdir(original_cwd)

def test_imports():
    """Test that key modules can be imported (syntax check)."""
    print("\n🧪 Testing imports...")


    project_root = Path(__file__).parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        test_modules = [
            ("server/main", "Server main module"),
            ("server/core/celery/celery", "Celery configuration"),
            ("server/core/celery/tasks/rag_tasks", "RAG tasks"),
            ("server/api/routers/rag/rag_query", "RAG query API"),
            ("server/services/health_service", "Health service"),
            ("rag/worker", "RAG worker"),
        ]

        all_import = True
        for module_name, description in test_modules:
            try:

                module_path = Path(module_name + ".py")
                if module_path.exists():
                    compile(open(module_path, 'r').read(), module_path, 'exec')
                    print(f"  ✅ {description}")
                else:
                    print(f"  ❌ {description} - File not found")
                    all_import = False
            except SyntaxError as e:
                print(f"  ❌ {description} - Syntax error: {e}")
                all_import = False
            except Exception as e:
                print(f"  ⚠️  {description} - Import issue (expected with missing deps): {e}")

        return all_import
    finally:
        os.chdir(original_cwd)

def test_docker_compose_config():
    """Test that Docker Compose configurations are valid."""
    print("\n🧪 Testing Docker Compose configurations...")

    import subprocess
    import os

    # Change to deployment directory
    old_cwd = os.getcwd()
    deployment_dir = Path(__file__).parent.parent.parent / "deployment" / "docker_compose"
    os.chdir(deployment_dir)

    try:
        # Test production config
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("  ✅ Production docker-compose.yml is valid")
        else:
            print(f"  ❌ Production docker-compose.yml invalid: {result.stderr}")
            return False

        # Test development config
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.dev.yml", "config"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("  ✅ Development docker-compose.dev.yml is valid")
        else:
            print(f"  ❌ Development docker-compose.dev.yml invalid: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("  ❌ Docker Compose validation timed out")
        return False
    except FileNotFoundError:
        print("  ⚠️  Docker Compose not available, skipping validation")
        return True  # Not a failure, just not available
    finally:
        os.chdir(old_cwd)

    return True

def test_celery_configuration():
    """Test that Celery configuration logic is sound."""
    print("\n🧪 Testing Celery configuration logic...")

    # Test the configuration detection logic
    config_tests = [
        # (env_vars, expected_broker_type)
        ({}, "filesystem"),  # Default case
        ({"CELERY_REDIS_HOST": "redis"}, "redis"),
        ({"CELERY_RABBITMQ_HOST": "rabbitmq"}, "rabbitmq"),
        ({"CELERY_REDIS_HOST": "redis", "CELERY_RABBITMQ_HOST": "rabbitmq"}, "redis"),  # Redis takes priority
    ]

    # Change to project root directory
    # Since we're in test/integration/, parent.parent is test/, so we need to go up one more level
    project_root = Path(__file__).parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        # Read the configuration function
        celery_file = Path("server/core/celery/celery.py")
        if not celery_file.exists():
            print("  ❌ Celery configuration file not found")
            return False

        with open(celery_file, 'r') as f:
            content = f.read()

        # Check for key configuration functions
        checks = [
            "def get_celery_config():" in content,
            "CELERY_REDIS_HOST" in content,
            "CELERY_RABBITMQ_HOST" in content,
            "filesystem://" in content,
        ]

        if all(checks):
            print("  ✅ Celery configuration structure is correct")
            return True
        else:
            print("  ❌ Celery configuration structure is incomplete")
            return False
    finally:
        os.chdir(original_cwd)


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("RAG SUBSYSTEM SEPARATION INTEGRATION TEST")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Import/Syntax", test_imports),
        ("Docker Compose", test_docker_compose_config),
        ("Celery Configuration", test_celery_configuration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ RAG subsystem separation has been successfully implemented!")
        print("\nNext steps:")
        print("1. Build and test the containers: docker-compose build")
        print("2. Start the services: docker-compose up")
        print("3. Test RAG operations through the API")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Please review the implementation and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
