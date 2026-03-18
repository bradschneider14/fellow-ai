import subprocess
import sys
import os

def test_build():
    """
    Tests that 'python -m build' works and generates a source distribution and a wheel.
    """
    # Assuming we are running from the root of the project
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    
    assert result.returncode == 0, f"Build failed with return code {result.returncode}"
    
    # Check for output directories
    assert os.path.exists("dist"), "dist directory was not created"
    
    files = os.listdir("dist")
    assert any(f.endswith(".tar.gz") for f in files), "No source distribution found in dist/"
    assert any(f.endswith(".whl") for f in files), "No wheel found in dist/"

if __name__ == "__main__":
    test_build()
    print("Build test passed!")
