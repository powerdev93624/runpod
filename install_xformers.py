import os
import sys
import subprocess
import platform

def main():
    """
    Helper script to install xformers based on the system configuration
    """
    print("Installing xformers for better performance...")
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. xformers with CUDA acceleration cannot be installed.")
            print("You can still run the model without xformers, but it will be slower.")
            return
        
        cuda_version = torch.version.cuda
        if cuda_version is None:
            print("PyTorch was not built with CUDA support. xformers with CUDA acceleration cannot be installed.")
            print("You can still run the model without xformers, but it will be slower.")
            return
            
        print(f"Detected CUDA version: {cuda_version}")
        
        # Get PyTorch version
        torch_version = torch.__version__.split('+')[0]
        print(f"Detected PyTorch version: {torch_version}")
        
        # Get Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"Detected Python version: {python_version}")
        
        # Install appropriate xformers version
        if platform.system() == "Windows":
            # For Windows, use pip to install pre-built wheels
            print("Installing xformers on Windows...")
            cmd = [sys.executable, "-m", "pip", "install", "xformers"]
            subprocess.run(cmd, check=True)
        else:
            # For Linux, we can be more specific with the version
            print("Installing xformers on Linux...")
            cmd = [sys.executable, "-m", "pip", "install", "xformers==0.0.21"]
            subprocess.run(cmd, check=True)
            
        print("xformers installation completed. Please restart the API server.")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("PyTorch is not installed. Please install PyTorch first.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        print("Please try installing xformers manually with:")
        print("pip install xformers")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 