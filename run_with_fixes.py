import os
import sys
import subprocess
import time

def check_xformers():
    """Check if xformers is installed"""
    try:
        import xformers
        return True
    except ImportError:
        return False

def run_server():
    """Run the API server with proper error handling"""
    try:
        # Create necessary directories
        os.makedirs("output", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        
        # Check for xformers
        if not check_xformers():
            print("xformers is not installed. For better performance, you can install it by running:")
            print("python install_xformers.py")
            print("Continuing without xformers...")
            time.sleep(2)
        
        # Try to run the server using the standard app.py
        print("Starting Realistic Vision API server...")
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running the server: {e}")
        print("\nThere might be issues with loading the model or dependencies.")
        print("Here are some troubleshooting steps:")
        
        print("\n1. Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        
        print("\n2. Try installing xformers for better performance:")
        print("   python install_xformers.py")
        
        print("\n3. If you're having model loading issues, try using a different model version:")
        print("   - Edit image_generator.py")
        print("   - Change model_id to 'SG161222/Realistic_Vision_V5.1'")
        
        print("\n4. If you're still having issues, try running with debug logs:")
        print("   PYTHONPATH='.' HF_HUB_ENABLE_HF_TRANSFER=1 python app.py")
        
        print("\nWould you like to try running with an alternative model? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            # Try running with the fallback model script
            print("Attempting to run with fallback model (V5.1)...")
            try:
                # Create a temporary file with the modified model ID
                with open("temp_generator.py", "w") as f:
                    with open("image_generator.py", "r") as original:
                        content = original.read()
                        modified = content.replace(
                            "SG161222/Realistic_Vision_V6.0_B1_noVAE", 
                            "SG161222/Realistic_Vision_V5.1"
                        )
                        f.write(modified)
                
                # Run with the modified generator
                os.environ["GENERATOR_MODULE"] = "temp_generator"
                subprocess.run([sys.executable, "app.py"], check=True)
                
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                print("Please check the error messages and try the troubleshooting steps above.")
    
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_server() 