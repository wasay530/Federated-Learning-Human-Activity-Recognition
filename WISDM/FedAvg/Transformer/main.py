import subprocess
import threading
import time
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging to include thread name and time for clarity
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def run_script(script_path: str):
    """Execute a Python script as a subprocess and capture errors."""
    while True:
        process = subprocess.Popen(
            ["python3", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        logging.info(f"Started script process for {script_path}")

        stdout, stderr = process.communicate()

        if stdout:
            logging.info(f"OUTPUT ({script_path}):\n{stdout}")
        if stderr:
            logging.error(f"ERROR ({script_path}):\n{stderr}")

        return_code = process.returncode

        if return_code == 0:
            logging.info(f"{script_path} completed successfully (exit code 0).")
            break
        else:
            logging.error(f"{script_path} crashed (exit code {return_code}). Restarting in 2 seconds...")
            time.sleep(2)

def main():
    # Start the server script
    server_thread = threading.Thread(name="Server", target=run_script, args=("server.py",), daemon=False)
    server_thread.start()
    
    logging.info("Waiting for server to start before launching clients...")
    time.sleep(5)  # Wait a few seconds for the server to initialize
    
    # List of client script files
    client_files = ["c1.py", "c2.py", "c3.py", "c4.py", "c5.py", "c6.py", "c7.py", "c8.py", "c9.py", "c10.py",
                    "c11.py", "c12.py", "c13.py", "c14.py", "c15.py"]
    client_threads = []
    
    # Launch each client in a separate thread
    for idx, client_file in enumerate(client_files, start=1):
        thread_name = f"Client{idx}"
        t = threading.Thread(name=thread_name, target=run_script, args=(client_file,), daemon=False)
        t.start()
        client_threads.append(t)
        logging.info(f"Started {thread_name} for {client_file}")
    
    # Wait for all client threads to complete
    for t in client_threads:
        t.join()
    
    logging.info("All script processes have finished. Main script exiting.")

if __name__ == "__main__":
    main()
