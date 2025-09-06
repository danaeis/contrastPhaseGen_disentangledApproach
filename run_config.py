import torch
import subprocess
import time
import os

def get_available_cuda_memory():
    """Returns the available CUDA memory in GB."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return 0
    free_memory, total_memory = torch.cuda.mem_get_info()
    free_memory_gb = free_memory / (1024 ** 3)  # Convert bytes to GB
    return free_memory_gb

def run_training_command():
    """Executes the training command and logs output."""
    command = (
        "python main.py --mode train --encoder medvit "
        "--data_path '../ncct_cect/vindr_ds/original_volumes/Abdomen/raw_image/' "
        "--batch_size 1 --skip_prep --checkpoint_dir checkpoints/medvit "
        "--output_path generated | tee training_medvit.log"
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    return True

def main():
    # Define the minimum memory required (in GB) - adjust based on your model's needs
    min_memory_required = 12.0  # Example: 4GB, adjust this based on your model

    print("Checking CUDA memory availability...")
    
    while True:
        available_memory = get_available_cuda_memory()
        print(f"Available CUDA memory: {available_memory:.2f} GB")
        
        if available_memory >= min_memory_required:
            print(f"Sufficient memory available ({available_memory:.2f} GB). Starting training...")
            success = run_training_command()
            if success:
                break
        else:
            print(f"Insufficient memory ({available_memory:.2f} GB < {min_memory_required} GB). "
                  "Waiting 60 seconds before retrying...")
            time.sleep(60)  # Wait for 60 seconds before checking again

if __name__ == "__main__":
    main()