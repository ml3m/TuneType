#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, log_prefix="", shell=False):
    """Run a command and log its output in real-time"""
    logger.info(f"Running command: {' '.join(command)}")
    
    try:
        # Start the process
        if shell:
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                shell=True
            )
        else:
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # Read and log output in real-time
        for line in process.stdout:
            logger.info(f"{log_prefix}{line.strip()}")
        
        # Wait for process to complete
        process.wait()
        
        # Check return code
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False

def main():
    """Main function to orchestrate training and calibration"""
    parser = argparse.ArgumentParser(description='Train and calibrate genre classification model')
    parser.add_argument('--sample-dir', type=str, default='samples', 
                        help='Directory containing genre samples')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip the training step, only run calibration')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip the calibration step, only run training')
    parser.add_argument('--intensive-training', action='store_true',
                        help='Run an intensive training process with more iterations')
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable data augmentation for training')
    
    args = parser.parse_args()
    
    # Get the Python interpreter from the virtual environment if available
    python_path = sys.executable
    logger.info(f"Using Python interpreter: {python_path}")
    
    # Initialize success status
    training_success = True
    calibration_success = True
    
    # Step 1: Train the model
    if not args.skip_training:
        logger.info("=== Step 1: Training the genre classification model ===")
        
        # Set up command for training
        train_cmd = [
            python_path, "train_with_samples.py",
            "--sample-dir", args.sample_dir
        ]
        
        # Add options for intensive training if requested
        if args.intensive_training:
            logger.info("Running intensive training with extended parameters")
            train_cmd.extend([
                "--force-recompute",  # Recompute features for better quality
                "--cross-validation"  # Enable cross-validation for better model evaluation
            ])
        
        # Add data augmentation if requested
        if not args.augmentation:
            train_cmd.append("--no-augmentation")
        
        # Run the training command
        start_time = time.time()
        training_success = run_command(train_cmd)
        training_time = time.time() - start_time
        
        if training_success:
            logger.info(f"Training completed successfully in {training_time:.2f} seconds!")
        else:
            logger.error("Training failed! Check the logs for details.")
            if not args.skip_calibration:
                logger.warning("Skipping calibration step due to training failure.")
                return
    
    # Step 2: Calibrate the model
    if not args.skip_calibration and training_success:
        logger.info("=== Step 2: Calibrating the genre classification model ===")
        
        # Set up command for calibration
        calibrate_cmd = [
            python_path, "tune_genre_bias.py",
            "--sample-dir", args.sample_dir,
            "--force"  # Always recalibrate after training
        ]
        
        # Run the calibration command
        start_time = time.time()
        calibration_success = run_command(calibrate_cmd)
        calibration_time = time.time() - start_time
        
        if calibration_success:
            logger.info(f"Calibration completed successfully in {calibration_time:.2f} seconds!")
        else:
            logger.error("Calibration failed! Check the logs for details.")
    
    # Report overall status
    if (not args.skip_training and training_success) and (not args.skip_calibration and calibration_success):
        logger.info("=== All requested steps completed successfully! ===")
        logger.info("The model is now ready for use.")
        logger.info("Run 'python app.py' to start the web application.")
    else:
        logger.error("=== Process completed with errors! ===")
        logger.error("Check the logs for details.")

if __name__ == "__main__":
    main() 