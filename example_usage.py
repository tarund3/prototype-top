#!/usr/bin/env python3
"""
Example usage of the TOP implementation.
This script demonstrates how to run different experiments.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False


def main():
    """Run example experiments."""
    print("TOKEN ORDER PREDICTION - EXAMPLE USAGE")
    print("="*60)
    
    # Set environment variable to avoid OpenMP issues
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Example 1: Train NTP baseline (quick test)
    print("\n1. Training NTP baseline (2 epochs, small model)...")
    cmd1 = "python main.py --objectives ntp --n_layer 2 --n_head 4 --d_model 128 --max_epochs 2 --batch_size 4 --seq_len 64 --output_dir outputs/ntp_baseline"
    success1 = run_command(cmd1, "NTP Baseline Training")
    
    # Example 2: Train NTP + TOP (quick test)
    print("\n2. Training NTP + TOP (2 epochs, small model)...")
    cmd2 = "python main.py --objectives ntp top --n_layer 2 --n_head 4 --d_model 128 --max_epochs 2 --batch_size 4 --seq_len 64 --lambda_top 0.5 --output_dir outputs/ntp_top"
    success2 = run_command(cmd2, "NTP + TOP Training")
    
    # Example 3: Compare results
    if success1 and success2:
        print("\n3. Comparing experiments...")
        cmd3 = "python main.py --mode compare --experiment_dirs outputs/ntp_baseline outputs/ntp_top"
        run_command(cmd3, "Experiment Comparison")
    
    # Example 4: Show help
    print("\n4. Available options...")
    cmd4 = "python main.py --help"
    run_command(cmd4, "Help Information")
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE COMPLETED")
    print("="*60)
    
    if success1 and success2:
        print("\n🎉 Examples completed successfully!")
        print("\nNext steps:")
        print("1. Run longer experiments with more epochs")
        print("2. Try different objective combinations")
        print("3. Experiment with different hyperparameters")
        print("4. Scale up to larger models and datasets")
    else:
        print("\n❌ Some examples failed. Check the error messages above.")


if __name__ == '__main__':
    main()
