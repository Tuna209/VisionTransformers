"""
Run script for DETR fine-tuning on AU-AIR dataset.
"""

import os
import argparse
import subprocess
import sys

def run_setup():
    """Run setup script."""
    print("Running setup script...")
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "setup.py")])

def run_training(args):
    """Run training script with specified arguments."""
    print("Running training script...")

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "detr_auair.py")]

    if args.pretrained:
        cmd.append("--pretrained")

    if args.skip_train:
        cmd.append("--skip_train")

    if args.skip_eval:
        cmd.append("--skip_eval")

    if args.skip_viz:
        cmd.append("--skip_viz")

    if args.no_wandb:
        cmd.append("--no_wandb")

    if args.num_epochs:
        cmd.extend(["--num_epochs", str(args.num_epochs)])

    subprocess.check_call(cmd)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run DETR fine-tuning on AU-AIR dataset")
    parser.add_argument("--skip_setup", action="store_true", help="Skip setup")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")

    args = parser.parse_args()

    # Run setup
    if not args.skip_setup:
        run_setup()

    # Run training
    run_training(args)

if __name__ == "__main__":
    main()
