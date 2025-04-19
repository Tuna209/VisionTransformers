"""
Setup script to install dependencies and prepare the environment.
"""

import os
import subprocess
import sys
import json


def install_dependencies():
    """
    Install required dependencies.
    """
    print("Installing dependencies...")

    # List of required packages
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "matplotlib",
        "tqdm",
        "wandb"
    ]

    # Install packages
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("Dependencies installed successfully!")


def create_directories():
    """
    Create necessary directories.
    """
    print("Creating directories...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("report", exist_ok=True)
    print("Directories created successfully!")


def setup_wandb():
    """
    Set up Weights & Biases.
    """
    print("Setting up Weights & Biases...")

    try:
        import wandb

        # Set API key
        api_key = "d52af6d683efe0a6949f6523a349f77d469f9c90"

        # Login to wandb
        wandb.login(key=api_key)

        print("Successfully logged in to Weights & Biases!")
    except Exception as e:
        print(f"Error setting up Weights & Biases: {e}")


def main():
    """
    Main setup function.
    """
    print("Setting up the environment...")

    # Install dependencies
    install_dependencies()

    # Create directories
    create_directories()

    # Setup Weights & Biases
    setup_wandb()

    print("Setup completed successfully!")


if __name__ == "__main__":
    main()
