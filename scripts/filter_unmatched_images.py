#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import shutil

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def find_image_path(text_path: Path):
    """Find the corresponding image file for a given text file."""
    base_name = text_path.stem
    parent_dir = text_path.parent
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        for f in parent_dir.glob(f"{base_name}*{ext}"):
            return f
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter images based on CLIP similarity with corresponding text."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to a text file or a directory containing text files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./unmatched",
        help="Directory to move images with low similarity scores.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold. Images below this will be moved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Name of the CLIP model to use from Hugging Face.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all text files
    if input_path.is_file():
        text_files = [input_path]
    elif input_path.is_dir():
        text_files = list(input_path.rglob("*.txt"))
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")
        return

    if not text_files:
        print(f"No .txt files found in {input_path}.")
        return

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    print(f"Found {len(text_files)} text files to process.")

    for text_file in text_files:
        try:
            # Find corresponding image
            image_file = find_image_path(text_file)
            if not image_file:
                print(f"Warning: No corresponding image found for {text_file}")
                continue

            # Read text prompt
            prompt = text_file.read_text(encoding="utf-8").strip()
            if not prompt:
                print(f"Warning: Text file {text_file} is empty.")
                continue

            # Open image
            image = Image.open(image_file).convert("RGB")

            # Process with CLIP
            inputs = processor(
                text=[prompt], images=image, return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # The model outputs normalized embeddings, so a simple dot product gives the cosine similarity.
            similarity = (image_embeds @ text_embeds.T).item()

            print(
                f"Processing {image_file.name} and {text_file.name}: Similarity = {similarity:.4f}"
            )

            # Filter and move
            if similarity < args.threshold:
                print(
                    f"  -> Similarity {similarity:.4f} < {args.threshold}. Moving {image_file.name} to {output_dir}"
                )
                shutil.move(str(image_file), str(output_dir / image_file.name))

        except Exception as e:
            print(f"Error processing {text_file}: {e}")

    print("Processing complete.")


if __name__ == "__main__":
    main()
