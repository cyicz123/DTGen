#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Filter images based on CLIP similarity with generated text from filename."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a directory containing image files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./unmatched",
        help="Directory to move images with low similarity scores.",
    )
    parser.add_argument(
        "-k",
        "--std-dev-factor",
        type=float,
        default=1.5,
        help="The factor for standard deviation to set the adaptive threshold (mean - k * std).",
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

    # Find all image files
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            image_files = [input_path]
        else:
            print(f"Error: Input file {input_path} is not a supported image file.")
            return
    elif input_path.is_dir():
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.rglob(f"*{ext}")))
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")
        return

    if not image_files:
        print(f"No image files found in {input_path}.")
        return

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print(f"Found {len(image_files)} image files to process.")

    dirtiness_levels = ["clean", "heavily_dirty", "moderately_dirty", "slightly_dirty"]

    # Data structure to hold scores and paths for each level
    level_data = {level: [] for level in dirtiness_levels}

    print("--- Pass 1: Calculating similarity scores ---")
    for image_file in image_files:
        try:
            # Extract dirtiness level from filename
            level_found = None
            for level in dirtiness_levels:
                if level in image_file.name:
                    level_found = level
                    break

            if not level_found:
                print(
                    f"Warning: No dirtiness level found in filename {image_file.name}. Skipping."
                )
                continue

            prompt = f"a photo of a {level_found} of tableware"

            # Open image
            image = Image.open(image_file).convert("RGB")

            # Process with CLIP
            inputs = processor(
                text=[prompt], images=image, return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            # The model outputs normalized embeddings, so a simple dot product gives the cosine similarity.
            similarity = (outputs.image_embeds @ outputs.text_embeds.T).item()

            print(f"Processing {image_file.name}: Similarity = {similarity:.4f}")

            level_data[level_found].append({"path": image_file, "score": similarity})

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print("\n--- Pass 2: Calculating thresholds and filtering images ---")

    for level, data in level_data.items():
        if not data:
            print(f"No images found for level '{level}'. Skipping.")
            continue

        scores = [item["score"] for item in data]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        threshold = mean_score - args.std_dev_factor * std_score

        print(f"\nStats for level '{level}':")
        print(f"  - Images: {len(scores)}")
        print(f"  - Mean Similarity: {mean_score:.4f}")
        print(f"  - Std Dev: {std_score:.4f}")
        print(
            f"  - Adaptive Threshold (mean - {args.std_dev_factor}*std): {threshold:.4f}"
        )

        # Filter and move
        moved_count = 0
        for item in data:
            if item["score"] < threshold:
                image_file = item["path"]
                print(
                    f"  -> Similarity {item['score']:.4f} < {threshold:.4f}. Moving {image_file.name} to {output_dir}"
                )
                shutil.move(str(image_file), str(output_dir / image_file.name))
                moved_count += 1
        print(f"  -> Moved {moved_count} images for level '{level}'.")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
