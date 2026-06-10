#!/usr/bin/env python3
"""
Dataset Image Downloader
A command-line interface for downloading images from various datasets.
"""

import argparse
import io
import os
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import open_clip
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from PIL import Image


class DatasetDownloader:
    """Main class for downloading images from different dataset sources."""
    
    def __init__(self):
        self.fs = HfFileSystem()
        
    def download_derm1m_images(
        self, 
        output_dir: str = "derm_train_images",
        pubmed_files: Optional[List[str]] = None,
        iiyi_files: Optional[List[str]] = None,
        youtube_files: Optional[List[str]] = None
    ) -> Dict[str, Image.Image]:
        """
        Download specific images from the Derm1M dataset.
        
        Args:
            output_dir: Directory to save images
            pubmed_files: List of PubMed filenames to download
            iiyi_files: List of IIYI filenames to download
            youtube_files: List of YouTube filenames to download
            
        Returns:
            Dictionary mapping filenames to PIL Images
        """
        # Default files if none specified
        if pubmed_files is None:
            pubmed_files = [
                "98_6c_PMC5383004_pone.0172624.g006_3.jpg",
                "00_39_PMC10381143_jimaging-09-00148-g011_1.jpg",
                "0d_59_PMC4458964_IJD_60_321e_g003_0.png",
                "e2_67_PMC3445843_1471_5945_12_7_7.png",
                "89_38_PMC3350198_CRIM.PEDIATRICS2012_152602.002_1.png",
                "0b_33_PMC3804142_CRIM.DENTISTRY2013_672383.001_0.png"
            ]
            
        if iiyi_files is None:
            iiyi_files = ["2281_1.png", "14083_1.png"]
            
        if youtube_files is None:
            youtube_files = ["OzBKmWt9zWo_frame_8071_0_0.jpg"]

        repo_id = "datasets/redlessone/Derm1M"
        downloaded_images = {}

        print("Fetching individual images via streamed remote zipfile...")

        download_tasks = {
            "pubmed": pubmed_files,
            "IIYI": iiyi_files,
            "youtube": youtube_files
        }

        for category, filenames in download_tasks.items():
            if not filenames:  # Skip if empty list
                continue
                
            zip_name = f"{category}.zip"
            remote_zip_path = f"{repo_id}/{zip_name}"
            
            try:
                with self.fs.open(remote_zip_path, "rb") as remote_file:
                    with zipfile.ZipFile(remote_file) as zf:
                        for filename in filenames:
                            try:
                                with zf.open(filename) as img_file:
                                    img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
                                    downloaded_images[filename] = img
                                    
                                    # Save locally
                                    local_save_path = Path(output_dir) / category / filename
                                    local_save_path.parent.mkdir(parents=True, exist_ok=True)
                                    img.save(local_save_path)
                                    
                                    print(f"[SUCCESS] Downloaded: {filename} to {local_save_path}")
                                    
                            except KeyError:
                                print(f"[ERROR] File '{filename}' not found in '{zip_name}'")
                            except Exception as e:
                                print(f"[ERROR] Failed to download {filename}: {e}")
                                
            except Exception as e:
                print(f"[ERROR] Could not open remote zip '{remote_zip_path}': {e}")

        return downloaded_images

    def download_skin_cancer_samples(
        self,
        output_dir: str = "ham_images",
        num_samples: int = 5,
        seed: int = 1,
        buffer_size: int = 1000
    ) -> None:
        """
        Download random samples from the skin cancer dataset.
        
        Args:
            output_dir: Directory to save images
            num_samples: Number of random samples to download
            seed: Random seed for reproducibility
            buffer_size: Streaming buffer size
        """
        print(f"Loading skin cancer dataset...")
        
        # Load dataset in streaming mode
        ds = load_dataset("marmal88/skin_cancer", streaming=True)
        split_name = 'train' if 'train' in ds else list(ds.keys())[0]
        data_stream = ds[split_name]

        # Find label column
        label_col_name = self._find_label_column(data_stream)
        print(f"Using label column: '{label_col_name}'" if label_col_name else "No label column found")

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        print(f"Downloading {num_samples} random samples to {output_dir}/")

        # Stream and sample images
        sampled_stream = data_stream.shuffle(buffer_size=buffer_size, seed=seed).take(num_samples)

        for i, sample in enumerate(sampled_stream):
            if 'image' not in sample:
                print(f"[ERROR] No 'image' column found in sample {i}")
                continue
                
            img = sample['image']
            
            # Get and clean label
            label_val = self._get_clean_label(sample, label_col_name, data_stream)
            
            # Save image
            filename = f"sample_{i}_{label_val}.jpg"
            save_path = Path(output_dir) / filename
            
            img.convert("RGB").save(save_path)
            print(f"[SUCCESS] Saved: {save_path}")

    def _find_label_column(self, data_stream) -> Optional[str]:
        """Find the label column in the dataset features."""
        # Check for ClassLabel features
        for col_name, feature in data_stream.features.items():
            if getattr(feature, 'names', None) is not None:
                print(f"Labels found in '{col_name}' column: {feature.names}")
                return col_name
        
        # Fallback to common label column names
        for fallback_col in ['dx', 'label', 'cell_type', 'diagnosis']:
            if fallback_col in data_stream.features:
                return fallback_col
                
        return None

    def _get_clean_label(self, sample: dict, label_col_name: Optional[str], data_stream) -> str:
        """Extract and clean the label value from a sample."""
        if not label_col_name:
            return "unknown"
            
        label_val = sample.get(label_col_name, "unknown")
        
        # Map integer labels to string names if available
        if (isinstance(label_val, int) and 
            hasattr(data_stream.features[label_col_name], 'names')):
            label_val = data_stream.features[label_col_name].names[label_val]
            
        # Clean for filename
        return str(label_val).replace(" ", "_").replace("/", "-")

    @staticmethod
    def list_available_models(filter_term: Optional[str] = None) -> None:
        """List available OpenCLIP models, optionally filtered by a term."""
        available_models = open_clip.list_models()
        
        if filter_term:
            filtered_models = [m for m in available_models if filter_term.lower() in m.lower()]
            print(f"Models containing '{filter_term}':")
            for model in filtered_models:
                print(f"  {model}")
        else:
            print("All available models:")
            for model in available_models:
                print(f"  {model}")

    @staticmethod
    def list_pretrained_weights(model_name: str) -> None:
        """List available pretrained weights for a specific model."""
        try:
            pretrained_options = open_clip.list_pretrained(model_name)
            print(f"Pretrained weights for '{model_name}':")
            for option in pretrained_options:
                print(f"  {option}")
        except Exception as e:
            print(f"Error getting pretrained weights for '{model_name}': {e}")


def main():
    """Command-line interface for the dataset downloader."""
    parser = argparse.ArgumentParser(description="Download images from various datasets")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Derm1M downloader
    derm_parser = subparsers.add_parser('derm1m', help='Download Derm1M dataset images')
    derm_parser.add_argument('--output-dir', default='derm_train_images', 
                           help='Output directory for images')
    derm_parser.add_argument('--pubmed-files', nargs='*',
                           help='Specific PubMed files to download')
    derm_parser.add_argument('--iiyi-files', nargs='*',
                           help='Specific IIYI files to download')
    derm_parser.add_argument('--youtube-files', nargs='*',
                           help='Specific YouTube files to download')

    # Skin cancer downloader
    skin_parser = subparsers.add_parser('skin-cancer', help='Download skin cancer dataset samples')
    skin_parser.add_argument('--output-dir', default='ham_images',
                           help='Output directory for images')
    skin_parser.add_argument('--num-samples', type=int, default=5,
                           help='Number of random samples to download')
    skin_parser.add_argument('--seed', type=int, default=1,
                           help='Random seed for reproducibility')
    skin_parser.add_argument('--buffer-size', type=int, default=1000,
                           help='Streaming buffer size')

    # Model listing
    model_parser = subparsers.add_parser('list-models', help='List available OpenCLIP models')
    model_parser.add_argument('--filter', help='Filter models by term')
    model_parser.add_argument('--pretrained', help='Show pretrained weights for specific model')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    downloader = DatasetDownloader()

    if args.command == 'derm1m':
        downloader.download_derm1m_images(
            output_dir=args.output_dir,
            pubmed_files=args.pubmed_files,
            iiyi_files=args.iiyi_files,
            youtube_files=args.youtube_files
        )
    
    elif args.command == 'skin-cancer':
        downloader.download_skin_cancer_samples(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            buffer_size=args.buffer_size
        )
    
    elif args.command == 'list-models':
        if args.pretrained:
            downloader.list_pretrained_weights(args.pretrained)
        else:
            downloader.list_available_models(args.filter)


if __name__ == "__main__":
    main()