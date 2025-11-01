"""
Biomass Dataset for loading and processing images
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable


class BiomassDataset(Dataset):
    """Dataset for CSIRO Biomass images"""
    
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        processor: Callable,
        mode: str = 'train',
        sample_every_n: int = 5,
    ):
        """
        Args:
            csv_path: Path to the CSV file with annotations
            root_dir: Root directory with all the images
            processor: Image processor (e.g., DINOv2 processor)
            mode: 'train' or 'test'
            sample_every_n: Sample every nth image (default: 5, as in notebook)
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.processor = processor
        self.mode = mode
        self.sample_every_n = sample_every_n
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.df) // self.sample_every_n
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # Sample every nth image as in the notebook
            actual_idx = idx * self.sample_every_n
        else:
            actual_idx = idx
            
        entry = self.df.iloc[actual_idx]
        file_path = self.root_dir + entry['image_path']
        
        # Load and process image
        img = Image.open(file_path)
        pixel_values = torch.tensor(self.processor(img).pixel_values)
        
        if self.mode == 'train':
            target = torch.tensor([[entry['target']]])
            return pixel_values, target, actual_idx
        else:
            sample_id = entry['sample_id']
            return pixel_values, sample_id

