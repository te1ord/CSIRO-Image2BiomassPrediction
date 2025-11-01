"""
Utility functions for extracting and managing embeddings
"""
import torch
import pandas as pd
from PIL import Image
from typing import List, Tuple
from tqdm import tqdm


def extract_train_embeddings(
    train_df: pd.DataFrame,
    root_dir: str,
    model: torch.nn.Module,
    processor,
    device: str = 'cuda',
    sample_every_n: int = 5
) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
    """
    Extract embeddings from training images
    
    Args:
        train_df: Training dataframe
        root_dir: Root directory for images
        model: Feature extraction model
        processor: Image processor
        device: Device to run on
        sample_every_n: Sample every nth image
        
    Returns:
        Tuple of (embeddings, targets)
    """
    embeds = []
    targets = [[] for _ in range(5)]
    counter = 0
    
    model.eval()
    
    for i in tqdm(range(len(train_df)), desc="Extracting embeddings"):
        entry = train_df.iloc[i]
        file_path = root_dir + entry['image_path']
        y = torch.tensor([[entry['target']]])
        targets[i % 5].append(y)
        
        if i % sample_every_n == 0:
            img = Image.open(file_path)
            x = torch.tensor(processor(img).pixel_values)
            
            with torch.no_grad():
                x = x.to(device)
                embeds.append(model(x).pooler_output.cpu())
                counter += 1
                
                if counter % 100 == 0:
                    print(f"{counter} batches processed.")
    
    return embeds, targets

