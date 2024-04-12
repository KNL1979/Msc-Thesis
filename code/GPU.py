# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:51:32 2024

@author: kimlu
"""

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")          # Use GPU if available
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")           # Use CPU if GPU is not available
    print("CUDA is not available. Using CPU.")