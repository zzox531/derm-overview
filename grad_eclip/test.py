import open_clip
import argparse
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import clip
# from open_clip import create_model_from_pretrained
import cv2
import numpy as np
import os
from urllib.request import urlopen
import matplotlib.pyplot as plt
from PIL import Image

# Load your target model
model, _, _ = open_clip.create_model_and_transforms("coca_ViT-B-32", pretrained="laion2b_s13b_b90k")

# Print just the vision encoder to avoid text decoder clutter
print(model.visual)