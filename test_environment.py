import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import osmnx as ox

print("=" * 50)
print("URBANWATCH ENVIRONMENT TEST")
print("=" * 50)

print("\nPyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)

print("\nCUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print("\nOpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("GeoPandas:", gpd.__version__)
print("Shapely:", shapely.__version__)
print("OSMnx:", ox.__version__)

print("\n" + "=" * 50)
print("EVERYTHING LOADED SUCCESSFULLY")
print("=" * 50)