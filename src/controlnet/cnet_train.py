from config import data_raw_dir
from data.Dataloaders import create_train_loader, create_test_loader
import matplotlib.pyplot as plt
from monai.utils import first
import torch
from models.cnet import train_cnet

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {dev} for training.")

train_loader = create_train_loader(data_raw_dir, batch_size = 48)
test_loader = create_test_loader(data_raw_dir, batch_size=48)
train_cnet(train_loader, test_loader)