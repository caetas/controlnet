from config import data_raw_dir
from data.Dataloaders import create_train_loader, create_test_loader
import matplotlib.pyplot as plt
from monai.utils import first
import torch
from models.DDPM import train_DDPM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {dev} for training.")

train_loader = create_train_loader(data_raw_dir)
test_loader = create_test_loader(data_raw_dir)
train_DDPM(train_loader, test_loader)
'''
check_data = first(train_loader)
print(f"Batch shape: {check_data['image'].shape}")
image_visualisation = torch.cat(
    (
        torch.cat(
            [
                check_data["image"][0, 0],
                check_data["image"][1, 0],
                check_data["image"][2, 0],
                check_data["image"][3, 0],
            ],
            dim=1,
        ),
        torch.cat(
            [check_data["label"][0, 0]/3.0, check_data["label"][1, 0]/3.0, check_data["label"][2, 0]/3.0, check_data["label"][3, 0]/3.0],
            dim=1,
        ),
    ),
    dim=0,
)
plt.figure(figsize=(6, 3))
plt.imshow(image_visualisation, vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
'''