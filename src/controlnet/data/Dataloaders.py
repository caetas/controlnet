from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader
import numpy as np
import torch

channel = 0
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
        transforms.AddChanneld(keys=["image"]),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
        transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 44)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 1), random_size=False),
        transforms.Lambdad(keys=["image", "label"], func=lambda x: x.squeeze(-1)),
        transforms.CopyItemsd(keys=["label"], times=1, names=["slice_label"]),
        transforms.Lambdad(keys=["slice_label"], func=lambda x: 2.0 if x.sum() > 0 else 1.0),
        transforms.CopyItemsd(keys=["image"], times=1, names=["mask"]),
        transforms.Lambdad(keys=["mask"], func=lambda x: torch.where(x > 0.1, 1, 0)),
        #transforms.FillHolesd(keys=["mask"]),
        transforms.CastToTyped(keys=["mask"], dtype=np.float32)
    ]
)



def create_train_loader(root_dir, batch_size=64, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True):
    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=4,
        download=True,
        seed=0,
        transform=train_transforms,
    )

    print(f"Length of training data: {len(train_ds)}")
    print(f'Train image shape {train_ds[0]["image"].shape}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, persistent_workers=persistent_workers)
    return train_loader

def create_test_loader(root_dir, batch_size=64, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True):
    test_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="validation",
        cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=4,
        download=True,
        seed=0,
        transform=train_transforms,
    )
    print(f"Length of test data: {len(test_ds)}")
    print(f'Test image shape {test_ds[0]["image"].shape}')

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, persistent_workers=persistent_workers)
    return test_loader




