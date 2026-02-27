from dataloader import CustomImageDataset, ZipDataloader, CombinedDatasetDataloader
from params import *
from torchvision import transforms
import torch
from torch import nn


# dataset_list = [
#                 '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170100.zip',
#                 '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170515.zip',
#                 '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170847.zip',
#                 '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170958.zip',
#                 '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_171249.zip']

dataset_list = [
    '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_164853',
    '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170100',
    '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170515',
    '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170847',
    '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_170958',
    '/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/session_20251008_171249'
]

# dataset = CustomImageDataset(
#     rgb_img_dir=DATASET_RGB_PATH, 
#     depth_img_dir=DATASET_DEPTH_PATH, 
#     img_w=IMAGE_W, 
#     img_h=IMAGE_H, 
#     img_datatype=IMAGE_TYPE, 
#     )

dataset = CombinedDatasetDataloader(
    root_directory='/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/DOE_3',
    rgb_img_dir='arducam',
    depth_img_dir='realdepth',
    img_w=IMAGE_W,
    img_h=IMAGE_H,
    img_datatype=IMAGE_TYPE
)

trainLoader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=NUM_WORKERS)


