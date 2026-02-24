import os
turning = False
HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "DOE_Diffuser_Test_2_no_mask"
MODEL_NAME  =   "mono_depth"

if (turning):
    DATASET_RGB_PATH     =   "/scratch/hkortus/nyu_data"
    DATASET_DEPTH_PATH     =   "/scratch/hkortus/nyu_data"
    DATASET_PATH = "/scratch/hkortus/nyu_data"
    OUT_PATH    =   "/home/hkortus/mqp/lensless_perception/Unet_Depth_Esimation/runs"
else:
    DATASET_RGB_PATH     =   "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/instute1/arducam"
    DATASET_DEPTH_PATH     =   "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/instute1/realdepth"
    DATASET_PATH = "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data/instute1"
    OUT_PATH    =   "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/nyu_runs"

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")


IMAGE_W = 320
IMAGE_H = 240
IMAGE_TYPE = '.png'


BATCH_SIZE          =   8
LR                  =   5e-6
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = False
NUM_WORKERS  =   1
EPOCHS = 500
