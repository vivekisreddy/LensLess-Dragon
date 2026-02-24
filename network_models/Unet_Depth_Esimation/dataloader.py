import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from zipfile import ZipFile
import cv2
import random
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from utils import * 



class CombinedDatasetDataloader(Dataset):
    def __init__(self, root_directory, rgb_img_dir, depth_img_dir, img_w, img_h, img_datatype):
        # load all the images and segmented into one large list
        self.DATA = []
        skip_count = 0

        rgb_transform = T.Compose([
            T.Resize((img_h,img_w)),  # Resize to your target size
            T.Lambda(lambda x: T.functional.rotate(x, 180)),
            T.ToTensor(),          # Convert PIL to tensor and normalize to [0,1]
        ])

        depth_transform = T.Compose([
            T.Resize((img_h,img_w)),
        ])

        root_path = Path(root_directory)
        subdirectories = [d for d in root_path.iterdir() if d.is_dir()]
        
        print(f"Found {len(subdirectories)} subdirectories in {root_directory}")
        
        for dataset in subdirectories:
            rgb_path = dataset / rgb_img_dir
            depth_path = dataset / depth_img_dir
            # Check if both RGB and depth directories exist
            if not rgb_path.exists() or not depth_path.exists():
                print(f"Skipping {dataset.name}: missing rgb_img_dir or depth_img_dir")
                continue


            for file in os.listdir(rgb_path):
                try:
                    if file.endswith(img_datatype):
                        depth_file_path = depth_path / file
                        rgb_file_path = rgb_path / file

                        depth = Image.open(depth_file_path)
                        depth = depth_transform(depth)
                        depth_tensor = self.process_depth_image(
                            img=depth, 
                            max_depth=6, 
                            invalid_threshold=.35
                        )
                        if depth_tensor is None:
                            skip_count += 1
                            continue

                        rgb = Image.open(rgb_file_path)
                        rgb_tensor = rgb_transform(rgb)

                        self.DATA.append([rgb_tensor, depth_tensor])
                except Exception as e:
                        print(f"Error processing files: {e}")
                        skip_count += 1
                        continue
            print(f"Dataset {dataset} initialized, skipped {skip_count} values out of {len(self.DATA)+skip_count}")
        print(f'all done, loaded {len(self.DATA)} sets')


    def __len__(self):
        N = len(self.DATA)
        return N

    # we do a little trick: to make the 50 000 images, if the image requested number is 47589, we will return image 4758 and a random augmentation (1-9, 0 will return the original)
    def __getitem__(self, idx):
        image_idx = idx
        rgb, label = self.DATA[image_idx]
        # add the random noises if the image is not the original
        return rgb, label
    
    def process_depth_image(self, img, max_depth, invalid_threshold):
        # print(f"image datatype {img.mode}")
        img = np.array(img, dtype=np.float32) / 1e4  # Convert to meters
        # print(f"image shape:{img.shape}, img min:{img.min()}, img max{img.max()}")

        non_zero_depth = img > 0
        percent_invalid = (np.count_nonzero(img[non_zero_depth] > max_depth) / np.count_nonzero(non_zero_depth))        
        # print(f'percent_invalid: {percent_invalid}')
        if percent_invalid < invalid_threshold:
            img = np.clip(img, a_min=0, a_max=max_depth) #remove negative values and clip at 4 meters
            img = img / max_depth #normalize image
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)  # Add channel dimension
            return img
        else:
            # print(f'failed with valid px {percent_invalid}')
            return None
        
    # def sample_dataset(self, num_images):
    #     rng = np.random.default_rng()
    #     for i in range(num_images):
    #         idx = np.floor(rng.random() * len(self.DATA)).astype(int)
    #         rgb, label = self.DATA[idx]
    #         rgb1, label1 = self.DATA[idx+1]
    #         CombineImages()
##########################################################################3
###### Data Augmentaion, feel free to add more####################################
#####################################################

    def guass_noise(self, input_img):
        inputs = T.ToTensor()(input_img)
        noise = inputs + torch.rand_like(inputs) * random.uniform(0, 1.0)
        noise = torch.clip(noise, 0, 1.)
        output_image = T.ToPILImage()
        image = output_image(noise)
        return image


    def blur(self, input_img):
        blur_transfrom = T.GaussianBlur(
            kernel_size=random.choice([3, 5, 7, 9, 11]), sigma=(0.1, 1.5))
        return blur_transfrom(input_img)


    def color_jit(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)

class ZipDataloader(Dataset):
    def __init__(self, datasets, rgb_img_dir, depth_img_dir, img_w, img_h, img_datatype):
        # load all the images and segmented into one large list
        self.DATA = []
        skip_count = 0

        rgb_transform = T.Compose([
            T.Resize((img_w, img_h)),  # Resize to your target size
            T.ToTensor(),          # Convert PIL to tensor and normalize to [0,1]
        ])

        depth_transform = T.Compose([
            T.Resize((240, 320)),
        ])
        for dataset in datasets:
            with ZipFile(dataset) as myzip:
                # List all files in the ZIP
                all_files = [f for f in myzip.namelist() if not f.endswith('/')]
                # print("all files",all_files )
                # Automatically detect the top-level directory (e.g., 'training-data/')
                root_dir = os.path.commonpath(all_files).split('/')[0] + '/'
                # print("root_dir", root_dir)
                # Build correct prefixes
                train_prefix = root_dir + rgb_img_dir + '/'
                label_prefix = root_dir + depth_img_dir + '/'

                # Filter files
                train_files = [f for f in all_files if f.startswith(train_prefix)]
                label_files = [f for f in all_files if f.startswith(label_prefix)]

                # Map by filename stem
                train_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in train_files}
                label_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

                # Match filenames
                common_keys = sorted(set(train_dict) & set(label_dict))

                # Read both files for each key
                for key in common_keys:
                    with myzip.open(train_dict[key]) as train_file, myzip.open(label_dict[key]) as label_file:
                        try:
                            with myzip.open(train_dict[key]) as train_file, myzip.open(label_dict[key]) as label_file:
                                # Read depth image using OpenCV
                                train_file.seek(0)
                                label_data = Image.open(train_file)
                                label_data = np.array(label_data)  # This preserves the original dtype

                                print(f"--label_data dtype: {label_data.dtype}, shape: {label_data.shape}")
                                print(f"--label_data min: {label_data.min()}, max: {label_data.max()}")

                                # file_bytes = train_file.read()
                                # nparr = np.frombuffer(file_bytes, np.uint8)
                                # label_data = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)  # Preserve original format
                                
                                # print(f"--label_data dtype: {label_data.dtype}, shape: {label_data.shape}")
                                # print(f"--label_data min: {label_data.min()}, max: {label_data.max()}")
                                
                                # label_data = depth_transform(label_data)
                                label_tensor = self.process_depth_image(
                                    img=label_data, 
                                    max_depth=4, 
                                    invalid_threshold=.25
                                )
                                if label_tensor is None:
                                    skip_count += 1
                                    continue

                                train_data = Image.open(train_file)
                                train_tensor = rgb_transform(train_data)
                                self.DATA.append([train_tensor, label_tensor])
                                
                        except Exception as e:
                            print(f"Error processing files {train_dict[key]} and {label_dict[key]}: {e}")
                            skip_count += 1
                            continue

                # print(f"Loaded pair: {train_dict[key]} ↔ {label_dict[key]} ({len(train_data)} / {len(label_data)} bytes)")
                print(f"Dataset initialized, skipped {skip_count} values out of {len(self.DATA)+skip_count}")
        print(f'all done, loaded {len(self.DATA)} sets')


    def __len__(self):
        N = len(self.DATA)
        return N

    # we do a little trick: to make the 50 000 images, if the image requested number is 47589, we will return image 4758 and a random augmentation (1-9, 0 will return the original)
    def __getitem__(self, idx):
        # idx is from 0 to N-1                
        # Open the RGB image and ground truth label
        # convert them to tensors
        # apply any transform (blur, noise...)

        image_idx = idx
        rgb, label = self.DATA[image_idx]
        # add the random noises if the image is not the original
        # if idx % 10 != 0:
        #     rgb = self.guass_noise(rgb)
        #     rgb = self.blur(rgb)
        #     rgb = self.color_jit(rgb)
        
        # get rid of alpha in the png
        # rgb = rgb.convert("RGB")
        # rgb = T.ToTensor()(rgb)
        # label = label.convert("L")
        # label = T.ToTensor()(label)     

        return rgb, label
    
    def process_depth_image(self, img, max_depth, invalid_threshold):
        print(f"image datatype {img.mode}")
        img = np.array(img, dtype=np.float32) #/ 1e4  # Convert to meters
        print(f"image shape:{img.shape}, img min:{img.min()}, img max{img.max()}")

        non_zero_depth = img > 0
        percent_invalid = (np.count_nonzero(img[non_zero_depth] > max_depth) / np.count_nonzero(non_zero_depth))        
        
        if percent_invalid < invalid_threshold:
            img = np.clip(img, a_min=0, a_max=max_depth) #remove negative values and clip at 4 meters
            img = img / max_depth #normalize image
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)  # Add channel dimension
            return img
        else:
            # print(f'failed with valid px {percent_invalid}')
            return None
        
    # def sample_dataset(self, num_images):
    #     rng = np.random.default_rng()
    #     for i in range(num_images):
    #         idx = np.floor(rng.random() * len(self.DATA)).astype(int)
    #         rgb, label = self.DATA[idx]
    #         rgb1, label1 = self.DATA[idx+1]
    #         CombineImages()
##########################################################################3
###### Data Augmentaion, feel free to add more####################################
#####################################################

    def guass_noise(self, input_img):
        inputs = T.ToTensor()(input_img)
        noise = inputs + torch.rand_like(inputs) * random.uniform(0, 1.0)
        noise = torch.clip(noise, 0, 1.)
        output_image = T.ToPILImage()
        image = output_image(noise)
        return image


    def blur(self, input_img):
        blur_transfrom = T.GaussianBlur(
            kernel_size=random.choice([3, 5, 7, 9, 11]), sigma=(0.1, 1.5))
        return blur_transfrom(input_img)


    def color_jit(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)

class CustomImageDataset(Dataset):
    def __init__(self, rgb_img_dir, depth_img_dir, img_w, img_h, img_datatype):
        # load all the images and segmented into one large list
        self.DATA = []
        rgb_img_path = rgb_img_dir
        ground_truth_path = depth_img_dir
        skip_count = 0

        rgb_transform = T.Compose([
            T.Resize((img_w, img_h)),  # Resize to your target size
            T.ToTensor(),          # Convert PIL to tensor and normalize to [0,1]
        ])

        depth_transform = T.Compose([
            T.Resize((240, 320)),
        ])
        #TODO add support for zip files
        for file in os.listdir(rgb_img_path):
            if file.endswith(img_datatype):
                depth = Image.open(ground_truth_path + '/' + file)
                depth = depth_transform(depth)
                depth_tensor = self.process_depth_image(
                    img=depth, 
                    max_depth=4, 
                    invalid_threshold=.25
                )
                if depth_tensor is None:
                    skip_count += 1
                    continue

                rgb = Image.open(rgb_img_path + '/' + file)
                rgb_tensor = rgb_transform(rgb)

                self.DATA.append([rgb_tensor, depth_tensor])
        print(f"Dataset initialized, skipped {skip_count} values out of {len(self.DATA)+skip_count}")

    def __len__(self):
        N = len(self.DATA)
        return N

    # we do a little trick: to make the 50 000 images, if the image requested number is 47589, we will return image 4758 and a random augmentation (1-9, 0 will return the original)
    def __getitem__(self, idx):
        # idx is from 0 to N-1                
        # Open the RGB image and ground truth label
        # convert them to tensors
        # apply any transform (blur, noise...)

        image_idx = idx
        rgb, label = self.DATA[image_idx]
        # add the random noises if the image is not the original
        # if idx % 10 != 0:
        #     rgb = self.guass_noise(rgb)
        #     rgb = self.blur(rgb)
        #     rgb = self.color_jit(rgb)
        
        # get rid of alpha in the png
        # rgb = rgb.convert("RGB")
        # rgb = T.ToTensor()(rgb)
        # label = label.convert("L")
        # label = T.ToTensor()(label)     

        return rgb, label
    
    def process_depth_image(self, img, max_depth, invalid_threshold):
        img = (np.array(img, dtype=np.float32)/255) * 8.5 # we have a 8int image where 255=8.5m, scale back to metric  depth# / 1e4  # Convert to meters
        print("image shape:", img.shape)

        non_zero_depth = img > 0
        percent_invalid = (np.count_nonzero(img[non_zero_depth] > max_depth) / np.count_nonzero(non_zero_depth))        
        
        if percent_invalid < invalid_threshold:
            img = np.clip(img, a_min=0, a_max=max_depth) #remove negative values and clip at 4 meters
            img = img / max_depth #normalize image
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)  # Add channel dimension
            return img
        else:
            print(f'failed with valid px {percent_invalid}')
            return None
        
    # def sample_dataset(self, num_images):
    #     rng = np.random.default_rng()
    #     for i in range(num_images):
    #         idx = np.floor(rng.random() * len(self.DATA)).astype(int)
    #         rgb, label = self.DATA[idx]
    #         rgb1, label1 = self.DATA[idx+1]
    #         CombineImages()
##########################################################################3
###### Data Augmentaion, feel free to add more####################################
#####################################################

    def guass_noise(self, input_img):
        inputs = T.ToTensor()(input_img)
        noise = inputs + torch.rand_like(inputs) * random.uniform(0, 1.0)
        noise = torch.clip(noise, 0, 1.)
        output_image = T.ToPILImage()
        image = output_image(noise)
        return image


    def blur(self, input_img):
        blur_transfrom = T.GaussianBlur(
            kernel_size=random.choice([3, 5, 7, 9, 11]), sigma=(0.1, 1.5))
        return blur_transfrom(input_img)


    def color_jit(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)

class NYUNativeTrain(Dataset):
    """
    root/nyu2_train/<scene>/*.jpg and matching *.png (same stem) for depth
    """
    def __init__(self, root, img_w=320, img_h=240, center_crop=True, jitter=False):
        self.root = Path(root)
        self.img_w =img_w
        self.img_h = img_h
        # self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.jitter = jitter

        self.pairs = []
        scenes = [d for d in self.root.glob("*") if d.is_dir()]
        for s in tqdm(scenes, desc="Scan train scenes"):
            jpgs = sorted(s.glob("*.jpg"))
            for rgbp in jpgs:
                dep = rgbp.with_suffix(".png")
                if dep.exists():
                    self.pairs.append((rgbp, dep))

        # standard NYU crop box on 480x640
        self.crop_box = (41,45,601,471)

    def __len__(self): 
        return len(self.pairs)

    def _apply_crop(self, rgb_pil, depth_np):
        l,t,r,b = self.crop_box
        return rgb_pil.crop((l,t,r,b)), depth_np[t:b, l:r]

    def _color_jitter(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)

    def __getitem__(self, idx):
        rgbp, depp = self.pairs[idx]
        rgb = Image.open(rgbp).convert("RGB")
        depth = read_depth_png_auto(depp)

        if self.center_crop:
            rgb, depth = self._apply_crop(rgb, depth)

        rgb = rgb.resize((self.img_w,self.img_h), Image.BILINEAR)
        depth = np.array(Image.fromarray(depth).resize((self.img_w,self.img_h), Image.NEAREST), dtype=np.float32)
        rgb_t = to_tensor_img(rgb)
        print("depth", depth)
        print(f"min depth {depth.min()}, max depth {depth.max()}")
        depth_t = torch.from_numpy(depth).float().clamp(0.3,10.0)
        print("depth_t", depth_t)
        print(f"min depth_t {depth_t.min()}, max depth_t {depth_t.max()}")

        if self.jitter:
            rgb_t = self._color_jitter(rgb_t)

        return rgb_t, depth_t

class NYUNativeTest(Dataset):
    """
    root/nyu2_test/*_colors.png & *_depth.png
    """
    def __init__(self, root, img_w=320, img_h=240, center_crop=True, jitter=True):
        self.root = Path(root)
        self.img_w =img_w
        self.img_h = img_h
        # self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.jitter = jitter

        self.pairs = []
        scenes = [d for d in self.root.glob("*") if d.is_dir()]
        for s in tqdm(scenes, desc="Scan train scenes"):
            jpgs = sorted(s.glob("*.jpg"))
            for rgbp in jpgs:
                dep = rgbp.with_suffix(".png")
                if dep.exists():
                    self.pairs.append((rgbp, dep))

        # standard NYU crop box on 480x640
        self.crop_box = (41,45,601,471)


    def __len__(self): return len(self.pairs)

    def _apply_crop(self, rgb_pil, depth_np):
        l,t,r,b = self.crop_box
        return rgb_pil.crop((l,t,r,b)), depth_np[t:b, l:r]

    def __getitem__(self, idx):
        rgbp, depp = self.files[idx]
        rgb = Image.open(rgbp).convert("RGB")
        depth = read_depth_png_auto(depp)
        if self.center_crop:
            rgb, depth = self._apply_crop(rgb, depth)
        H,W = self.resize_hw
        rgb = rgb.resize((W,H), Image.BILINEAR)
        depth = np.array(Image.fromarray(depth).resize((W,H), Image.NEAREST), dtype=np.float32)
        rgb_t = to_tensor_img(rgb)
        depth_t = torch.from_numpy(depth).float().clamp(0.3,10.0)
        return rgb_t, depth_t, rgbp.stem  # stem like "00000_colors"
    
    
    
    
    
    # # verify the dataloader

