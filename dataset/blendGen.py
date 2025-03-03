import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Resize
import math
import random

class BlenderGenDataset_old(Dataset):
    def __init__(self, root_dir, mode='train', random_flip=False, random_rotation=False, resize=None):
        assert mode in ['train', 'test', 'val'], "Mode must be 'train', 'test', or 'val'"

        self.root_dir = root_dir
        self.mode = mode
        self.resize = resize

        self.random_flip = random_flip
        self.random_rotation = random_rotation
        # Conditionally add Resize transformation if resize argument is provided
        transform_list =  []
        # transform_list += [ToTensor()]  # Always include ToTensor transformation
        #transform_list.append(RandomRotation(degrees=60))
        # Add a random horizontal flip with a default 50% chance
        
        
        #self.transform = Compose(transform_list)

        self.samples = []

        # Adjust the subfolder name based on the mode
        mode_folder = f"{mode}_000"

        # Traverse through each main category directory (e.g., coffee_gen)
        for category_dir in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category_dir)
            if os.path.isdir(category_path):
                # Traverse through each roughness directory
                for roughness_dir in os.listdir(category_path):
                    roughness_path = os.path.join(category_path, roughness_dir)
                    if os.path.isdir(roughness_path):
                        # Go into the specified mode directory (train_000, test_000, val_000)
                        specific_mode_path = os.path.join(roughness_path, mode_folder)
                        if os.path.exists(specific_mode_path):
                            # Store the path to the specific mode directory
                            self.samples.append(specific_mode_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mode_path = self.samples[idx]
        
        # Load each image separately by specifying its filename
        rgba_img_path = os.path.join(mode_path, 'rgba.png')
        metallic_img_path = os.path.join(mode_path, 'metallic.png')
        roughness_img_path = os.path.join(mode_path, 'roughness.png')
        normal_img_path = os.path.join(mode_path, 'normal.png')
        light_image_path = '/hpc2hdd/home/wge950/projects/abandoned_factory_canteen_01_2k.png'
        
        # Use PIL to open and convert images
        rgba_image = Image.open(rgba_img_path).convert('RGB') # Convert RGBA to RGB
        metallic_image = Image.open(metallic_img_path).convert('RGB')
        roughness_image = Image.open(roughness_img_path).convert('RGB')
        normal_image = Image.open(normal_img_path).convert('RGB')
        light_image = Image.open(light_image_path).convert('RGB')

        if self.random_rotation:
            angle = random.randint(-60, 60)  # Choose a random angle between -30 and 30 degrees
            rgba_image = rgba_image.rotate(angle, expand=True)
            metallic_image = metallic_image.rotate(angle, expand=True)
            roughness_image = roughness_image.rotate(angle, expand=True)
            normal_image = normal_image.rotate(angle, expand=True)

        rgba_image = np.array(rgba_image.resize(self.resize))  # Convert RGBA to RGB
        metallic_image = np.array(metallic_image.resize(self.resize))
        roughness_image = np.array(roughness_image.resize(self.resize))
        normal_image = np.array(normal_image.resize(self.resize))
        light_image = np.array(light_image.resize(self.resize))
        
        # Use PIL to open and convert images
        # rgba_image = np.array(Image.open(rgba_img_path).convert('RGB').resize(self.resize))  # Convert RGBA to RGB
        # metallic_image = np.array(Image.open(metallic_img_path).convert('RGB').resize(self.resize))
        # roughness_image = np.array(Image.open(roughness_img_path).convert('RGB').resize(self.resize))
        # normal_image = np.array(Image.open(normal_img_path).convert('RGB').resize(self.resize))


        #Apply transformations if any
        if self.random_flip and random.random() < 0.5:
            rgba_image = rgba_image[:, ::-1]
            metallic_image = metallic_image[:, ::-1]
            roughness_image = roughness_image[:, ::-1]
            normal_image = normal_image[:, ::-1]

        rgba_image = rgba_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        metallic_image = metallic_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        roughness_image = roughness_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        normal_image = normal_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        light_image = light_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]



        # Return the images explicitly in the specified order
        #return rgba_image, {'metallic_image': np.transpose(metallic_image, [2, 0, 1]), 'roughness_image': np.transpose(roughness_image, [2, 0, 1]), 'normal_image': np.transpose(normal_image, [2, 0, 1])}

        return {'rgb': np.transpose(rgba_image, [2, 0, 1]), 'light': np.transpose(light_image, [2, 0, 1]), 'metallic': np.transpose(metallic_image, [2, 0, 1]), 'roughness': np.transpose(roughness_image, [2, 0, 1]), 'normal': np.transpose(normal_image, [2, 0, 1])}

# # Define any transforms you want to apply to every image
class BlenderGenDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, resize=None):
        assert mode in ['train', 'test', 'val'], "Mode must be 'train', 'test', or 'val'"

        self.root_dir = root_dir
        self.mode = mode
        self.resize = resize
        # Conditionally add Resize transformation if resize argument is provided
        transform_list = [Resize(resize)] if resize is not None else []
        # transform_list += [ToTensor()]  # Always include ToTensor transformation
        
        self.transform = Compose(transform_list)

        self.samples = []

        # Adjust the subfolder name based on the mode
        mode_folder = f"{mode}_000"

        # Traverse through each main category directory (e.g., coffee_gen)
        for category_dir in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category_dir)
            if os.path.isdir(category_path):
                # Traverse through each roughness directory
                for roughness_dir in os.listdir(category_path):
                    roughness_path = os.path.join(category_path, roughness_dir)
                    if os.path.isdir(roughness_path):
                        # Go into the specified mode directory (train_000, test_000, val_000)
                        specific_mode_path = os.path.join(roughness_path, mode_folder)
                        if os.path.exists(specific_mode_path):
                            # Store the path to the specific mode directory
                            self.samples.append(specific_mode_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mode_path = self.samples[idx]
        
        # Load each image separately by specifying its filename
        rgba_img_path = os.path.join(mode_path, 'rgba.png')
        metallic_img_path = os.path.join(mode_path, 'metallic.png')
        roughness_img_path = os.path.join(mode_path, 'roughness.png')
        normal_img_path = os.path.join(mode_path, 'normal.png')
        
        # Use PIL to open and convert images
        rgba_image = np.array(Image.open(rgba_img_path).convert('RGB').resize(self.resize))  # Convert RGBA to RGB
        metallic_image = np.array(Image.open(metallic_img_path).convert('RGB').resize(self.resize))
        roughness_image = np.array(Image.open(roughness_img_path).convert('RGB').resize(self.resize))
        normal_image = np.array(Image.open(normal_img_path).convert('RGB').resize(self.resize))

        # Apply transformations if any
        rgba_image = rgba_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        metallic_image = metallic_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        roughness_image = roughness_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]
        normal_image = normal_image.astype(np.float32) / 127.5 - 1.0  # range [-1, 1]



        # if self.transform:
        #     rgba_image = self.transform(rgba_image)
        #     metallic_image = self.transform(metallic_image)
        #     roughness_image = self.transform(roughness_image)
        #     normal_image = self.transform(normal_image)

        # Return the images explicitly in the specified order
        return np.transpose(rgba_image, [2, 0, 1]), {'metallic_image': np.transpose(metallic_image, [2, 0, 1]), 'roughness_image': np.transpose(roughness_image, [2, 0, 1]), 'normal_image': np.transpose(normal_image, [2, 0, 1])}

# Define any transforms you want to apply to every image



class BlenderGenDataset_3mod(Dataset):
    def __init__(self, root_dir, transform=None, resize=None):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        self.resize = resize

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])
    
    def check_corrupted(self, img_path):
        try:
        # Attempt to open the image
            image = Image.open(img_path)
            return True
        except (FileNotFoundError, OSError) as e:
        # If an error occurs, print the error and return False
            print(f"Error opening image: {e}")
            print(img_path)
            return False

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for inter_folder_name in os.listdir(subdir_path):
                inter_folder = os.path.join(subdir_path, inter_folder_name)
                
                if not os.path.isdir(inter_folder):
                    continue
                
                for id_name in os.listdir(inter_folder):
                    id_folder = os.path.join(inter_folder, id_name)
                    if not os.path.isdir(id_folder):
                        continue
                    #print(id_folder)
                    albedo_dir = os.path.join(id_folder, 'albedo')
                    normal_dir = os.path.join(id_folder, 'normal')
                    env_dir = os.path.join(id_folder, 'env')
                    
                    rgb_dir = os.path.join(id_folder, 'rgb')
                    #roughness_dir = os.path.join(id_folder, 'roughness')
                    #print(rgb_dir)
                    if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                        continue

                    albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
                    
                    for f in albedo_files:
                        index = os.path.splitext(f)[0]
                        #print(index)
                        normal_files = self._find_matching_files(normal_dir, index)
                        #print(normal_files)
                        #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                        index_digit = int(index)  # Converts "000" to 0, "001" to 1, etc.
                        #index_str = str(index)
      
                        rgb_files = self._find_matching_files(rgb_dir, index_digit, with_params=True)

                        env_files = []

                        for filepath in rgb_files:
                            env_path =  filepath.replace('/rgb/', '/env/') # does nothing
                            #print(env_path)
                            env_files.append(env_path)
                        
                        #env_files = self._find_matching_files(env_dir, index_digit, with_params=True)

                        if normal_files and rgb_files and env_files:
                            normal_file = os.path.join(normal_dir, normal_files[0])
                            #metallic_file = os.path.join(metallic_dir, metallic_files[0])
                            for rgb_file, env_file in zip(rgb_files, env_files):
                                rgb_file = os.path.join(rgb_dir, rgb_file)
                                env_file = os.path.join(env_dir, env_file)
                                #print(rgb_file, env_file)
                                samples.append((
                                        os.path.join(albedo_dir, f),
                                        normal_file,
                                        #metallic_file,
                                        rgb_file,
                                        env_file
                                    ))

        return samples

    def __len__(self):
        return len(self.samples)
    
    def preprocess(self, image_path, bool_op=False):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        image = image.astype(np.float32) / 127.5 - 1.0 
        # if bool_op:
        #     bool_threshold = -0.97
        #     image[image<bool_threshold] = -1.
        #     unique_elements, counts = np.unique(image, return_counts=True)
        #     indices = np.argsort(-counts)[:2] # get top-2
        #     assert unique_elements[indices][0] == -1.
        #     try:
        #         image[image>=bool_threshold] = unique_elements[indices][1] # top2
        #     except:
        #         print(image_path)
        #         print(unique_elements[indices], counts[indices])
            
        image = np.transpose(image, [2, 0, 1])

        return image.copy()
    
    def create_from_name(self, image_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask = torch.from_numpy(image[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2,0,1)
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)
        


    def __getitem__(self, idx):
        albedo_path, normal_path, rgb_path, env_path = self.samples[idx]
        #breakpoint()
        #light_image_path = '/hpc2hdd/home/zchen379/working/abandoned_factory_canteen_01_2k.png'
        
        material = self.create_from_name(rgb_path)
        
        sample = {
            'albedo': self.preprocess(albedo_path),
            'normal': self.preprocess(normal_path),
            'material': material,
            'rgb': self.preprocess(rgb_path),
            'light': self.preprocess(env_path)
        }

        return sample



class BlenderGenDataset_3mod_final(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, resize=None, random_flip=False, random_crop=False):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.samples = self._load_samples()
        self.resize = resize
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.image_aug_pad_width = ((0, 0), (64, 64), (64, 64))
        self.image_size = resize[0]
        print(f"dataset augmentation, random_flip: {random_flip}, random_crop: {random_crop}, image_size: {self.image_size}.")

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for inter_folder_name in os.listdir(subdir_path):
                inter_folder = os.path.join(subdir_path, inter_folder_name)
                if not os.path.isdir(inter_folder):
                    continue
                
                for id_name in os.listdir(inter_folder):
                    id_folder = os.path.join(inter_folder, id_name)
                    if not os.path.isdir(id_folder):
                        continue

                    albedo_dir = os.path.join(id_folder, 'albedo')
                    normal_dir = os.path.join(id_folder, 'normal')
                    #metallic_dir = os.path.join(id_folder, 'metallic')
                    
                    rgb_dir = os.path.join(id_folder, f'rgb_{self.mode}')
                    #roughness_dir = os.path.join(id_folder, 'roughness')

                    if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                        continue

                    albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
                    
                    for f in albedo_files:
                        index = os.path.splitext(f)[0]
                        normal_files = self._find_matching_files(normal_dir, index)
                        #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                        rgb_files = self._find_matching_files(rgb_dir, index, with_params=True)
                        #roughness_files = self._find_matching_files(roughness_dir, index, with_params=True)

                        if normal_files  and rgb_files:
                            normal_file = os.path.join(normal_dir, normal_files[0])
                            #metallic_file = os.path.join(metallic_dir, metallic_files[0])
                            for rgb_file in rgb_files:
                                rgb_file = os.path.join(rgb_dir, rgb_file)
                            #rgb_file = os.path.join(rgb_dir, rgb_files[0])
                            #roughness_file = os.path.join(roughness_dir, roughness_files[0])

                                samples.append((
                                    os.path.join(albedo_dir, f),
                                    normal_file,
                                    #metallic_file,
                                    rgb_file,
                                    #roughness_file
                                ))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def random_crop_arr(self, image, pad_width, crop_x, crop_y, pad_value=-1.):
        padded_image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=pad_value)
        return padded_image[:, crop_y : crop_y + self.image_size, crop_x : crop_x + self.image_size]

    def preprocess(self, image_path, random_crop_flag=False, random_flip_flag=False, crop=None):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        image = image.astype(np.float32) / 127.5 - 1.0 
        
        # if bool_op:
        #     bool_threshold = -0.97
        #     image[image<bool_threshold] = -1.
        #     unique_elements, counts = np.unique(image, return_counts=True)
        #     indices = np.argsort(-counts)[:2] # get top-2
        #     assert unique_elements[indices][0] == -1.
        #     try:
        #         image[image>=bool_threshold] = unique_elements[indices][1] # top2
        #     except:
        #         print(image_path)
        #         print(unique_elements[indices], counts[indices])
            
        image = np.transpose(image, [2, 0, 1])

        if random_crop_flag:
            assert crop is not None
            image = self.random_crop_arr(image, self.image_aug_pad_width, crop_x=crop['x'], crop_y=crop['y'])

        if random_flip_flag:
            image = image[:, ::-1]
            #image = torch.flip(image, [1])

        return image.copy()
    
    def create_from_name(self, image_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask = torch.from_numpy(image[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2, 0, 1)
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)

    def __getitem__(self, idx):
        albedo_path, normal_path, rgb_path = self.samples[idx]
        #breakpoint()
        light_image_path = '/hpc2hdd/home/zchen379/working/abandoned_factory_canteen_01_2k.png'
        
        material = self.create_from_name(rgb_path)

        random_crop_flag = False
        crop = None
        if self.random_crop and random.random() < 0.8:
            padded_image = self.image_size + 2 * self.image_aug_pad_width[1][0]
            crop_y = random.randrange(padded_image - self.image_size + 1)
            crop_x = random.randrange(padded_image - self.image_size + 1)
            crop = {'x': crop_x, 'y': crop_y}

            material = torch.from_numpy(self.random_crop_arr(material, self.image_aug_pad_width, crop_x, crop_y)) # will cast into ndarray
            random_crop_flag = True    
            
        random_flip_flag = False
        if self.random_flip and random.random() < 0.4:
            material = torch.from_numpy(material.numpy()[:, ::-1].copy())
            random_flip_flag = True

        #print(random_crop_flag, random_flip_flag, crop)

        albedo = self.preprocess(albedo_path, random_crop_flag, random_flip_flag, crop=crop)
        normal = self.preprocess(normal_path, random_crop_flag, random_flip_flag, crop=crop)
        rgb = self.preprocess(rgb_path, random_crop_flag, random_flip_flag, crop=crop)
        light = self.preprocess(light_image_path, random_crop_flag, random_flip_flag, crop=crop)
        #print(albedo.shape, normal.shape, rgb.shape, light.shape)

        return {'albedo': albedo, 'normal': normal, 'material': material, 'rgb': rgb, 'light': light}
    
class BlenderGenDataset_3mod_final_validation(Dataset):
    def __init__(self, root_dir, mode='test', transform=None, resize=None):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.samples = self._load_samples()
        self.resize = resize

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for inter_folder_name in os.listdir(subdir_path):
                inter_folder = os.path.join(subdir_path, inter_folder_name)
                if not os.path.isdir(inter_folder):
                    continue
                
                for id_name in os.listdir(inter_folder):
                    id_folder = os.path.join(inter_folder, id_name)
                    if not os.path.isdir(id_folder):
                        continue

                    albedo_dir = os.path.join(id_folder, 'albedo')
                    normal_dir = os.path.join(id_folder, 'normal')
                    #metallic_dir = os.path.join(id_folder, 'metallic')
                    
                    rgb_dir = os.path.join(id_folder, f'rgb_{self.mode}')
                    #roughness_dir = os.path.join(id_folder, 'roughness')

                    if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                        continue

                    albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
                    
                    for f in albedo_files:
                        index = os.path.splitext(f)[0]
                        normal_files = self._find_matching_files(normal_dir, index)
                        #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                        rgb_files = self._find_matching_files(rgb_dir, index, with_params=True)
                        #roughness_files = self._find_matching_files(roughness_dir, index, with_params=True)

                        if normal_files  and rgb_files:
                            normal_file = os.path.join(normal_dir, normal_files[0])
                            #metallic_file = os.path.join(metallic_dir, metallic_files[0])
                            for rgb_file in rgb_files:
                                rgb_file = os.path.join(rgb_dir, rgb_file)
                            #rgb_file = os.path.join(rgb_dir, rgb_files[0])
                            #roughness_file = os.path.join(roughness_dir, roughness_files[0])

                                samples.append((
                                    os.path.join(albedo_dir, f),
                                    normal_file,
                                    #metallic_file,
                                    rgb_file,
                                    #roughness_file
                                ))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def preprocess(self, image_path, bool_op=False):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        image = image.astype(np.float32) / 127.5 - 1.0 
        # if bool_op:
        #     bool_threshold = -0.97
        #     image[image<bool_threshold] = -1.
        #     unique_elements, counts = np.unique(image, return_counts=True)
        #     indices = np.argsort(-counts)[:2] # get top-2
        #     assert unique_elements[indices][0] == -1.
        #     try:
        #         image[image>=bool_threshold] = unique_elements[indices][1] # top2
        #     except:
        #         print(image_path)
        #         print(unique_elements[indices], counts[indices])
            
        image = np.transpose(image, [2, 0, 1])

        return image.copy()
    
    def create_from_name(self, image_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask = torch.from_numpy(image[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2, 0, 1), mask.permute(2, 0, 1)
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)
        


    def __getitem__(self, idx):
        albedo_path, normal_path, rgb_path = self.samples[idx]
        #breakpoint()
        light_image_path = '/hpc2hdd/home/zchen379/working/abandoned_factory_canteen_01_2k.png'
        
        material, mask = self.create_from_name(rgb_path)
        
        sample = {
            'albedo': albedo_path,
            'normal': normal_path,
            'material': material,
            'mask': mask,
            'rgb': rgb_path,
            'light': light_image_path

        }

        return sample



class BlenderGenDataset_3mod_new(Dataset):  




    def __init__(self, root_dir, env_dir, transform=None, resize=None):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples(env_dir)
        self.resize = resize

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])
    
    def check_corrupted(self, img_path):
        try:
        # Attempt to open the image
            image = Image.open(img_path)
            return True
        except (FileNotFoundError, OSError) as e:
        # If an error occurs, print the error and return False
            print(f"Error opening image: {e}")
            print(img_path)
            return False

    def _load_samples(self, env_dir):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for inter_folder_name in os.listdir(subdir_path):
                inter_folder = os.path.join(subdir_path, inter_folder_name)
                
                if not os.path.isdir(inter_folder):
                    continue
                
                for id_name in os.listdir(inter_folder):
                    id_folder = os.path.join(inter_folder, id_name)
                    if not os.path.isdir(id_folder):
                        continue
                    #print(id_folder)
                    albedo_dir = os.path.join(id_folder, 'albedo')
                    normal_dir = os.path.join(id_folder, 'normal')
                    
                    rgb_dir = os.path.join(id_folder, 'rgb')
                    #roughness_dir = os.path.join(id_folder, 'roughness')
                    #print(rgb_dir)
                    if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                        continue

                    # albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
                    _path = rgb_dir + '*.png' if rgb_dir[-1] == '/' else rgb_dir + '/*.png'
                    for rgb_path in glob(_path):
                        index = rgb_path.split('/')[-1][:-len(".png")].split("_")

                        normal_file = os.path.join(normal_dir, f'{index[0].zfill(3)}.png')
                        albedo_file = os.path.join(albedo_dir, f'{index[0].zfill(3)}.png')
                        env_file = os.path.join(env_dir, f'{index[0].zfill(3)}.png')
                        # index = os.path.splitext(f)[0]
                        #print(index)
                        # normal_files = self._find_matching_files(normal_dir, index)
                        #print(normal_files)
                        #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                        #index_digit = int(index)  # Converts "000" to 0, "001" to 1, etc.
                        #index_str = str(index)
      
                        #rgb_files = self._find_matching_files(rgb_dir, index_digit, with_params=True)
        
                        # env_files = []

                        # for filepath in rgb_files:
                        #     env_path =  filepath.replace('/rgb/', '/env/') # does nothing
                        #     #print(env_path)
                        #     env_files.append(env_path)
                        
                        #env_files = self._find_matching_files(env_dir, index_digit, with_params=True)

                        if normal_file and rgb_path and env_file and albedo_file:
                            samples.append((albedo_file, normal_file, rgb_path, env_file))

        return samples

    def __len__(self):
        return len(self.samples)
    
    def preprocess(self, image_path, bool_op=False):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        image = image.astype(np.float32) / 127.5 - 1.0 
        # if bool_op:
        #     bool_threshold = -0.97
        #     image[image<bool_threshold] = -1.
        #     unique_elements, counts = np.unique(image, return_counts=True)
        #     indices = np.argsort(-counts)[:2] # get top-2
        #     assert unique_elements[indices][0] == -1.
        #     try:
        #         image[image>=bool_threshold] = unique_elements[indices][1] # top2
        #     except:
        #         print(image_path)
        #         print(unique_elements[indices], counts[indices])
            
        image = np.transpose(image, [2, 0, 1])

        return image.copy()
    
    def create_from_name(self, image_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask = torch.from_numpy(image[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2,0,1)
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)
        


    def __getitem__(self, idx):
        albedo_path, normal_path, rgb_path, env_path = self.samples[idx]
        #breakpoint()
        #light_image_path = '/hpc2hdd/home/zchen379/working/abandoned_factory_canteen_01_2k.png'
        
        material = self.create_from_name(rgb_path)
        
        sample = {
            'albedo': self.preprocess(albedo_path),
            'normal': self.preprocess(normal_path),
            'material': material,
            'rgb': self.preprocess(rgb_path),
            'light': self.preprocess(env_path)
        }

        return sample
    

class BlenderGenDataset_3mod_demo(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, resize=None, random_flip=False, random_crop=False):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        #self.mask_img_path = None
        self.samples = self._load_samples()
        self.resize = resize
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.image_aug_pad_width = ((0, 0), (64, 64), (64, 64))
        self.image_size = resize[0]
        print(f"dataset augmentation, random_flip: {random_flip}, random_crop: {random_crop}, image_size: {self.image_size}.")

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

                # for inter_folder_name in os.listdir(subdir_path):
                #     inter_folder = os.path.join(subdir_path, inter_folder_name)
                #     if not os.path.isdir(inter_folder):
                #         continue
                    
                #     for id_name in os.listdir(inter_folder):
                #         id_folder = os.path.join(inter_folder, id_name)
                #         if not os.path.isdir(id_folder):
                #             continue
            
            mask_img_path = os.path.join(subdir_path, 'mask.png') 
     
            albedo_dir = os.path.join(subdir_path, 'albedo')
            normal_dir = os.path.join(subdir_path, 'normal')
            #metallic_dir = os.path.join(id_folder, 'metallic')
            
            rgb_dir = os.path.join(subdir_path, 'rgb')
            #roughness_dir = os.path.join(id_folder, 'roughness')
            #breakpoint()
            if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                continue

            albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
            
            for f in albedo_files:
                #breakpoint()
                index = os.path.splitext(f)[0]
                normal_files = self._find_matching_files(normal_dir, index)
                #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                rgb_files = self._find_matching_files(rgb_dir, index, with_params=True)
                #roughness_files = self._find_matching_files(roughness_dir, index, with_params=True)

                if normal_files  and rgb_files:
                    normal_file = os.path.join(normal_dir, normal_files[0])
                    #metallic_file = os.path.join(metallic_dir, metallic_files[0])
                    for rgb_file in rgb_files:
                        rgb_file = os.path.join(rgb_dir, rgb_file)
                    #rgb_file = os.path.join(rgb_dir, rgb_files[0])
                    #roughness_file = os.path.join(roughness_dir, roughness_files[0])

                        samples.append((
                            os.path.join(albedo_dir, f),
                            normal_file,
                            #metallic_file,
                            rgb_file,
                            #roughness_file
                            mask_img_path,
                        ))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def random_crop_arr(self, image, pad_width, crop_x, crop_y, pad_value=-1.):
        padded_image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=pad_value)
        return padded_image[:, crop_y : crop_y + self.image_size, crop_x : crop_x + self.image_size]

    def preprocess(self, image_path, random_crop_flag=False, random_flip_flag=False, crop=None, mask_path=None):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        

        if mask_path is not None:
            mask_img = np.array(Image.open(mask_path).resize(self.resize))
            mask = np.array(mask_img)[:, :, 3] > 128
            mask = mask.astype(bool)
            mask = np.stack([mask]*3, axis=-1)
            masked_image = np.zeros_like(image)  # Start with an all-black image
            masked_image[mask] = image[mask]
            image = masked_image
        image = image.astype(np.float32) / 127.5 - 1.0 


        
        
        # if bool_op:
        #     bool_threshold = -0.97
        #     image[image<bool_threshold] = -1.
        #     unique_elements, counts = np.unique(image, return_counts=True)
        #     indices = np.argsort(-counts)[:2] # get top-2
        #     assert unique_elements[indices][0] == -1.
        #     try:
        #         image[image>=bool_threshold] = unique_elements[indices][1] # top2
        #     except:
        #         print(image_path)
        #         print(unique_elements[indices], counts[indices])
            
        image = np.transpose(image, [2, 0, 1])

        if random_crop_flag:
            assert crop is not None
            image = self.random_crop_arr(image, self.image_aug_pad_width, crop_x=crop['x'], crop_y=crop['y'])

        if random_flip_flag:
            image = image[:, ::-1]
            #image = torch.flip(image, [1])

        return image.copy()
    
    def create_from_name(self, image_path, mask_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])
        

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask_img = np.array(Image.open(mask_path).resize(self.resize))

        #breakpoint()
        mask = torch.from_numpy(mask_img[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2, 0, 1)
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)

    def __getitem__(self, idx):
        albedo_path, normal_path, rgb_path, mask_path = self.samples[idx]
        #breakpoint()
        light_image_path = '/hpc2hdd/home/zchen379/working/christmas_photo_studio_07_4k.png'
        
        material = self.create_from_name(rgb_path, mask_path)

        random_crop_flag = False
        crop = None
        if self.random_crop and random.random() < 0.8:
            padded_image = self.image_size + 2 * self.image_aug_pad_width[1][0]
            crop_y = random.randrange(padded_image - self.image_size + 1)
            crop_x = random.randrange(padded_image - self.image_size + 1)
            crop = {'x': crop_x, 'y': crop_y}

            material = torch.from_numpy(self.random_crop_arr(material, self.image_aug_pad_width, crop_x, crop_y)) # will cast into ndarray
            random_crop_flag = True    
            
        random_flip_flag = False
        if self.random_flip and random.random() < 0.4:
            material = torch.from_numpy(material.numpy()[:, ::-1].copy())
            random_flip_flag = True

        #print(random_crop_flag, random_flip_flag, crop)

        albedo = self.preprocess(albedo_path, random_crop_flag, random_flip_flag, crop=crop, mask_path=mask_path)
        normal = self.preprocess(normal_path, random_crop_flag, random_flip_flag, crop=crop, mask_path=mask_path)
        rgb = self.preprocess(rgb_path, random_crop_flag, random_flip_flag, crop=crop)
        light = self.preprocess(light_image_path, random_crop_flag, random_flip_flag, crop=crop)
        #print(albedo.shape, normal.shape, rgb.shape, light.shape)

        return {'albedo': albedo, 'normal': normal, 'material': material, 'rgb': rgb, 'light': light}


class BlenderGenDataset_3mod_demo_validation(Dataset):
    def __init__(self, root_dir, mode='test', transform=None, resize=None, random_flip=False, random_crop=False):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        #self.mask_img_path = None
        self.samples = self._load_samples()
        self.resize = resize

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

                # for inter_folder_name in os.listdir(subdir_path):
                #     inter_folder = os.path.join(subdir_path, inter_folder_name)
                #     if not os.path.isdir(inter_folder):
                #         continue
                    
                #     for id_name in os.listdir(inter_folder):
                #         id_folder = os.path.join(inter_folder, id_name)
                #         if not os.path.isdir(id_folder):
                #             continue
            
            mask_img_path = os.path.join(subdir_path, 'mask.png') 
     
            albedo_dir = os.path.join(subdir_path, 'albedo')
            normal_dir = os.path.join(subdir_path, 'normal')
            #metallic_dir = os.path.join(id_folder, 'metallic')
            
            rgb_dir = os.path.join(subdir_path, 'rgb')
            #roughness_dir = os.path.join(id_folder, 'roughness')
            #breakpoint()
            if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                continue

            albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
            
            for f in albedo_files:
                #breakpoint()
                index = os.path.splitext(f)[0]
                normal_files = self._find_matching_files(normal_dir, index)
                #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                rgb_files = self._find_matching_files(rgb_dir, index, with_params=True)
                #roughness_files = self._find_matching_files(roughness_dir, index, with_params=True)

                if normal_files  and rgb_files:
                    normal_file = os.path.join(normal_dir, normal_files[0])
                    #metallic_file = os.path.join(metallic_dir, metallic_files[0])
                    for rgb_file in rgb_files:
                        rgb_file = os.path.join(rgb_dir, rgb_file)
                    #rgb_file = os.path.join(rgb_dir, rgb_files[0])
                    #roughness_file = os.path.join(roughness_dir, roughness_files[0])

                        samples.append((
                            os.path.join(albedo_dir, f),
                            normal_file,
                            #metallic_file,
                            rgb_file,
                            #roughness_file
                            mask_img_path,
                        ))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def preprocess(self, image_path, random_crop_flag=False, random_flip_flag=False, crop=None, mask_path=None):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        

        if mask_path is not None:
            mask_img = np.array(Image.open(mask_path).resize(self.resize))
            mask = np.array(mask_img)[:, :, 3] > 128
            mask = mask.astype(bool)
            mask = np.stack([mask]*3, axis=-1)
            masked_image = np.zeros_like(image)  # Start with an all-black image
            masked_image[mask] = image[mask]
            image = masked_image
        image = image.astype(np.float32) / 127.5 - 1.0 


            
        image = np.transpose(image, [2, 0, 1])

        if random_crop_flag:
            assert crop is not None
            image = self.random_crop_arr(image, self.image_aug_pad_width, crop_x=crop['x'], crop_y=crop['y'])

        if random_flip_flag:
            image = image[:, ::-1]
            #image = torch.flip(image, [1])

        return image.copy()
    
    def create_from_name(self, image_path, mask_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])
        

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask_img = np.array(Image.open(mask_path).resize(self.resize))
    
        mask = torch.from_numpy(mask_img[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2, 0, 1)
        
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)
        


    def __getitem__(self, idx):

        
        albedo_path, normal_path, rgb_path, mask_path = self.samples[idx]
        #breakpoint()
        light_image_path = '/hpc2hdd/home/zchen379/working/christmas_photo_studio_07_4k.png'
        material = self.create_from_name(rgb_path, mask_path)
        
        sample = {
            'albedo': albedo_path,
            'normal': normal_path,
            'material': material,
            'rgb': rgb_path,
            'light': light_image_path,
            'mask_path': mask_path
        }

        return sample


        
class BlenderGenDataset_3mod_demo_teaser(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, resize=None, random_flip=False, random_crop=False):
        """
        Args:
            root_dir (string): Directory with all the 'geometry data'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        #self.mask_img_path = None
        self.samples = self._load_samples()
        self.resize = resize
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.image_aug_pad_width = ((0, 0), (64, 64), (64, 64))
        self.image_size = resize[0]
        print(f"dataset augmentation, random_flip: {random_flip}, random_crop: {random_crop}, image_size: {self.image_size}.")

    def _find_matching_files(self, directory, index, with_params=False):
        """
        Find files in a directory that match a given index, with optional parameter matching.
        """
        if with_params:
            return sorted([f for f in os.listdir(directory) if f.startswith(f"{index}_")])
        else:
            return sorted([f for f in os.listdir(directory) if os.path.splitext(f)[0] == index])

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

                # for inter_folder_name in os.listdir(subdir_path):
                #     inter_folder = os.path.join(subdir_path, inter_folder_name)
                #     if not os.path.isdir(inter_folder):
                #         continue
                    
                #     for id_name in os.listdir(inter_folder):
                #         id_folder = os.path.join(inter_folder, id_name)
                #         if not os.path.isdir(id_folder):
                #             continue
            
            mask_img_path = os.path.join(subdir_path, 'mask.png') 
     
            albedo_dir = os.path.join(subdir_path, 'albedo')
            normal_dir = os.path.join(subdir_path, 'normal')
            #metallic_dir = os.path.join(id_folder, 'metallic')
            
            rgb_dir = os.path.join(subdir_path, 'rgb')
            #roughness_dir = os.path.join(id_folder, 'roughness')
            #breakpoint()
            if not all(os.path.exists(d) for d in [albedo_dir, normal_dir, rgb_dir]):
                continue

            albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
            
            for f in albedo_files:
                #breakpoint()
                index = os.path.splitext(f)[0]
                normal_files = self._find_matching_files(normal_dir, index)
                #metallic_files = self._find_matching_files(metallic_dir, index, with_params=True)
                rgb_files = self._find_matching_files(rgb_dir, index, with_params=True)
                #roughness_files = self._find_matching_files(roughness_dir, index, with_params=True)

                if normal_files  and rgb_files:
                    normal_file = os.path.join(normal_dir, normal_files[0])
                    #metallic_file = os.path.join(metallic_dir, metallic_files[0])
                    for rgb_file in rgb_files:
                        rgb_file = os.path.join(rgb_dir, rgb_file)
                    #rgb_file = os.path.join(rgb_dir, rgb_files[0])
                    #roughness_file = os.path.join(roughness_dir, roughness_files[0])

                        samples.append((
                            os.path.join(albedo_dir, f),
                            normal_file,
                            #metallic_file,
                            rgb_file,
                            #roughness_file
                            mask_img_path,
                        ))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def random_crop_arr(self, image, pad_width, crop_x, crop_y, pad_value=-1.):
        padded_image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=pad_value)
        return padded_image[:, crop_y : crop_y + self.image_size, crop_x : crop_x + self.image_size]

    def preprocess(self, image_path, random_crop_flag=False, random_flip_flag=False, crop=None, mask_path=None, albedo=False):
        image = np.array(Image.open(image_path).convert('RGB').resize(self.resize))
        

        if mask_path is not None:
            mask_img = np.array(Image.open(mask_path).resize(self.resize))
            mask = np.array(mask_img)[:, :, 3] > 128
            mask = mask.astype(bool)
            mask = np.stack([mask]*3, axis=-1)
            if albedo:
                masked_image = np.ones_like(image)*255
            else:
                masked_image = np.zeros_like(image)  # Start with an all-black image
            masked_image[mask] = image[mask]
            image = masked_image
        image = image.astype(np.float32) / 127.5 - 1.0 


        
        
        # if bool_op:
        #     bool_threshold = -0.97
        #     image[image<bool_threshold] = -1.
        #     unique_elements, counts = np.unique(image, return_counts=True)
        #     indices = np.argsort(-counts)[:2] # get top-2
        #     assert unique_elements[indices][0] == -1.
        #     try:
        #         image[image>=bool_threshold] = unique_elements[indices][1] # top2
        #     except:
        #         print(image_path)
        #         print(unique_elements[indices], counts[indices])
            
        image = np.transpose(image, [2, 0, 1])

        if random_crop_flag:
            assert crop is not None
            image = self.random_crop_arr(image, self.image_aug_pad_width, crop_x=crop['x'], crop_y=crop['y'])

        if random_flip_flag:
            image = image[:, ::-1]
            #image = torch.flip(image, [1])

        return image.copy()
    
    def create_from_name(self, image_path, mask_path):

        base_name = os.path.basename(image_path)
        metallic_num = float(base_name.split('_')[1])
        roughness_num = float(base_name.split('_')[2])
        

        # parts = base_name.split('_')

        # # Extract the metallic value directly
        # metallic_num = float(parts[1])

        # # Split the last part by '.' to separate '0.4' from 'png'
        # roughness_part = parts[2].split('.')[0]
        # roughness_num = float(roughness_part)
        
        image = np.array(Image.open(image_path).resize(self.resize))
        mask_img = np.array(Image.open(mask_path).resize(self.resize))

        #breakpoint()
        mask = torch.from_numpy(mask_img[..., 3:]).bool()
        metallic_init = torch.zeros(image.shape[0], image.shape[1], 1)
        roughness_init = torch.zeros(image.shape[0], image.shape[1], 1)
        
        metallic_init[mask] = metallic_num
        roughness_init[mask] = roughness_num
        
        #metallic_init = torch.cat((metallic_init,metallic_init,metallic_init), dim=-1)
        #roughness_init = torch.cat((roughness_init,roughness_init,roughness_init), dim=-1)
        # print(mask.int().view(-1).sum())
        # print(metallic_init.min(), metallic_init.max())

        mask = mask.to(dtype=roughness_init.dtype) * 2 - 1.0
        metallic = metallic_init * 2 - 1.0   # (0, 1) -> (-1, 1)
        roughness = roughness_init * 2 - 1.0
        return_img = torch.cat((metallic, roughness, mask), dim=-1)
        
        
        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        return return_img.permute(2, 0, 1)
        

        #metallic = metallic_init.reshape(self.resize)
        #roughness = roughness_init.reshape(self.resize)
     
        #return metallic.permute(2,0,1), roughness.permute(2,0,1)

    def __getitem__(self, idx):
        albedo_path, normal_path, rgb_path, mask_path = self.samples[idx]
        #breakpoint()
        light_image_path = '/hpc2hdd/home/zchen379/working/christmas_photo_studio_07_4k.png'
        
        material = self.create_from_name(rgb_path, mask_path)

        random_crop_flag = False
        crop = None
        if self.random_crop and random.random() < 0.8:
            padded_image = self.image_size + 2 * self.image_aug_pad_width[1][0]
            crop_y = random.randrange(padded_image - self.image_size + 1)
            crop_x = random.randrange(padded_image - self.image_size + 1)
            crop = {'x': crop_x, 'y': crop_y}

            material = torch.from_numpy(self.random_crop_arr(material, self.image_aug_pad_width, crop_x, crop_y)) # will cast into ndarray
            random_crop_flag = True    
            
        random_flip_flag = False
        if self.random_flip and random.random() < 0.4:
            material = torch.from_numpy(material.numpy()[:, ::-1].copy())
            random_flip_flag = True

        #print(random_crop_flag, random_flip_flag, crop)

        albedo = self.preprocess(albedo_path, random_crop_flag, random_flip_flag, crop=crop, mask_path=mask_path, albedo=True)
        normal = self.preprocess(normal_path, random_crop_flag, random_flip_flag, crop=crop, mask_path=mask_path)
        rgb = self.preprocess(rgb_path, random_crop_flag, random_flip_flag, crop=crop)
        light = self.preprocess(light_image_path, random_crop_flag, random_flip_flag, crop=crop)
        #print(albedo.shape, normal.shape, rgb.shape, light.shape)

        return {'albedo': albedo, 'normal': normal, 'material': material, 'rgb': rgb, 'light': light}
