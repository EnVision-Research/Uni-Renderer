import os, sys
import math
import json
import glm
from pathlib import Path

import random
import numpy as np
from PIL import Image
import webdataset as wds
import pytorch_lightning as pl
import sys
from src.utils import obj, render_utils
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import random
import itertools
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import re

def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws

def find_matching_files(base_path, idx):
    formatted_idx = '%03d' % idx
    pattern = re.compile(r'^%s_\d+\.png$' % formatted_idx)
    matching_files = []
    
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            if pattern.match(filename):
                matching_files.append(filename)
                
    return os.path.join(base_path, matching_files[0])

def load_mipmap(env_path):
    diffuse_path = os.path.join(env_path, "diffuse.pth")
    diffuse = torch.load(diffuse_path, map_location=torch.device('cpu'))

    specular = []
    for i in range(6):
        specular_path = os.path.join(env_path, f"specular_{i}.pth")
        specular_tensor = torch.load(specular_path, map_location=torch.device('cpu'))
        specular.append(specular_tensor)
    return [specular, diffuse]

def convert_to_white_bg(image, write_bg=True):
    alpha = image[:, :, 3:]
    if write_bg:
        return image[:, :, :3] * alpha + 1. * (1 - alpha)
    else:
        return image[:, :, :3] * alpha
    
def load_obj(path, return_attributes=False, scale_factor=1.0):
    return obj.load_obj(path, clear_ks=True, mtl_override=None, return_attributes=return_attributes, scale_factor=scale_factor)

def custom_collate_fn(batch):
    return batch


def collate_fn_wrapper(batch):
    return custom_collate_fn(batch)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test

    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError
    
    def custom_collate_fn(self, batch):
        collated_batch = {}
        for key in batch[0].keys():
            if key == 'input_env' or key == 'target_env':
                collated_batch[key] = [d[key] for d in batch]
            else:
                collated_batch[key] = torch.stack([d[key] for d in batch], dim=0)
        return collated_batch
    
    def convert_to_white_bg(self, image):
        alpha = image[:, :, 3:]
        return image[:, :, :3] * alpha + 1. * (1 - alpha)
    
    def load_obj(self, path):
        return obj.load_obj(path, clear_ks=True, mtl_override=None)

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler, collate_fn=collate_fn_wrapper)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=1, num_workers=self.num_workers, shuffle=False, sampler=sampler, collate_fn=collate_fn_wrapper)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='Objaverse_highQuality',
        light_dir= 'env_mipmap',
        input_view_num=6,
        target_view_num=4,
        total_view_n=18,
        distance=3.5,
        fov=50,
        camera_random=False,
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.light_dir = light_dir
        self.all_env_name = []
        for temp_dir in os.listdir(light_dir):
            if os.listdir(os.path.join(self.light_dir, temp_dir)):
                self.all_env_name.append(temp_dir)

        self.input_view_num = input_view_num
        self.target_view_num = target_view_num
        self.total_view_n = total_view_n
        self.fov = fov
        self.camera_random = camera_random
        
        self.train_res = [512, 512]
        self.cam_near_far = [0.1, 1000.0]
        self.fov_rad = np.deg2rad(fov)
        self.fov_deg = fov
        self.spp = 1
        self.cam_radius = distance
        self.layers = 1
        
        numbers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.combinations = list(itertools.product(numbers, repeat=2))
        
        self.paths = os.listdir(self.root_dir)
        
        # with open("BJ_Mesh_list.json", 'r') as file:
        #     self.paths = json.load(file)

        print('total training object num:', len(self.paths))

        self.depth_scale = 6.0
            
        total_objects = len(self.paths)
        print('============= length of dataset %d =============' % total_objects)

    def __len__(self):
        return len(self.paths)
    
    def load_obj(self, path):
        return obj.load_obj(path, clear_ks=True, mtl_override=None)
    
    def sample_spherical(self, phi, theta, cam_radius):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)   

        z = cam_radius * np.cos(phi) * np.sin(theta)
        x = cam_radius * np.sin(phi) * np.sin(theta)
        y = cam_radius * np.cos(theta)
 
        return x, y, z
    
    def _random_scene(self, cam_radius, fov_rad):
        iter_res = self.train_res
        proj_mtx = render_utils.perspective(fov_rad, iter_res[1] / iter_res[0], self.cam_near_far[0], self.cam_near_far[1])

        azimuths = random.uniform(0, 360)
        elevations = random.uniform(30, 150)
        mv_embedding = spherical_camera_pose(azimuths, 90-elevations, cam_radius)
        x, y, z = self.sample_spherical(azimuths, elevations, cam_radius)
        eye = glm.vec3(x, y, z)
        at = glm.vec3(0.0, 0.0, 0.0)
        up = glm.vec3(0.0, 1.0, 0.0)
        view_matrix = glm.lookAt(eye, at, up)
        mv = torch.from_numpy(np.array(view_matrix))
        mvp    = proj_mtx @ (mv)  #w2c
        campos = torch.linalg.inv(mv)[:3, 3]
        return mv[None, ...], mvp[None, ...], campos[None, ...], mv_embedding[None, ...], iter_res, self.spp # Add batch dimension
        
    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_albedo(self, path, color, mask):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        color = torch.ones_like(image)
        image = image * mask + color * (1 - mask)
        return image
    
    def convert_to_white_bg(self, image):
        alpha = image[:, :, 3:]
        return image[:, :, :3] * alpha + 1. * (1 - alpha)

    def calculate_fov(self, initial_distance, initial_fov, new_distance):
        initial_fov_rad = math.radians(initial_fov)
        
        height = 2 * initial_distance * math.tan(initial_fov_rad / 2)
        
        new_fov_rad = 2 * math.atan(height / (2 * new_distance))
        
        new_fov = math.degrees(new_fov_rad)
        
        return new_fov
    
    def __getitem__(self, index):
        obj_path = os.path.join(self.root_dir, self.paths[index])
        mesh_attributes = torch.load(obj_path, map_location=torch.device('cpu'))
        pose_list = []
        env_list = []
        material_list = []
        camera_pos = []
        c2w_list = []
        camera_embedding_list = []
        random_env = False
        random_mr = False
        if random.random() > 0.5:
            random_env = True
        if random.random() > 0.5:
            random_mr = True
        selected_env = random.randint(0, len(self.all_env_name)-1)
        materials = random.choice(self.combinations)
        if self.camera_random:
            random_perturbation = random.uniform(-1.5, 1.5)
            cam_radius = self.cam_radius + random_perturbation
            fov_deg = self.calculate_fov(initial_distance=self.cam_radius, initial_fov=self.fov_deg, new_distance=cam_radius)
            fov_rad = np.deg2rad(fov_deg)
        else:
            cam_radius = self.cam_radius
            fov_rad = self.fov_rad
            fov_deg = self.fov_deg

        if len(self.input_view_num) >= 1:
            input_view_num = random.choice(self.input_view_num)
        else:
            input_view_num = self.input_view_num
        for _ in range(input_view_num + self.target_view_num):
            mv, mvp, campos, mv_mebedding, iter_res, iter_spp = self._random_scene(cam_radius, fov_rad)
            if random_env:
                selected_env = random.randint(0, len(self.all_env_name)-1)
            env_path = os.path.join(self.light_dir, self.all_env_name[selected_env])
            env = load_mipmap(env_path)
            if random_mr:
                materials = random.choice(self.combinations)
            pose_list.append(mvp)
            camera_pos.append(campos)
            c2w_list.append(mv)
            env_list.append(env)
            material_list.append(materials)
            camera_embedding_list.append(mv_mebedding)
        data = {
            'mesh_attributes': mesh_attributes,
            'input_view_num': input_view_num,
            'target_view_num': self.target_view_num,
            'obj_path': obj_path,
            'pose_list': pose_list,
            'camera_pos': camera_pos,
            'c2w_list': c2w_list,
            'env_list': env_list,
            'material_list': material_list,
            'camera_embedding_list': camera_embedding_list,
            'fov_deg':fov_deg,
            'raduis': cam_radius
        }
        
        return data

class ValidationData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        input_view_num=6,
        input_image_size=320,
        fov=30,
    ):
        self.root_dir = Path(root_dir)
        self.input_view_num = input_view_num
        self.input_image_size = input_image_size
        self.fov = fov
        self.light_dir = 'env_mipmap'

        # with open('Mesh_list.json') as f:
        #     filtered_dict = json.load(f)
        
        self.paths = os.listdir(self.root_dir)
            
        # self.paths = filtered_dict
        print('============= length of dataset %d =============' % len(self.paths))

        cam_distance = 4.0
        azimuths = np.array([30, 90, 150, 210, 270, 330])
        elevations = np.array([20, -10, 20, -10, 20, -10])
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)

        x = cam_distance * np.cos(elevations) * np.cos(azimuths)
        y = cam_distance * np.cos(elevations) * np.sin(azimuths)
        z = cam_distance * np.sin(elevations)

        cam_locations = np.stack([x, y, z], axis=-1)
        cam_locations = torch.from_numpy(cam_locations).float()
        c2ws = center_looking_at_camera_pose(cam_locations)
        self.c2ws = c2ws.float()
        self.Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(6, 1, 1).float()

        render_c2ws = get_circular_camera_poses(M=8, radius=cam_distance, elevation=20.0)
        render_Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(render_c2ws.shape[0], 1, 1)
        self.render_c2ws = render_c2ws.float()
        self.render_Ks = render_Ks.float()

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_mat(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((384,384), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_albedo(self, path, color, mask):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        color = torch.ones_like(image)
        image = image * mask + color * (1 - mask)
        return image
    
    def __getitem__(self, index):
        
        # load data
        input_image_path = os.path.join(self.root_dir, self.paths[index])

        '''background color, default: white'''
        bkg_color = [1.0, 1.0, 1.0]

        image_list = []
        albedo_list = []
        alpha_list = []
        specular_list = []
        diffuse_list = []
        metallic_list = []
        roughness_list = []
        
        exist_comb_list = []
        for subfolder in os.listdir(input_image_path):
            found_numeric_subfolder=False
            subfolder_path = os.path.join(input_image_path, subfolder)
            if os.path.isdir(subfolder_path) and '_' in subfolder and 'specular' not in subfolder and 'diffuse' not in subfolder:
                try:
                    parts = subfolder.split('_')
                    float(parts[0])  # 尝试将分隔符前后的字符串转换为浮点数
                    float(parts[1])
                    found_numeric_subfolder = True
                except ValueError:
                    continue
            if found_numeric_subfolder:
                exist_comb_list.append(subfolder)
                
        selected_one_comb = random.choice(exist_comb_list)


        for idx in range(self.input_view_num):
            img_path = find_matching_files(os.path.join(input_image_path, selected_one_comb, 'rgb'), idx)
            albedo_path = img_path.replace('rgb', 'albedo')
            metallic_path = img_path.replace('rgb', 'metallic')
            roughness_path = img_path.replace('rgb', 'roughness')
            
            image, alpha = self.load_im(img_path, bkg_color)
            albedo = self.load_albedo(albedo_path, bkg_color, alpha)
            metallic,_ = self.load_mat(metallic_path, bkg_color)
            roughness,_ = self.load_mat(roughness_path, bkg_color)
            
            light_num = os.path.basename(img_path).split('_')[1].split('.')[0]
            light_path = os.path.join(self.light_dir, str(int(light_num)+1))
            
            specular, diffuse = load_mipmap(light_path)
            
            image_list.append(image)
            alpha_list.append(alpha)
            albedo_list.append(albedo)
            metallic_list.append(metallic)
            roughness_list.append(roughness)
            specular_list.append(specular)
            diffuse_list.append(diffuse)
        
        images = torch.stack(image_list, dim=0).float()
        alphas = torch.stack(alpha_list, dim=0).float()
        albedo = torch.stack(albedo_list, dim=0).float()    
        metallic = torch.stack(metallic_list, dim=0).float()    
        roughness = torch.stack(roughness_list, dim=0).float() 

        data = {
            'input_images': images,
            'input_alphas': alphas,
            'input_c2ws': self.c2ws,
            'input_Ks': self.Ks,
            
            'input_albedos': albedo[:self.input_view_num], 
            'input_metallics': metallic[:self.input_view_num], 
            'input_roughness': roughness[:self.input_view_num], 
            
            'specular': specular_list[:self.input_view_num],
            'diffuse': diffuse_list[:self.input_view_num],

            'render_c2ws': self.render_c2ws,
            'render_Ks': self.render_Ks,
        }
        return data


if __name__ == '__main__':
    dataset = ObjaverseData()
    dataset.new(1)
