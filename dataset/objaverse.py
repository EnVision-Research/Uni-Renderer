import os, sys
import math
import json
import importlib
import glm
from pathlib import Path

import cv2
import torchvision
import random
import numpy as np
from PIL import Image
import webdataset as wds
# import pytorch_lightning as pl
import sys
sys.path.append('/hpc2hdd/home/zchen379/sd3/intrinsic-LRM')
from src.utils import obj, mesh, render_utils, render
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
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
    
def load_obj(path, return_attributes=False):
    return obj.load_obj(path, clear_ks=True, mtl_override=None, return_attributes=return_attributes)

def custom_collate_fn(batch):
    return batch


def collate_fn_wrapper(batch):
    return custom_collate_fn(batch)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='obj_demo',
        root_dir_new='',
        meta_fname='valid_paths.json',
        high_metallic=False,
        input_image_dir='rendering_random_32views',
        target_image_dir='rendering_random_32views',
        light_dir= 'data/env_mipmap/', # required
        input_view_num=1,
        target_view_num=1,
        total_view_n=18,
        fov=50,
        camera_rotation=True,
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir_new = Path(root_dir_new)
        self.input_image_dir = input_image_dir
        self.target_image_dir = target_image_dir
        self.light_dir = light_dir
        self.all_env_name = []
        for temp_dir in os.listdir(light_dir):
            if len(os.listdir(os.path.join(self.light_dir, temp_dir))) == 7:
                self.all_env_name.append(temp_dir)
        print(len(self.all_env_name))
        self.input_view_num = input_view_num
        self.target_view_num = target_view_num
        self.total_view_n = total_view_n
        self.fov = fov
        self.camera_rotation = camera_rotation
        
        self.train_res = [512, 512]
        self.cam_near_far = [0.1, 1000.0]
        self.fovy = np.deg2rad(50)
        self.spp = 1
        self.cam_radius = 3.5
        self.layers = 1
        
        #if random.random() < 0.5:
            # First block: 50% chance to execute this block
        numbers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.combinations = list(itertools.product(numbers, repeat=2))
        # elif random.random() > 0.75:
        #     # Second block: 25% chance to run high metallic 
        #     numbers_1 = [0.7, 0.8, 0.9, 1.0]
        #     numbers_2 = [0.0, 0.1, 0.2, 0.3]
        #     self.combinations = []
        #     for n1 in numbers_1:
        #         for n2 in numbers_2:
        #             self.combinations.append((n1, n2))
        # else: # Second block: 25% chance to run high roughness 
        #     numbers_1 = [0.0, 0.1, 0.2, 0.3]
        #     numbers_2 = [0.7, 0.8, 0.9, 1.0]
        #     self.combinations = []
        #     for n1 in numbers_1:
        #         for n2 in numbers_2:
        #             self.combinations.append((n1, n2))


        
        # self.paths = os.listdir(root_dir)
        with open('/hpc2hdd/home/zchen379/sd3/objaverse_data/Mesh_final_valid_texturemap.json') as f:
            filtered_dict = json.load(f)
            
        # with open('/hpc2hdd/home/zchen379/sd3/objaverse_data/Mesh_final_valid_texturemap.json') as f:
        #     filtered_dict_new = json.load(f)
  
        # paths = filtered_dict['good_objs']
        self.paths = filtered_dict
        self.num_old = len(filtered_dict)
        # self.paths.extend(filtered_dict_new)

        # with open("overfit_uids.json", 'r') as file:
        #     self.paths = json.load(file)
            
        #self.paths = list(set(self.paths) - set(error_paths))
        # self.paths = [name for name in filtered_dict if 'DS_Stor' not in name]
        # with open("Objaverse_highQuality_singleObj_OBJ_Mesh.json", 'r') as file:
        #     self.paths = json.load(file)
        print('total training object num:', len(self.paths))
        
        self.depth_scale = 6.0
            
        total_objects = len(self.paths)
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)
    
    def load_obj(self, path):
        return obj.load_obj(path, clear_ks=True, mtl_override=None)
    
    def sample_spherical(self, phi, theta):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)   

        z = self.cam_radius * np.cos(phi) * np.sin(theta)
        x = self.cam_radius * np.sin(phi) * np.sin(theta)
        y = self.cam_radius * np.cos(theta)
 
        return x, y, z
    
    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.train_res
        proj_mtx = render_utils.perspective(self.fovy, iter_res[1] / iter_res[0], self.cam_near_far[0], self.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        azimuths = 0 #random.uniform(0, 360)
        elevations = 90 #random.uniform(30, 150)
        mv_embedding = spherical_camera_pose(azimuths, 90-elevations, self.cam_radius)
        # print(mv_embedding)
        x, y, z = self.sample_spherical(azimuths, elevations)
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

    def __getitem__(self, index):
        while True:
            # if index <= self.num_old:
            #     obj_path = os.path.join(self.root_dir, self.paths[index])
            # else:
            obj_path = os.path.join(self.root_dir_new, self.paths[index]+'.pth')
            if os.path.exists(obj_path):
                pose_list = []
                env_list = []
                material_list = []
                camera_pos = []
                c2w_list = []
                camera_embedding_list = []
          
                #selected_env = random.randint(0, len(self.all_env_name)-1)
                #materials = random.choice(self.combinations)
                # index = 0
                for _ in range(self.input_view_num + self.target_view_num):
                    mv, mvp, campos, mv_mebedding, iter_res, iter_spp = self._random_scene()
                    # if random_env:
                    selected_env = random.randint(0, len(self.all_env_name)-1)
                    env_path = os.path.join(self.light_dir, self.all_env_name[selected_env])
                    env = load_mipmap(env_path)
                    #if random_mr:
                    materials = random.choice(self.combinations)
                    pose_list.append(mvp)
                    camera_pos.append(campos)
                    c2w_list.append(mv)
                    env_list.append(env)
                    material_list.append(materials)
                    camera_embedding_list.append(mv_mebedding)
                break
            else:
                index = np.random.randint(0, len(self.paths))
                continue

        data = {
                'input_view_num': self.input_view_num,
                'target_view_num': self.target_view_num,
                'obj_path': obj_path,
                'pose_list': pose_list,
                'camera_pos': camera_pos,
                'c2w_list': c2w_list,
                'env_list': env_list,
                'material_list': material_list,
                'camera_embedding_list': camera_embedding_list,
                # 'ref_mesh': ref_mesh
                }
            
        return data


class ObjaverseData_test(Dataset):
    def __init__(self,
        root_dir='obj_demo',
        root_dir_new='',
        meta_fname='valid_paths.json',
        high_metallic=False,
        input_image_dir='rendering_random_32views',
        target_image_dir='rendering_random_32views',
        light_dir= 'data/env_mipmap/', # required
        input_view_num=50,
        target_view_num=50,
        total_view_n=18,
        fov=50,
        camera_rotation=True,
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir_new = Path(root_dir_new)
        self.input_image_dir = input_image_dir
        self.target_image_dir = target_image_dir
        self.light_dir = light_dir
        self.all_env_name = []
        for temp_dir in os.listdir(light_dir):
            if len(os.listdir(os.path.join(self.light_dir, temp_dir))) == 7:
                self.all_env_name.append(temp_dir)
        print(len(self.all_env_name))
        self.input_view_num = input_view_num
        self.target_view_num = target_view_num
        self.total_view_n = total_view_n
        self.fov = fov
        self.camera_rotation = camera_rotation
        
        self.train_res = [512, 512]
        self.cam_near_far = [0.1, 1000.0]
        self.fovy = np.deg2rad(50)
        self.spp = 1
        self.cam_radius = 3.5
        self.layers = 1
        
        #if random.random() < 0.5:
            # First block: 50% chance to execute this block
        numbers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.combinations = list(itertools.product(numbers, repeat=2))
        # elif random.random() > 0.75:
        #     # Second block: 25% chance to run high metallic 
        #     numbers_1 = [0.7, 0.8, 0.9, 1.0]
        #     numbers_2 = [0.0, 0.1, 0.2, 0.3]
        #     self.combinations = []
        #     for n1 in numbers_1:
        #         for n2 in numbers_2:
        #             self.combinations.append((n1, n2))
        # else: # Second block: 25% chance to run high roughness 
        #     numbers_1 = [0.0, 0.1, 0.2, 0.3]
        #     numbers_2 = [0.7, 0.8, 0.9, 1.0]
        #     self.combinations = []
        #     for n1 in numbers_1:
        #         for n2 in numbers_2:
        #             self.combinations.append((n1, n2))


        
        # self.paths = os.listdir(root_dir)
        with open('/hpc2hdd/home/zchen379/sd3/objaverse_data/test_ood_one.json') as f:
            filtered_dict = json.load(f)
            
        # with open('/hpc2hdd/home/zchen379/sd3/objaverse_data/Mesh_final_valid_texturemap.json') as f:
        #     filtered_dict_new = json.load(f)
  
        # paths = filtered_dict['good_objs']
        self.paths = filtered_dict
        self.num_old = len(filtered_dict)
        # self.paths.extend(filtered_dict_new)

        # with open("overfit_uids.json", 'r') as file:
        #     self.paths = json.load(file)
            
        #self.paths = list(set(self.paths) - set(error_paths))
        # self.paths = [name for name in filtered_dict if 'DS_Stor' not in name]
        # with open("Objaverse_highQuality_singleObj_OBJ_Mesh.json", 'r') as file:
        #     self.paths = json.load(file)
        print('total testing object num:', len(self.paths))
        
        self.depth_scale = 6.0
            
        total_objects = len(self.paths)
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)
    
    def load_obj(self, path):
        return obj.load_obj(path, clear_ks=True, mtl_override=None)
    
    def sample_spherical(self, phi, theta):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)   

        z = self.cam_radius * np.cos(phi) * np.sin(theta)
        x = self.cam_radius * np.sin(phi) * np.sin(theta)
        y = self.cam_radius * np.cos(theta)
 
        return x, y, z
    
    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.train_res
        proj_mtx = render_utils.perspective(self.fovy, iter_res[1] / iter_res[0], self.cam_near_far[0], self.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        azimuths = random.uniform(0, 360)
        elevations = random.uniform(0, 180)
        mv_embedding = spherical_camera_pose(azimuths, 90-elevations, self.cam_radius)
        # print(mv_embedding)
        x, y, z = self.sample_spherical(azimuths, elevations)
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

    def __getitem__(self, index):
        while True:
            # if index <= self.num_old:
            #     obj_path = os.path.join(self.root_dir, self.paths[index])
            # else:
            obj_path = os.path.join(self.root_dir_new, self.paths[index])
            #breakpoint()

            if os.path.exists(obj_path):
                pose_list = []
                env_list = []
                material_list = []
                camera_pos = []
                c2w_list = []
                camera_embedding_list = []
          
                selected_env = random.randint(0, len(self.all_env_name)-1)
                materials = random.choice(self.combinations)
                # index = 0
                for _ in range(self.input_view_num + self.target_view_num):
                    mv, mvp, campos, mv_mebedding, iter_res, iter_spp = self._random_scene()
                    # if random_env:
                    #selected_env = random.randint(0, len(self.all_env_name)-1)
                    env_path = os.path.join(self.light_dir, self.all_env_name[selected_env])
                    env = load_mipmap(env_path)
                    #if random_mr:
                    #materials = random.choice(self.combinations)
                    pose_list.append(mvp)
                    camera_pos.append(campos)
                    c2w_list.append(mv)
                    env_list.append(env)
                    material_list.append(materials)
                    camera_embedding_list.append(mv_mebedding)
                break
            else:
                index = np.random.randint(0, len(self.paths))
                continue
        
        data = {
                'input_view_num': self.input_view_num,
                'target_view_num': self.target_view_num,
                'obj_path': obj_path,
                'pose_list': pose_list,
                'camera_pos': camera_pos,
                'c2w_list': c2w_list,
                'env_list': env_list,
                'material_list': material_list,
                'camera_embedding_list': camera_embedding_list,
                # 'ref_mesh': ref_mesh
                }
            
        return data

if __name__ == '__main__':
    dataset = ObjaverseData()
    dataset.new(1)
