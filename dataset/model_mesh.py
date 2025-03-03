import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_lightning as pl
from einops import rearrange, repeat
from src.data.light import _load_env_hdr
from src.utils.camera_util import FOV_to_intrinsics
from src.utils.material import Material
from src.utils.train_util import instantiate_from_config
import nvdiffrast.torch as dr
from src.utils import obj, render_utils, render
from src.utils.mesh import Mesh, compute_tangents

# from pytorch3d.transforms import quaternion_to_matrix, euler_angles_to_matrix
mesh_save_root = "obj_good_mesh"
GLCTX = [None] * torch.cuda.device_count()  # 存储每个 GPU 的上下文

def initialize_extension(gpu_id):
    global GLCTX
    if GLCTX[gpu_id] is None:
        print(f"Initializing extension module renderutils_plugin on GPU {gpu_id}...")
        torch.cuda.set_device(gpu_id)
        GLCTX[gpu_id] = dr.RasterizeCudaContext()
    return GLCTX[gpu_id]

# Regulrarization loss for FlexiCubes
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1, 0, 0, 0], 
                         [0, c,-s, 0], 
                         [0, s, c, 0], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)


def convert_to_white_bg(image, write_bg=True):
    alpha = image[:, :, 3:]
    if write_bg:
        return image[:, :, :3] * alpha + 1. * (1 - alpha)
    else:
        return image[:, :, :3] * alpha
    

class MVRecon(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        input_size=256,
        render_size=512,
        init_ckpt=None,
    ):
        super(MVRecon, self).__init__()

        self.input_size = input_size
        self.render_size = render_size
        # gpu_id = torch.cuda.current_device()  # 获取当前线程的 GPU ID
        # initialize_extension(gpu_id)
        # self.GLCTX = dr.RasterizeCudaContext()
        # init modules
        self.lrm_generator = instantiate_from_config(lrm_generator_config)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        # Load weights from pretrained MVRecon model, and use the mlp 
        # weights to initialize the weights of sdf and rgb mlps.

        if init_ckpt is not None:
            sd = torch.load(init_ckpt, map_location='cpu')['state_dict']
            # breakpoint()
            sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
            sd_fc = {}
            for k, v in sd.items():
                if k.startswith('lrm_generator.synthesizer.decoder.net.'):
                    if k.startswith('lrm_generator.synthesizer.decoder.net.6.'):    # last layer
                        # Here we assume the density filed's isosurface threshold is t, 
                        # we reverse the sign of density filed to initialize SDF field.  
                        # -(w*x + b - t) = (-w)*x + (t - b)
                        if 'weight' in k:
                            sd_fc[k.replace('net.', 'net_sdf.')] = -v[0:1]
                        else:
                            sd_fc[k.replace('net.', 'net_sdf.')] = 10.0 - v[0:1]
                        sd_fc[k.replace('net.', 'net_rgb.')] = v[1:4]
                    else:
                        sd_fc[k.replace('net.', 'net_sdf.')] = v
                        sd_fc[k.replace('net.', 'net_rgb.')] = v
                else:
                    sd_fc[k] = v
            sd_fc = {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}
            # missing `net_deformation` and `net_weight` parameters
            self.lrm_generator.load_state_dict(sd_fc, strict=False)
            print(f'Loaded weights from {init_ckpt}')
        
        self.validation_step_outputs = []
    
    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.lrm_generator.init_flexicubes_geometry(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)

    def collate_fn(self, batch):
        gpu_id = torch.cuda.current_device()  # 获取当前线程的 GPU ID
        glctx = initialize_extension(gpu_id)
        batch_size = len(batch)
        input_view_num = batch[0]["input_view_num"]
        target_view_num = batch[0]["target_view_num"]
        iter_res = [512, 512]
        iter_spp = 1
        layers = 1

        # Initialize lists for input and target data
        input_images, input_alphas, input_depths, input_normals, input_albedos = [], [], [], [], []
        input_spec_light, input_diff_light, input_spec_albedo,input_diff_albedo = [], [], [], []
        input_w2cs, input_Ks, input_camera_pos, input_c2ws = [], [], [], []
        input_env, input_materials = [], []
        input_camera_embeddings = []    # camera_embedding_list

        target_images, target_alphas, target_depths, target_normals, target_albedos = [], [], [], [], []
        target_spec_light, target_diff_light, target_spec_albedo, target_diff_albedo = [], [], [], []
        target_w2cs, target_Ks, target_camera_pos = [], [], []
        target_env, target_materials = [], []

        for sample in batch:
            obj_path = sample['obj_path']
            mtl_path = os.path.splitext(sample['obj_path'])[0] + '.mtl'
            # try:
            with torch.no_grad():
                # mesh_attributes = torch.load(f"{mesh_save_root}/{os.path.basename(obj_path)[:-4]}", map_location=torch.device('cpu'))
                # vertices = mesh_attributes["vertices"].to(self.device)
                # faces = mesh_attributes["faces"].to(self.device)
                # normals = mesh_attributes["normals"].to(self.device)
                # nfaces = mesh_attributes["nfaces"].to(self.device)
                # texcoords = mesh_attributes["texcoords"].to(self.device)
                # tfaces = mesh_attributes["tfaces"].to(self.device)
                # material = Material(mesh_attributes["mat_dict"])
                # material = material.to(self.device)
                # ref_mesh = Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=material)
                # ref_mesh = compute_tangents(ref_mesh)
                mesh_attributes = torch.load(obj_path, map_location=torch.device('cpu'))
                v_pos = mesh_attributes["v_pos"].to(self.device)
                v_nrm = mesh_attributes["v_nrm"].to(self.device)
                v_tex = mesh_attributes["v_tex"].to(self.device)
                v_tng = mesh_attributes["v_tng"].to(self.device)
                t_pos_idx = mesh_attributes["t_pos_idx"].to(self.device)
                t_nrm_idx = mesh_attributes["t_nrm_idx"].to(self.device)
                t_tex_idx = mesh_attributes["t_tex_idx"].to(self.device)
                t_tng_idx = mesh_attributes["t_tng_idx"].to(self.device)
                material = Material(mesh_attributes["mat_dict"])
                material = material.to(self.device)
                ref_mesh = Mesh(v_pos=v_pos, v_nrm=v_nrm, v_tex=v_tex, v_tng=v_tng, 
                                t_pos_idx=t_pos_idx, t_nrm_idx=t_nrm_idx, 
                                t_tex_idx=t_tex_idx, t_tng_idx=t_tng_idx, material=material)
                
            pose_list_sample = sample['pose_list']  # mvp
            camera_pos_sample = sample['camera_pos'] # campos, mv.inverse
            c2w_list_sample = sample['c2w_list']    # mv
            env_list_sample = sample['env_list']
            material_list_sample = sample['material_list']
            camera_embeddings = sample["camera_embedding_list"]


            sample_input_images, sample_input_alphas, sample_input_depths, sample_input_normals, sample_input_albedos = [], [], [], [], []
            sample_input_w2cs, sample_input_Ks, sample_input_camera_pos, sample_input_c2ws = [], [], [], []
            sample_input_camera_embeddings = []
            sample_input_spec_light, sample_input_diff_light, sample_input_spec_albedo, sample_input_diff_albedo = [], [], [], []

            sample_target_images, sample_target_alphas, sample_target_depths, sample_target_normals, sample_target_albedos = [], [], [], [], []
            sample_target_w2cs, sample_target_Ks, sample_target_camera_pos = [], [], []
            sample_target_spec_light, sample_target_diff_light, sample_target_spec_albedo, sample_target_diff_albedo = [], [], [], []

            sample_input_env = []
            sample_input_materials = []
            sample_target_env = []
            sample_target_materials = []

            for i in range(len(pose_list_sample)):
                mvp = pose_list_sample[i]
                campos = camera_pos_sample[i]
                env = env_list_sample[i]
                materials = material_list_sample[i]
                camera_embedding = camera_embeddings[i]
                # breakpoint()
                with torch.no_grad():
                    buffer_dict = render.render_mesh(glctx, ref_mesh, mvp.to(self.device), campos.to(self.device), [env], None, None, 
                                                    materials, iter_res, spp=iter_spp, num_layers=layers, msaa=True, 
                                                    background=None, gt_render=True)

                image = convert_to_white_bg(buffer_dict['shaded'][0])
                albedo = convert_to_white_bg(buffer_dict['albedo'][0]).clamp(0., 1.)
                alpha = buffer_dict['mask'][0][:, :, 3:]  
                depth = convert_to_white_bg(buffer_dict['depth'][0])
                normal = convert_to_white_bg(buffer_dict['normal'][0], write_bg=False)
                spec_light = convert_to_white_bg(buffer_dict['spec_light'][0])
                diff_light = convert_to_white_bg(buffer_dict['diff_light'][0])
                # spec_albedo = convert_to_white_bg(buffer_dict['spec_albedo'][0])
                # diff_albedo = convert_to_white_bg(buffer_dict['diff_albedo'][0])

                if i < input_view_num:
                    sample_input_images.append(image)
                    sample_input_albedos.append(albedo)
                    sample_input_alphas.append(alpha)
                    sample_input_depths.append(depth)
                    sample_input_normals.append(normal)
                    sample_input_spec_light.append(spec_light)
                    sample_input_diff_light.append(diff_light)
                    # sample_input_spec_albedo.append(spec_albedo)
                    # sample_input_diff_albedo.append(diff_albedo)
                    sample_input_w2cs.append(mvp)
                    sample_input_camera_pos.append(campos)
                    sample_input_c2ws.append(c2w_list_sample[i])
                    sample_input_Ks.append(FOV_to_intrinsics(50))
                    sample_input_env.append(env)
                    sample_input_materials.append(materials)
                    sample_input_camera_embeddings.append(camera_embedding)
                else:
                    sample_target_images.append(image)
                    sample_target_albedos.append(albedo)
                    sample_target_alphas.append(alpha)
                    sample_target_depths.append(depth)
                    sample_target_normals.append(normal)
                    sample_target_spec_light.append(spec_light)
                    sample_target_diff_light.append(diff_light)
                    # sample_target_spec_albedo.append(spec_albedo)
                    # sample_target_diff_albedo.append(diff_albedo)
                    sample_target_w2cs.append(mvp)
                    sample_target_camera_pos.append(campos)
                    sample_target_Ks.append(FOV_to_intrinsics(50))
                    sample_target_env.append(env)
                    sample_target_materials.append(materials)

            input_images.append(torch.stack(sample_input_images, dim=0).permute(0, 3, 1, 2))
            input_albedos.append(torch.stack(sample_input_albedos, dim=0).permute(0, 3, 1, 2))
            input_alphas.append(torch.stack(sample_input_alphas, dim=0).permute(0, 3, 1, 2))
            input_depths.append(torch.stack(sample_input_depths, dim=0).permute(0, 3, 1, 2))
            input_normals.append(torch.stack(sample_input_normals, dim=0).permute(0, 3, 1, 2))
            input_spec_light.append(torch.stack(sample_input_spec_light, dim=0).permute(0, 3, 1, 2))
            input_diff_light.append(torch.stack(sample_input_diff_light, dim=0).permute(0, 3, 1, 2))
            # input_spec_albedo.append(torch.stack(sample_input_spec_albedo, dim=0).permute(0, 3, 1, 2))
            # input_diff_albedo.append(torch.stack(sample_input_diff_albedo, dim=0).permute(0, 3, 1, 2))
            input_w2cs.append(torch.stack(sample_input_w2cs, dim=0))
            input_camera_pos.append(torch.stack(sample_input_camera_pos, dim=0))
            input_c2ws.append(torch.stack(sample_input_c2ws, dim=0))
            input_camera_embeddings.append(torch.stack(sample_input_camera_embeddings, dim=0))
            input_Ks.append(torch.stack(sample_input_Ks, dim=0))
            input_env.append(sample_input_env)
            input_materials.append(sample_input_materials)

            target_images.append(torch.stack(sample_target_images, dim=0).permute(0, 3, 1, 2))
            target_albedos.append(torch.stack(sample_target_albedos, dim=0).permute(0, 3, 1, 2))
            target_alphas.append(torch.stack(sample_target_alphas, dim=0).permute(0, 3, 1, 2))
            target_depths.append(torch.stack(sample_target_depths, dim=0).permute(0, 3, 1, 2))
            target_normals.append(torch.stack(sample_target_normals, dim=0).permute(0, 3, 1, 2))
            target_spec_light.append(torch.stack(sample_target_spec_light, dim=0).permute(0, 3, 1, 2))
            target_diff_light.append(torch.stack(sample_target_diff_light, dim=0).permute(0, 3, 1, 2))
            # target_spec_albedo.append(torch.stack(sample_target_spec_albedo, dim=0).permute(0, 3, 1, 2))
            # target_diff_albedo.append(torch.stack(sample_target_diff_albedo, dim=0).permute(0, 3, 1, 2))
            target_w2cs.append(torch.stack(sample_target_w2cs, dim=0))
            target_camera_pos.append(torch.stack(sample_target_camera_pos, dim=0))
            target_Ks.append(torch.stack(sample_target_Ks, dim=0))
            target_env.append(sample_target_env)
            target_materials.append(sample_target_materials)
        # breakpoint()
            del ref_mesh
            del material
            del mesh_attributes
            torch.cuda.empty_cache()
        data = {
            'input_images': torch.stack(input_images, dim=0).detach().cpu(),           # (batch_size, input_view_num, 3, H, W)
            'input_alphas': torch.stack(input_alphas, dim=0).detach().cpu(),           # (batch_size, input_view_num, 1, H, W) 
            'input_depths': torch.stack(input_depths, dim=0).detach().cpu(),  
            'input_normals': torch.stack(input_normals, dim=0).detach().cpu(), 
            'input_albedos': torch.stack(input_albedos, dim=0).detach().cpu(), 
            'input_spec_light': torch.stack(input_spec_light, dim=0).detach().cpu(), 
            'input_diff_light': torch.stack(input_diff_light, dim=0).detach().cpu(), 
            # 'input_spec_albedo': torch.stack(input_spec_albedo, dim=0), 
            # 'input_diff_albedo': torch.stack(input_diff_albedo, dim=0), 
            'input_materials': input_materials,
            'input_w2cs': torch.stack(input_w2cs, dim=0).squeeze(2),               # (batch_size, input_view_num, 4, 4)
            'input_Ks': torch.stack(input_Ks, dim=0).float(),                   # (batch_size, input_view_num, 3, 3)
            'input_env': input_env,
            'input_camera_pos': torch.stack(input_camera_pos, dim=0).squeeze(2),   # (batch_size, input_view_num, 3)
            'input_c2ws': torch.stack(input_c2ws, dim=0).squeeze(2),               # (batch_size, input_view_num, 4, 4)
            'input_camera_embedding': torch.stack(input_camera_embeddings, dim=0).squeeze(2),

            'target_images': torch.stack(target_images, dim=0).detach().cpu(),         # (batch_size, target_view_num, 3, H, W)
            'target_alphas': torch.stack(target_alphas, dim=0).detach().cpu(),         # (batch_size, target_view_num, 1, H, W)
            'target_depths': torch.stack(target_depths, dim=0).detach().cpu(),  
            'target_normals': torch.stack(target_normals, dim=0).detach().cpu(), 
            'target_albedos': torch.stack(target_albedos, dim=0).detach().cpu(), 
            'target_spec_light': torch.stack(target_spec_light, dim=0).detach().cpu(), 
            'target_diff_light': torch.stack(target_diff_light, dim=0).detach().cpu(), 
            # 'target_spec_albedo': torch.stack(target_spec_albedo, dim=0), 
            # 'target_diff_albedo': torch.stack(target_diff_albedo, dim=0), 
            'target_materials': target_materials,
            'target_w2cs': torch.stack(target_w2cs, dim=0).squeeze(2),             # (batch_size, target_view_num, 4, 4)
            'target_Ks': torch.stack(target_Ks, dim=0).float(),                 # (batch_size, target_view_num, 3, 3)
            'target_env': target_env,
            'target_camera_pos': torch.stack(target_camera_pos, dim=0).squeeze(2)  # (batch_size, target_view_num, 3)
        }

        return data
    def prepare_batch_data(self, batch):
        # breakpoint()
        lrm_generator_input = {}
        render_gt = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)
        batch_size = images.shape[0]
        # breakpoint()
        lrm_generator_input['images'] = images.to(self.device)

        # input cameras and render cameras
        # input_c2ws = batch['input_c2ws']
        input_Ks = batch['input_Ks']
        # target_c2ws = batch['target_c2ws']
        input_camera_embedding = batch["input_camera_embedding"].to(self.device)

        input_w2cs = batch['input_w2cs']
        target_w2cs = batch['target_w2cs']
        render_w2cs =  torch.cat([input_w2cs, target_w2cs], dim=1)
        
        input_camera_pos = batch['input_camera_pos']
        target_camera_pos = batch['target_camera_pos']
        render_camera_pos = torch.cat([input_camera_pos, target_camera_pos], dim=1)

        input_extrinsics = input_camera_embedding.flatten(-2)
        input_extrinsics = input_extrinsics[:, :, :12]
        input_intrinsics = input_Ks.flatten(-2).to(self.device)
        input_intrinsics = torch.stack([
            input_intrinsics[:, :, 0], input_intrinsics[:, :, 4], 
            input_intrinsics[:, :, 2], input_intrinsics[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        # add noise to input_cameras
        cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02

        lrm_generator_input['cameras'] = cameras.to(self.device)
        lrm_generator_input['render_cameras'] =  render_w2cs.to(self.device)# @ rotate_x(np.pi / 2, self.device) )
        lrm_generator_input['cameras_pos'] = render_camera_pos.to(self.device)   #torch.linalg.inv(render_w2cs.to(self.device) @ rotate_x(np.pi / 2, self.device))[..., :3, 3]
        lrm_generator_input['env'] = []
        lrm_generator_input['materials'] = []
        for i in range(batch_size):
            lrm_generator_input['env'].append( batch['input_env'][i] + batch['target_env'][i])
            lrm_generator_input['materials'].append( batch['input_materials'][i] +  batch['target_materials'][i]) 
        lrm_generator_input['albedo'] = torch.cat([batch['input_albedos'],batch['target_albedos']],dim=1) 
    
        # target images
        target_images = torch.cat([batch['input_images'], batch['target_images']], dim=1)
        target_albedos = torch.cat([batch['input_albedos'], batch['target_albedos']], dim=1)
        target_depths = torch.cat([batch['input_depths'], batch['target_depths']], dim=1)
        target_alphas = torch.cat([batch['input_alphas'], batch['target_alphas']], dim=1)
        target_normals = torch.cat([batch['input_normals'], batch['target_normals']], dim=1)
        target_spec_lights = torch.cat([batch['input_spec_light'], batch['target_spec_light']], dim=1)
        target_diff_lights = torch.cat([batch['input_diff_light'], batch['target_diff_light']], dim=1)
        # target_spec_albedos = torch.cat([batch['input_spec_albedo'], batch['target_spec_albedo']], dim=1)
        # target_diff_albedos = torch.cat([batch['input_diff_albedo'], batch['target_diff_albedo']], dim=1)

        render_size = self.render_size
        target_images = v2.functional.resize(
            target_images, render_size, interpolation=3, antialias=True).clamp(0, 1)
        target_depths = v2.functional.resize(
            target_depths, render_size, interpolation=0, antialias=True)
        target_alphas = v2.functional.resize(
            target_alphas, render_size, interpolation=0, antialias=True)
        target_normals = v2.functional.resize(
            target_normals, render_size, interpolation=3, antialias=True)

        lrm_generator_input['render_size'] = render_size

        render_gt['target_images'] = target_images.to(self.device)
        render_gt['target_albedos'] = target_albedos.to(self.device)
        render_gt['target_depths'] = target_depths.to(self.device)
        render_gt['target_alphas'] = target_alphas.to(self.device)
        render_gt['target_normals'] = target_normals.to(self.device)
        render_gt['target_spec_lights'] = target_spec_lights.to(self.device)
        render_gt['target_diff_lights'] = target_diff_lights.to(self.device)
        # render_gt['target_spec_albedos'] = target_spec_albedos.to(self.device)
        # render_gt['target_diff_albedos'] = target_diff_albedos.to(self.device)
     
        return lrm_generator_input, render_gt
    
    def prepare_validation_batch_data(self, batch):
        lrm_generator_input = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)
        lrm_generator_input['specular_light'] = batch['specular']
        lrm_generator_input['diffuse_light'] = batch['diffuse']
        
        lrm_generator_input['metallic'] = batch['input_metallics']
        lrm_generator_input['roughness'] = batch['input_roughness']

        proj = self.perspective(0.449, 1,  0.1, 1000., self.device)
        
        # input cameras
        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)
        
        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        lrm_generator_input['cameras'] = cameras.to(self.device)

        # render cameras
        render_c2ws = batch['render_c2ws']
        # render_w2cs = torch.linalg.inv(render_c2ws)
        
        lrm_generator_input['camera_pos'] =  torch.linalg.inv(render_w2cs.to(self.device) @ rotate_x(np.pi / 2, self.device))[..., :3, 3]
        render_w2cs = ( render_w2cs @ rotate_x(np.pi / 2) )

        lrm_generator_input['render_cameras'] = render_w2cs.to(self.device)
        lrm_generator_input['render_size'] = 384

        return lrm_generator_input
    
    def forward_lrm_generator(self, images, cameras, camera_pos,env, materials, albedo_map, render_cameras, render_size=512):
        planes = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        out = self.lrm_generator.forward_geometry(
            planes, 
            render_cameras, 
            camera_pos,
            env,
            materials,
            albedo_map,
            render_size,
        )
        return out
    
    def forward(self, lrm_generator_input):
        images = lrm_generator_input['images']
        cameras = lrm_generator_input['cameras']
        render_cameras = lrm_generator_input['render_cameras']
        render_size = lrm_generator_input['render_size']
        env = lrm_generator_input['env']
        materials = lrm_generator_input['materials']
        albedo_map = lrm_generator_input['albedo']
        camera_pos = lrm_generator_input['cameras_pos']

        out = self.forward_lrm_generator(
            images, cameras, camera_pos, env, materials, albedo_map, render_cameras, render_size=render_size)

        return out

    def training_step(self, batch, batch_idx):
        batch = self.collate_fn(batch)
        lrm_generator_input, render_gt = self.prepare_batch_data(batch)

        render_out = self.forward(lrm_generator_input)

        loss, loss_dict = self.compute_loss(render_out, render_gt)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=len(batch['input_images']), sync_dist=True)

        if self.global_step % 100 == 0 and self.global_rank == 0 :
            B, N, C, H, W = render_gt['target_images'].shape
            N_in = lrm_generator_input['images'].shape[1]

            target_images = rearrange(render_gt['target_images'], 'b n c h w -> b c h (n w)')
            render_images = rearrange(render_out['pbr_img'], 'b n c h w -> b c h (n w)')
            target_alphas = rearrange(repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            target_spec_light =  rearrange(render_gt['target_spec_lights'], 'b n c h w -> b c h (n w)') 
            target_diff_light =  rearrange(render_gt['target_diff_lights'], 'b n c h w -> b c h (n w)') 
            # target_spec_albedo =  rearrange(render_gt['target_spec_albedos'], 'b n c h w -> b c h (n w)') 
            # target_diff_albedo =  rearrange(render_gt['target_diff_albedos'], 'b n c h w -> b c h (n w)') 

            render_alphas = rearrange(render_out['mask'], 'b n c h w -> b c h (n w)')
            render_albodos =  rearrange(render_out['albedo'], 'b n c h w -> b c h (n w)')
            target_albedos = rearrange(render_gt['target_albedos'], 'b n c h w -> b c h (n w)')

            # render_spec_albodos =  rearrange(render_out['spec_albedo'], 'b n c h w -> b c h (n w)')
            # render_diff_albodos =  rearrange(render_out['diff_albedo'], 'b n c h w -> b c h (n w)')

            render_spec_light = rearrange(render_out['pbr_spec_light'], 'b n c h w -> b c h (n w)')
            render_diffuse_light = rearrange(render_out['pbr_diffuse_light'], 'b n c h w -> b c h (n w)')
            render_normal = rearrange(render_out['normal_img'], 'b n c h w -> b c h (n w)')
            target_depths = rearrange(render_gt['target_depths'], 'b n c h w -> b c h (n w)')
            render_depths = rearrange(render_out['depth'], 'b n c h w -> b c h (n w)')
            target_normals = rearrange(render_gt['target_normals'], 'b n c h w -> b c h (n w)')
            
            MAX_DEPTH = torch.max(target_depths)
            target_depths = target_depths / MAX_DEPTH * target_alphas
            render_depths = render_depths / MAX_DEPTH

            grid = torch.cat([
                target_images, render_images, 
                target_alphas, render_alphas, 
                target_albedos, render_albodos,
                target_spec_light, render_spec_light, 
                target_diff_light, render_diffuse_light,
                # target_spec_albedo, render_spec_albodos, 
                # target_diff_albedo, render_diff_albodos,
                target_normals, render_normal,
                target_depths, render_depths
            ], dim=-2).detach().cpu()
            grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))

            image_path = os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png')
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")
        return loss
    

    
    def total_variation_loss(self, img, beta=2.0):
        bs_img, n_view, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[...,1:,:]-img[...,:-1,:], beta).sum()
        tv_w = torch.pow(img[...,:,1:]-img[...,:,:-1], beta).sum()
        return (tv_h+tv_w)/(bs_img*n_view*c_img*h_img*w_img)
    
    def compute_loss(self, render_out, render_gt):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        # breakpoint()
        render_albedo_image = render_out['albedo']
        render_pbr_image = render_out['pbr_img']
        render_spec_light = render_out['pbr_spec_light']
        render_diff_light = render_out['pbr_diffuse_light']
        # render_metallic_images = render_out['metallic_img']
        # render_roughness_images = render_out['roughness_img']
        
        target_images = render_gt['target_images'].to(render_albedo_image)
        target_albedos = render_gt['target_albedos'].to(render_albedo_image)
        # target_metallic =  render_gt['target_metallics'].to(render_albedo_image)
        # target_roughness =  render_gt['target_roughness'].to(render_albedo_image)
        target_spec_light = render_gt['target_spec_lights'].to(render_albedo_image)
        target_diff_light = render_gt['target_diff_lights'].to(render_albedo_image)

        render_images = rearrange(render_pbr_image, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        
        render_albedos = rearrange(render_albedo_image, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_albedos = rearrange(target_albedos, 'b n ... -> (b n) ...') * 2.0 - 1.0

        render_spec_light = rearrange(render_spec_light, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_spec_light = rearrange(target_spec_light, 'b n ... -> (b n) ...') * 2.0 - 1.0

        render_diff_light = rearrange(render_diff_light, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_diff_light = rearrange(target_diff_light, 'b n ... -> (b n) ...') * 2.0 - 1.0

        # render_metallics = rearrange(render_metallic_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        # render_roughnesss = rearrange(render_roughness_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        # target_metallic = rearrange(target_metallic, 'b n ... -> (b n) ...') * 2.0 - 1.0
        # target_roughness = rearrange(target_roughness, 'b n ... -> (b n) ...') * 2.0 - 1.0
        
        loss_mse = F.mse_loss(render_images, target_images)
        loss_mse_albedo = F.mse_loss(render_albedos, target_albedos)
        loss_rgb_lpips = 2.0 * self.lpips(render_images, target_images)
        loss_albedo_lpips = 2.0 * self.lpips(render_albedos, target_albedos)

        loss_spec_light = F.mse_loss(render_spec_light, target_spec_light)
        loss_diff_light = F.mse_loss(render_diff_light, target_diff_light)

        loss_spec_light_lpips = 2.0 * self.lpips(render_spec_light.clamp(-1., 1.), target_spec_light.clamp(-1., 1.))
        loss_diff_light_lpips = 2.0 * self.lpips(render_diff_light, target_diff_light.clamp(-1., 1.))

        render_alphas = render_out['mask'][:,:,:1,:,:]
        target_alphas = render_gt['target_alphas']
 
        loss_mask = F.mse_loss(render_alphas, target_alphas)
        # breakpoint()
        render_depths = torch.mean(render_out['depth'], dim=2, keepdim=True)
        target_depths = torch.mean(render_gt['target_depths'], dim=2, keepdim=True)
        loss_depth = 0.5 * F.l1_loss(render_depths[(target_alphas>0)], target_depths[target_alphas>0])

        # 这里出来的时候本来就是【-1，1】
        render_normals = render_out['normal'][...,:3].permute(0,3,1,2).unsqueeze(0)
        target_normals = render_gt['target_normals']
        # breakpoint()
        similarity = (render_normals * target_normals).sum(dim=-3).abs()
        normal_mask = target_alphas.squeeze(-3)
        loss_normal = 1 - similarity[normal_mask>0].mean()
        loss_normal = 0.2 * loss_normal

        # tv loss
        triplane = render_out['triplane']
        tv_loss = self.total_variation_loss(triplane, beta=2.0)

        # flexicubes regularization loss
        sdf = render_out['sdf']
        sdf_reg_loss = render_out['sdf_reg_loss']
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.lrm_generator.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

        loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

        loss = loss_mse + loss_rgb_lpips + loss_albedo_lpips + loss_mask + loss_reg + loss_mse_albedo + loss_depth + \
            loss_normal + loss_spec_light + loss_diff_light + loss_spec_light_lpips + loss_diff_light_lpips + tv_loss * 2e-4

        prefix = 'train'
        loss_dict = {}

        loss_dict.update({f'{prefix}/loss_mse': loss_mse.item()})
        loss_dict.update({f'{prefix}/loss_mse_albedo': loss_mse_albedo.item()})
        loss_dict.update({f'{prefix}/loss_rgb_lpips': loss_rgb_lpips.item()})
        loss_dict.update({f'{prefix}/loss_albedo_lpips': loss_albedo_lpips.item()})
        loss_dict.update({f'{prefix}/loss_mask': loss_mask.item()})
        loss_dict.update({f'{prefix}/loss_normal': loss_normal.item()})
        loss_dict.update({f'{prefix}/loss_depth': loss_depth.item()})
        loss_dict.update({f'{prefix}/loss_spec_light': loss_spec_light.item()})
        loss_dict.update({f'{prefix}/loss_diff_light': loss_diff_light.item()})
        loss_dict.update({f'{prefix}/loss_spec_light_lpips': loss_spec_light_lpips.item()})
        loss_dict.update({f'{prefix}/loss_diff_light_lpips': loss_diff_light_lpips.item()})
        loss_dict.update({f'{prefix}/loss_reg_sdf': sdf_reg_loss_entropy.item()})
        loss_dict.update({f'{prefix}/loss_reg_surface': flexicubes_surface_reg.item()})
        loss_dict.update({f'{prefix}/loss_reg_weights': flexicubes_weights_reg.item()})
        loss_dict.update({f'{prefix}/loss_tv': tv_loss.item()})
        loss_dict.update({f'{prefix}/loss': loss.item()})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        lrm_generator_input = self.prepare_validation_batch_data(batch)

        render_out = self.forward(lrm_generator_input)
        render_images = rearrange(render_out['pbr_img'], 'b n c h w -> b c h (n w)')
        render_albodos =  rearrange(render_out['img'], 'b n c h w -> b c h (n w)')
        # render_matallics = rearrange(render_out['metallic_img'], 'b n c h w -> b c h (n w)').repeat(1, 3, 1, 1)
        # render_roughness = rearrange(render_out['roughness_img'], 'b n c h w -> b c h (n w)').repeat(1, 3, 1, 1)

        self.validation_step_outputs.append(render_images)
        self.validation_step_outputs.append(render_albodos)
        # self.validation_step_outputs.append(render_matallics)
        # self.validation_step_outputs.append(render_roughness)
    
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=0)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')

            # 每个样本的四张图竖直排列，因此 nrow 设置为1，使每张图独占一行
            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))

            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(
            self.lrm_generator.parameters(), lr=lr, betas=(0.90, 0.95), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100000, eta_min=0)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
