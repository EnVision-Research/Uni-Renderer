import sys
from src.models.geometry.render import renderutils as ru
import torch
from src.models.geometry.render import util
import nvdiffrast.torch as dr
import os

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import imageio
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
LIGHT_MIN_RES = 16

MIN_ROUGHNESS = 0.04
MAX_ROUGHNESS = 1.00

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out
    
def build_mips(base, cutoff=0.99):
    specular = [base]
    while specular[-1].shape[1] > LIGHT_MIN_RES:
        specular.append(cubemap_mip.apply(specular[-1]))
        #specular.append(util.avg_pool_nhwc(specular[-1], (2,2)))

    diffuse = ru.diffuse_cubemap(specular[-1])

    for idx in range(len(specular) - 1):
        roughness = (idx / (len(specular) - 2)) * (MAX_ROUGHNESS - MIN_ROUGHNESS) + MIN_ROUGHNESS
        specular[idx] = ru.specular_cubemap(specular[idx], roughness, cutoff)
    specular[-1] = ru.specular_cubemap(specular[-1], 1.0, cutoff)

    return specular, diffuse

        
# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    specular, diffuse = build_mips(cubemap)

    return specular, diffuse

def main(path_hdr, save_path_map):
    all_envs = os.listdir(path_hdr)

    for env in all_envs:
        env_path = os.path.join(path_hdr, env)
        base_n = os.path.basename(env_path).split('.')[0]

        try:
            if not os.path.exists(os.path.join(save_path_map, base_n)):
                os.makedirs(os.path.join(save_path_map, base_n))
                specular, diffuse = _load_env_hdr(env_path)
                for i in range(len(specular)):
                    tensor = specular[i]
                    torch.save(tensor, os.path.join(save_path_map, base_n, f'specular_{i}.pth'))
                
                torch.save(diffuse, os.path.join(save_path_map, base_n, 'diffuse.pth'))
        except Exception as e:
            print(f"Error processing {env}: {e}")
            continue

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_hdr> <save_path_map>")
        sys.exit(1)
    
    path_hdr = sys.argv[1]
    save_path_map = sys.argv[2]

    if not os.path.exists(path_hdr):
        print(f"Error: path_hdr '{path_hdr}' does not exist.")
        sys.exit(1)

    if not os.path.exists(save_path_map):
        os.makedirs(save_path_map)

    main(path_hdr, save_path_map)