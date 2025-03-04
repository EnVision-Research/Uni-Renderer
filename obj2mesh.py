import json
import os
import torch
import psutil
import gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.objaverse import load_obj
from src.utils import mesh
from src.utils.material import Material
import argparse


def bytes_to_megabytes(bytes):
    return bytes / (1024 * 1024)


def bytes_to_gigabytes(bytes):
    return bytes / (1024 * 1024 * 1024)


def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(
        f"[{stage}] Process memory: {memory_info.rss / 1024**2:.2f} MB, "
        f"Allocated CUDA memory: {allocated:.2f} MB, Cached CUDA memory: {cached:.2f} MB"
    )


def process_obj(index, root_dir, final_save_dir, paths):
    obj_path = os.path.join(root_dir, paths[index], paths[index] + '.obj')
    mtl_path = os.path.join(root_dir, paths[index], paths[index] + '.mtl')

    if os.path.exists(os.path.join(final_save_dir, f"{paths[index]}.pth")):
        return None

    try:
        with torch.no_grad():
            ref_mesh, vertices, faces, normals, nfaces, texcoords, tfaces, uber_material = load_obj(
                obj_path, return_attributes=True
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ref_mesh = mesh.compute_tangents(ref_mesh)

        with open(mtl_path, 'r') as file:
            lines = file.readlines()

        if len(lines) >= 250:
            return None

        final_mesh_attributes = {
            "v_pos": ref_mesh.v_pos.detach().cpu(),
            "v_nrm": ref_mesh.v_nrm.detach().cpu(),
            "v_tex": ref_mesh.v_tex.detach().cpu(),
            "v_tng": ref_mesh.v_tng.detach().cpu(),
            "t_pos_idx": ref_mesh.t_pos_idx.detach().cpu(),
            "t_nrm_idx": ref_mesh.t_nrm_idx.detach().cpu(),
            "t_tex_idx": ref_mesh.t_tex_idx.detach().cpu(),
            "t_tng_idx": ref_mesh.t_tng_idx.detach().cpu(),
            "mat_dict": {key: ref_mesh.material[key] for key in ref_mesh.material.mat_keys},
        }

        torch.save(final_mesh_attributes, f"{final_save_dir}/{paths[index]}.pth")
        print(f"==> Saved to {final_save_dir}/{paths[index]}.pth")

        del ref_mesh
        torch.cuda.empty_cache()
        return paths[index]

    except Exception as e:
        print(f"Failed to process {paths[index]}: {e}")
        return None

    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main(root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    finish_lists = os.listdir(save_dir)
    paths = os.listdir(root_dir)

    valid_uid = []

    print_memory_usage("Start")

    batch_size = 100
    num_batches = (len(paths) + batch_size - 1) // batch_size

    for batch in tqdm(range(num_batches)):
        start_index = batch * batch_size
        end_index = min(start_index + batch_size, len(paths))

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_obj, index, root_dir, save_dir, paths)
                for index in range(start_index, end_index)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    valid_uid.append(result)

        print_memory_usage(f"=====> After processing batch {batch + 1}")
        torch.cuda.empty_cache()
        gc.collect()

    print_memory_usage("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OBJ files and save final results.")
    parser.add_argument("root_dir", type=str, help="Directory containing the root OBJ files.")
    parser.add_argument("save_dir", type=str, help="Directory to save the processed results.")
    args = parser.parse_args()

    main(args.root_dir, args.save_dir)
