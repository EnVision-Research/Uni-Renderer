import torch
import torch.nn as nn
from torchvision import transforms
from utils_metrics.inception import InceptionV3
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import linalg
import pickle
import os
import logging
from ipdb import set_trace

# def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
#     cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

#     if not np.isfinite(cov_sqrt).all():
#         print("product of cov matrices is singular")
#         offset = np.eye(sample_cov.shape[0]) * eps
#         cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

#     if np.iscomplexobj(cov_sqrt):
#         if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
#             m = np.max(np.abs(cov_sqrt.imag))

#             raise ValueError(f"Imaginary component {m}")

#         cov_sqrt = cov_sqrt.real

#     mean_diff = sample_mean - real_mean
#     mean_norm = mean_diff @ mean_diff

#     trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

#     fid = mean_norm + trace

#     return fid

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def extract_features_from_samples(model, images, batch_size, device):
    '''
    model: inception v3 model
    images: Image
    '''
    # convert them to tensor
    fid_images_tensor = []
    for image in images:
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2,0,1).to(device) # CHW
        fid_images_tensor.append(image)

    # prepare batch
    num_images = len(images)
    num_batches = num_images // batch_size
    num_res = num_images % batch_size
    batch_images = []
    for b in range(num_batches):
        batch = fid_images_tensor[b*batch_size:(b+1)*batch_size]
        batch_images.append(torch.stack(batch, dim=0))
    if num_res > 0:
        last_batch = fid_images_tensor[num_images - num_res:]
        batch_images.append(torch.stack(last_batch, dim=0))

    # extract sample image featrues
    features = []
    for batch in batch_images:
        feat = model(batch)[0].view(batch.shape[0], -1)
        features.append(feat.to("cpu"))
    features = torch.cat(features, 0).numpy()

    return features

def calculate_fid(pipeline, prompt, height, width, num_images_fid, num_inference_steps, generator, accelerator, real_path, output_dir, step, batch_size=20):
    fid_outdir = os.path.join(output_dir, "fids")
    os.makedirs(fid_outdir, exist_ok=True)

    device = 'cuda:0'

    # prepare model
    inception = InceptionV3([3], normalize_input=False)
    # inception = accelerator.prepare(inception)
    inception = nn.DataParallel(inception).to(device)

    # generate images (PIL Image)
    n_batch = num_images_fid // batch_size
    resid = num_images_fid - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid] if resid > 0 else [batch_size] * n_batch

    fake_images = []
    for batch in tqdm(batch_sizes):
        with torch.autocast("cuda"):
            image = pipeline([prompt]*batch, height, width, num_inference_steps=num_inference_steps, generator=generator).images
        fake_images.extend(image)

    # extract sample features
    fake_features = extract_features_from_samples(inception, fake_images, batch_size, device)
    sample_mean = np.mean(fake_features, 0)
    sample_cov = np.cov(fake_features, rowvar=False)
    sample_statistic = {
        "mean": sample_mean,
        "cov": sample_cov
    }
    logging.info(f"Saving sample statistic into {os.path.join(fid_outdir, f'{step}.pkl')}")
    with open(os.path.join(fid_outdir, f'{step}.pkl'), 'wb') as f:
        pickle.dump(sample_statistic, f)

    # extract / load real features
    if os.path.exists(os.path.join(fid_outdir, f'real.pkl')):
        with open(os.path.join(fid_outdir, f'real.pkl'), "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
        logging.info(f"Loading real statistic from {os.path.join(fid_outdir, f'{step}.pkl')}")
    else:
        real_image_names = [os.path.join(real_path, f) for f in os.listdir(real_path)][:num_images_fid]
        real_images = [Image.open(img).convert("RGB") for img in real_image_names]
        real_features = extract_features_from_samples(inception, real_images, batch_size, device)

        real_mean = np.mean(real_features, 0)
        real_cov = np.cov(real_features, rowvar=False)
        real_statistic = {
            "mean": real_mean,
            "cov": real_cov
        }

        with open(os.path.join(fid_outdir, f'real.pkl'), 'wb') as f:
            pickle.dump(real_statistic, f)
        logging.info(f"Saving real statistic into {os.path.join(fid_outdir, f'{step}.pkl')}")

    # calculate fid
    fid = calculate_frechet_distance(sample_mean, sample_cov, real_mean, real_cov)

    del fake_images
    torch.cuda.empty_cache()
    return fid

    
    
    