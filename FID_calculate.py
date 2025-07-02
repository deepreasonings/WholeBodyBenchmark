#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import csv
from PIL import Image
import torch.nn.functional as F
from torchvision import models, transforms
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import scipy.linalg

# Create output directory if it doesn't exist
os.makedirs('FIDres', exist_ok=True)

# Constants
MODEL_FOLDERS = ['1', '2', '3', '4', '4-mp4', '5', '6']
METRICS = ['filename', 'FID', 'FVD', 'SSIM', 'PSNR', 'E-FID', 'CSIM']

# Configuration options
SKIP_FVD = False  # Set to False if you want to compute FVD (may cause errors)

# Load required models
print("Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load InceptionV3 model for FID and E-FID
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()
inception_model = inception_model.to(device)
inception_model.eval()

# Define transformation pipeline for InceptionV3
transform_pipeline = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load I3D model for FVD
try:
    i3d_model = torch.jit.load("eval/i3d_torchscript.pt").eval().to(device)
except Exception as e:
    print(f"Warning: Could not load I3D model: {e}")
    i3d_model = None

# Load groundtruth mapping
gt_mapping = {}
unmatched_mapping = {}  # 新增映射存储unmatched状态
skipped_count = 0  # 跟踪被跳过的视频数量

try:
    # 修改为读取新的CSV文件
    df = pd.read_csv('groundtruth_results.csv')
    for _, row in df.iterrows():
        model_num = str(row['模型号'])
        video_type = row['视频类型']
        original_path = row['原视频地址']
        gt_path = row['Groundtruth视频地址']
        
        # 读取unmatched状态
        unmatched = 0
        if 'unmatched' in row:
            unmatched = int(row['unmatched']) if not pd.isna(row['unmatched']) else 0
        
        # Create a unique key for each video
        key = f"{original_path}"
        gt_mapping[key] = gt_path
        unmatched_mapping[key] = unmatched  # 存储unmatched状态
except Exception as e:
    print(f"Error loading groundtruth mapping: {e}")

# Utility Functions for Video Processing
def get_video_frames(video_path, max_frames=16):
    """Extract frames from a video file using OpenCV."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []
        
        frames = []
        count = 0
        
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
        
        cap.release()
        
        if not frames:
            print(f"No frames extracted from video: {video_path}")
            # Create a dummy black frame if no frames were extracted
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames = [dummy_frame]
            
        return frames
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        return [dummy_frame]

def pil_from_frame(frame):
    """Convert a BGR frame (numpy array) to a PIL image in RGB."""
    if frame is None:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

# FID and E-FID Calculation Functions
def get_inception_features(frames, model, transform, device, batch_size=4):
    """Extract InceptionV3 features for a list of frames."""
    features = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_images = []
            
            for frame in batch_frames:
                try:
                    pil_img = pil_from_frame(frame)
                    if pil_img is None:
                        continue
                    img_tensor = transform(pil_img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            
            if not batch_images:
                continue
            
            try:    
                batch_tensor = torch.stack(batch_images).to(device)
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu().numpy())
                
                # Free up GPU memory
                del batch_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"CUDA out of memory. Trying with smaller batch...")
                    # Try with smaller batch
                    for single_img in batch_images:
                        try:
                            single_tensor = single_img.unsqueeze(0).to(device)
                            single_feature = model(single_tensor)
                            features.append(single_feature.cpu().numpy())
                            del single_tensor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    print(f"Error processing batch: {e}")
    
    if features:
        try:
            features = np.concatenate(features, axis=0)
        except ValueError as e:
            print(f"Error concatenating features: {e}")
            return np.array([])
    else:
        features = np.array([])
    
    return features

def get_edge_features(frames, model, transform, device, batch_size=4):
    """Extract InceptionV3 features for edge maps of frames."""
    features = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_images = []
            
            for frame in batch_frames:
                try:
                    # Convert to grayscale and apply Canny edge detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    pil_img = Image.fromarray(edges_rgb)
                    img_tensor = transform(pil_img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing edge frame: {e}")
            
            if not batch_images:
                continue
            
            try:    
                batch_tensor = torch.stack(batch_images).to(device)
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu().numpy())
                
                # Free up GPU memory
                del batch_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"CUDA out of memory for edge features. Trying with smaller batch...")
                    # Try with smaller batch
                    for single_img in batch_images:
                        try:
                            single_tensor = single_img.unsqueeze(0).to(device)
                            single_feature = model(single_tensor)
                            features.append(single_feature.cpu().numpy())
                            del single_tensor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    print(f"Error processing edge batch: {e}")
    
    if features:
        try:
            features = np.concatenate(features, axis=0)
        except ValueError as e:
            print(f"Error concatenating edge features: {e}")
            return np.array([])
    else:
        features = np.array([])
    
    return features

def calculate_statistics(features):
    """Compute mean and covariance of features."""
    if features.size == 0:
        print("Empty features array")
        return None, None
    
    try:
        mu = np.mean(features, axis=0)  # [d]
        sigma = np.cov(features, rowvar=False)  # [d, d]
        
        # Check for valid computation
        if np.isnan(mu).any() or np.isnan(sigma).any():
            print("NaN values in statistics")
            return None, None
            
        return mu, sigma
    except Exception as e:
        print(f"Error in calculate_statistics: {e}")
        return None, None

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate FID between two distributions."""
    if mu1 is None or mu2 is None or sigma1 is None or sigma2 is None:
        return None
    
    diff = mu1 - mu2
    
    # Calculate sqrt(sigma1 * sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical stability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # FID formula
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return float(fid)

def compute_fid_efid(video_path, gt_path, model, transform, device):
    """Compute FID and E-FID for a pair of videos."""
    # 如果视频标记为unmatched，则跳过计算
    if unmatched_mapping.get(video_path, 0) == 1:
        print(f"Skipping unmatched video for FID/E-FID: {os.path.basename(video_path)}")
        return None, None
    
    # Extract frames
    gen_frames = get_video_frames(video_path)
    gt_frames = get_video_frames(gt_path)
    
    if not gen_frames or not gt_frames:
        print(f"Empty frames in {video_path} or {gt_path}")
        return None, None
    
    # Compute regular FID
    gt_features = get_inception_features(gt_frames, model, transform, device)
    gen_features = get_inception_features(gen_frames, model, transform, device)
    
    if gt_features.size == 0 or gen_features.size == 0:
        print(f"Feature extraction failed for {video_path}")
        return None, None
    
    mu_gt, sigma_gt = calculate_statistics(gt_features)
    mu_gen, sigma_gen = calculate_statistics(gen_features)
    fid_value = calculate_fid(mu_gt, sigma_gt, mu_gen, sigma_gen)
    
    # Compute E-FID (Edge-FID)
    gt_features_edge = get_edge_features(gt_frames, model, transform, device)
    gen_features_edge = get_edge_features(gen_frames, model, transform, device)
    
    if gt_features_edge.size == 0 or gen_features_edge.size == 0:
        print(f"Edge feature extraction failed for {video_path}")
        return fid_value, None
    
    mu_gt_edge, sigma_gt_edge = calculate_statistics(gt_features_edge)
    mu_gen_edge, sigma_gen_edge = calculate_statistics(gen_features_edge)
    efid_value = calculate_fid(mu_gt_edge, sigma_gt_edge, mu_gen_edge, sigma_gen_edge)
    
    return fid_value, efid_value

# FVD Calculation Functions
def compute_stats(feats):
    """Compute mean and covariance of features for FVD calculation."""
    if feats.size == 0:
        print("Empty features array")
        return None, None
    
    try:
        mu = np.mean(feats, axis=0)  # [d]
        
        # Special handling when we have only one sample
        if feats.shape[0] == 1:
            print("Only one sample in features, using identity matrix for covariance")
            # For a single sample, use a small identity matrix instead of computing covariance
            sigma = np.eye(feats.shape[1]) * 1e-6
        else:
            sigma = np.cov(feats, rowvar=False)  # [d, d]
        
        # Check for valid computation
        if np.isnan(mu).any() or np.isnan(sigma).any():
            print("NaN values in statistics")
            return None, None
            
        # Ensure sigma is a 2D array
        if np.ndim(sigma) == 0:
            print("Converting scalar covariance to 2D matrix")
            sigma = np.array([[sigma]])
            
        return mu, sigma
    except Exception as e:
        print(f"Error in compute_statistics: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compute_fvd(feats_fake, feats_real):
    """
    Calculate FVD between two sets of features.
    Exact implementation from /root/autodl-tmp/eval/fvd.py with additional handling for single samples.
    """
    if feats_fake is None or feats_real is None:
        return None
    
    try:
        # Debug information
        print(f"FVD feature shapes: fake={feats_fake.shape}, real={feats_real.shape}")
        
        # Handle samples with only one video
        if feats_fake.shape[0] == 1 or feats_real.shape[0] == 1:
            print("Warning: Computing FVD with single samples may be unreliable")
        
        mu_gen, sigma_gen = compute_stats(feats_fake)
        mu_real, sigma_real = compute_stats(feats_real)
        
        if mu_gen is None or mu_real is None or sigma_gen is None or sigma_real is None:
            print("Error: stats computation returned None")
            return None
        
        # Debug shapes
        print(f"Mean shapes: gen={mu_gen.shape}, real={mu_real.shape}")
        print(f"Sigma shapes: gen={sigma_gen.shape}, real={sigma_real.shape}")
        
        # Check for NaN or Inf values
        if not np.isfinite(sigma_gen).all() or not np.isfinite(sigma_real).all() or not np.isfinite(mu_gen).all() or not np.isfinite(mu_real).all():
            print("Error: non-finite values in statistics")
            return None
        
        # Ensure matrices are properly conditioned by adding a small epsilon
        eps = 1e-6
        sigma_gen_reg = sigma_gen + np.eye(sigma_gen.shape[0]) * eps
        sigma_real_reg = sigma_real + np.eye(sigma_real.shape[0]) * eps
        
        # Direct implementation from eval/fvd.py but with regularization
        m = np.square(mu_gen - mu_real).sum()
        print(f"Squared difference of means: {m}")
        
        # Use scipy.linalg.sqrtm with regularization
        sigma_product = np.dot(sigma_gen_reg, sigma_real_reg)
        
        # Ensure the input is valid for sqrtm
        if np.isnan(sigma_product).any() or np.isinf(sigma_product).any():
            print("Error: NaN or Inf values in sigma product")
            return None
            
        s, _ = scipy.linalg.sqrtm(sigma_product, disp=False)
        
        if np.iscomplexobj(s):
            s = s.real
            
        trace_term = np.trace(sigma_gen + sigma_real - 2*s)
        print(f"Trace term: {trace_term}")
        
        fid = float(m + trace_term)
        print(f"Final FVD score: {fid}")
        
        return fid
    except Exception as e:
        print(f"Error in FVD calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

@torch.no_grad()
def compute_our_fvd(videos_fake, videos_real, i3d_model, device="cuda"):
    """
    Compute FVD between two sets of videos.
    Direct implementation from /root/autodl-tmp/eval/fvd.py
    """
    if i3d_model is None:
        return None
    
    try:
        print("Starting compute_our_fvd")
        # Prepare arguments exactly as in eval/fvd.py
        i3d_kwargs = dict(
            rescale=False, resize=False, return_features=True
        )
        
        # Make sure videos are in the correct format: [B, F, H, W, C]
        if not isinstance(videos_fake, torch.Tensor) or not isinstance(videos_real, torch.Tensor):
            print("Error: Inputs must be torch tensors")
            return None
            
        print(f"Input tensor shapes: fake={videos_fake.shape}, real={videos_real.shape}")
        
        # Check for expected shape
        if len(videos_fake.shape) != 5 or len(videos_real.shape) != 5:
            print("Error: Input tensors must have 5 dimensions: [B, F, H, W, C]")
            return None
        
        # Permute to [B, C, F, H, W] as expected by I3D
        videos_fake = videos_fake.permute(0, 4, 1, 2, 3).to(device)
        videos_real = videos_real.permute(0, 4, 1, 2, 3).to(device)
        
        print(f"Permuted tensor shapes: fake={videos_fake.shape}, real={videos_real.shape}")
        
        # Extract features
        try:
            feats_fake = i3d_model(videos_fake, **i3d_kwargs).cpu().numpy()
            print(f"Generated features shape: {feats_fake.shape}")
            
            feats_real = i3d_model(videos_real, **i3d_kwargs).cpu().numpy()
            print(f"GT features shape: {feats_real.shape}")
            
            # Free memory
            del videos_fake
            del videos_real
            torch.cuda.empty_cache()
            
            return compute_fvd(feats_fake, feats_real)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("CUDA out of memory error. Try reducing batch size or video dimensions.")
                torch.cuda.empty_cache()
            raise
    except Exception as e:
        print(f"Error in compute_our_fvd: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_fvd_for_videos(video_path, gt_path, i3d_model, device):
    """
    Prepare videos and compute FVD using the approach in /root/autodl-tmp/eval/fvd.py
    """
    # 如果视频标记为unmatched，则跳过计算
    if unmatched_mapping.get(video_path, 0) == 1:
        print(f"Skipping unmatched video for FVD: {os.path.basename(video_path)}")
        return None
    
    if i3d_model is None:
        return None
    
    try:
        print(f"Preparing videos for FVD: {os.path.basename(video_path)} and {os.path.basename(gt_path)}")
        
        # Extract frames from both videos
        gen_frames = get_video_frames(video_path)
        gt_frames = get_video_frames(gt_path)
        
        if not gen_frames or not gt_frames:
            print("Empty frames in one or both videos")
            return None
        
        print(f"Frame counts: Generated={len(gen_frames)}, GT={len(gt_frames)}")
        
        # Ensure we have exactly 16 frames for each video
        if len(gen_frames) < 16:
            print(f"Padding generated video with {16-len(gen_frames)} black frames")
            black_frame = np.zeros_like(gen_frames[0])
            while len(gen_frames) < 16:
                gen_frames.append(black_frame.copy())
        
        if len(gt_frames) < 16:
            print(f"Padding GT video with {16-len(gt_frames)} black frames")
            black_frame = np.zeros_like(gt_frames[0])
            while len(gt_frames) < 16:
                gt_frames.append(black_frame.copy())
        
        # Take exactly 16 frames if more are available
        gen_frames = gen_frames[:16]
        gt_frames = gt_frames[:16]
        
        # Resize all frames to exactly 224x224 as expected by I3D
        gen_frames_resized = []
        gt_frames_resized = []
        
        for frame in gen_frames:
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            # Convert to RGB (I3D expects RGB format)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            gen_frames_resized.append(frame_rgb)
        
        for frame in gt_frames:
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            # Convert to RGB (I3D expects RGB format)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            gt_frames_resized.append(frame_rgb)
        
        print("Frames resized and converted to RGB format")
        
        # Convert to tensors exactly as in eval/fvd.py
        # input shape: (b, f, h, w, c) where b=1
        fake_np = np.stack(gen_frames_resized)
        real_np = np.stack(gt_frames_resized) 
        
        # Normalize to [0, 1]
        fake_tensor = torch.from_numpy(fake_np).float() / 255.0
        real_tensor = torch.from_numpy(real_np).float() / 255.0
        
        # Add batch dimension to get [1, 16, 224, 224, 3]
        fake_tensor = fake_tensor.unsqueeze(0)
        real_tensor = real_tensor.unsqueeze(0)
        
        print(f"Tensor shapes: fake={fake_tensor.shape}, real={real_tensor.shape}")
        
        # Compute FVD using our implementation of eval/fvd.py
        return compute_our_fvd(fake_tensor, real_tensor, i3d_model, device)
    except Exception as e:
        print(f"Error in compute_fvd_for_videos: {e}")
        import traceback
        traceback.print_exc()
        return None

# PSNR, SSIM, CSIM Functions
def cosine_similarity(img1, img2):
    """Calculate cosine similarity between two flattened images."""
    a = img1.flatten().astype(np.float64)
    b = img2.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def compute_psnr_ssim_csim(video_path, gt_path):
    """Compute PSNR, SSIM, and CSIM for a pair of videos."""
    # 如果视频标记为unmatched，则跳过计算
    if unmatched_mapping.get(video_path, 0) == 1:
        print(f"Skipping unmatched video for PSNR/SSIM/CSIM: {os.path.basename(video_path)}")
        return None, None, None
    
    gen_frames = get_video_frames(video_path)
    gt_frames = get_video_frames(gt_path)
    
    if not gen_frames or not gt_frames:
        print(f"Empty frames in {video_path} or {gt_path}")
        return None, None, None
    
    n_frames = min(len(gen_frames), len(gt_frames))
    psnr_list, ssim_list, csim_list = [], [], []
    
    for i in range(n_frames):
        gt_frame = gt_frames[i]
        gen_frame = gen_frames[i]
        
        # Resize if needed
        if gt_frame.shape != gen_frame.shape:
            gen_frame = cv2.resize(gen_frame, (gt_frame.shape[1], gt_frame.shape[0]))
        
        # Calculate PSNR
        try:
            psnr = peak_signal_noise_ratio(gt_frame, gen_frame, data_range=255)
            psnr_list.append(psnr)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
        
        # Calculate SSIM
        try:
            min_dim = min(gt_frame.shape[0], gt_frame.shape[1])
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            win_size = max(min(win_size, 7), 3)
            ssim = structural_similarity(gt_frame, gen_frame, channel_axis=-1, data_range=255, win_size=win_size)
            ssim_list.append(ssim)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
        
        # Calculate CSIM
        try:
            csim = cosine_similarity(gt_frame, gen_frame)
            csim_list.append(csim)
        except Exception as e:
            print(f"Error calculating CSIM: {e}")
    
    # Return average values
    avg_psnr = np.mean(psnr_list) if psnr_list else None
    avg_ssim = np.mean(ssim_list) if ssim_list else None
    avg_csim = np.mean(csim_list) if csim_list else None
    
    return avg_psnr, avg_ssim, avg_csim

# Main Video Processing Function
def process_video(video_path, gt_path, inception_model, i3d_model, transform, device):
    """Process a single video and extract all metrics."""
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    
    # Initialize metrics
    metrics = {
        'filename': os.path.basename(video_path),
        'FID': None,
        'FVD': None,
        'SSIM': None,
        'PSNR': None, 
        'E-FID': None,
        'CSIM': None
    }
    
    # Compute FID and E-FID
    try:
        print("Computing FID and E-FID...")
        fid, efid = compute_fid_efid(video_path, gt_path, inception_model, transform, device)
        metrics['FID'] = fid
        metrics['E-FID'] = efid
        print(f"FID: {fid}, E-FID: {efid}")
    except Exception as e:
        print(f"Error computing FID/E-FID for {video_path}: {e}")
    
    # Compute FVD if not skipped
    if not SKIP_FVD:
        try:
            print("Computing FVD...")
            fvd = compute_fvd_for_videos(video_path, gt_path, i3d_model, device)
            metrics['FVD'] = fvd
            print(f"FVD: {fvd}")
        except Exception as e:
            print(f"Error computing FVD for {video_path}: {e}")
    else:
        metrics['FVD'] = "SKIPPED"  # Indicate FVD was skipped
        print("FVD: SKIPPED")
    
    # Compute PSNR, SSIM, CSIM
    try:
        print("Computing PSNR, SSIM, CSIM...")
        psnr, ssim, csim = compute_psnr_ssim_csim(video_path, gt_path)
        metrics['PSNR'] = psnr
        metrics['SSIM'] = ssim
        metrics['CSIM'] = csim
        print(f"PSNR: {psnr}, SSIM: {ssim}, CSIM: {csim}")
    except Exception as e:
        print(f"Error computing PSNR/SSIM/CSIM for {video_path}: {e}")
    
    # Print the results
    print("Results summary:")
    for key, value in metrics.items():
        if key != 'filename':
            print(f"  {key}: {value}")
    
    return metrics

# Process Videos in Model Folders
def process_model_folder(model_folder, inception_model, i3d_model, transform, device):
    """Process all videos in a model folder."""
    results = []
    folder_path = os.path.join('/root/autodl-tmp', model_folder)
    
    if not os.path.exists(folder_path):
        print(f"Model folder not found: {folder_path}")
        return results
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    print(f"Processing {len(video_files)} videos in {model_folder}...")
    
    skipped_count = 0  # 计数被跳过的视频
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(folder_path, video_file)
        key = video_path
        
        # 检查是否应该跳过这个视频
        if unmatched_mapping.get(key, 0) == 1:
            skipped_count += 1
            continue
        
        # Find ground truth for this video
        gt_path = gt_mapping.get(key)
        if gt_path is None:
            print(f"No ground truth found for {video_path}")
            continue
        
        if not os.path.exists(gt_path):
            print(f"Ground truth not found: {gt_path}")
            continue
        
        # Process the video
        metrics = process_video(video_path, gt_path, inception_model, i3d_model, transform, device)
        results.append(metrics)
    
    print(f"Skipped {skipped_count} unmatched videos in {model_folder}")
    
    return results

# Process Face/Hand Videos
def process_video_group(video_dir, model_prefix, inception_model, i3d_model, transform, device):
    """Process videos for a specific model from a directory."""
    results = []
    
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return results
    
    # Filter videos by model prefix
    video_files = [f for f in os.listdir(video_dir) 
                  if f.startswith(f"{model_prefix}-") and f.endswith('.mp4')]
    
    print(f"Processing {len(video_files)} videos for model {model_prefix} in {os.path.basename(video_dir)}...")
    
    skipped_count = 0  # 计数被跳过的视频
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        key = video_path
        
        # 检查是否应该跳过这个视频
        if unmatched_mapping.get(key, 0) == 1:
            skipped_count += 1
            continue
        
        # Find ground truth for this video
        gt_path = gt_mapping.get(key)
        if gt_path is None:
            # For face/hand videos where we might not have GT reference, use the same video
            gt_path = video_path
        
        if not os.path.exists(gt_path):
            print(f"Ground truth not found: {gt_path}")
            continue
        
        # Process the video
        metrics = process_video(video_path, gt_path, inception_model, i3d_model, transform, device)
        results.append(metrics)
    
    print(f"Skipped {skipped_count} unmatched videos for model {model_prefix} in {os.path.basename(video_dir)}")
    
    return results

# Save Results to CSV
def save_to_csv(results, filename):
    """Save results to a CSV file."""
    if not results:
        print(f"No results to save for {filename}")
        return
    
    filepath = os.path.join('FIDres', filename)
    
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=METRICS)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")

# Main Function
def main():
    # Create progress tracking file
    progress_log = os.path.join('FIDres', 'processing_log.txt')
    with open(progress_log, 'w') as f:
        f.write("Video Processing Started\n")
        f.write(f"Using groundtruth mapping from groundtruth_results.csv\n")
        if SKIP_FVD:
            f.write("NOTE: FVD computation is SKIPPED to avoid errors. Set SKIP_FVD=False to enable.\n")
        else:
            f.write("NOTE: FVD computation is ENABLED. Set SKIP_FVD=True to disable if errors occur.\n")
    
    # Print FVD status
    if SKIP_FVD:
        print("NOTE: FVD computation is SKIPPED to avoid errors. Set SKIP_FVD=False to enable.")
    else:
        print("NOTE: FVD computation is ENABLED. Set SKIP_FVD=True to disable if errors occur.")
    
    # Track failures
    failures = []
    
    # 1. First, test with just one video pair to check all metrics
    print("\n=== TESTING METRICS WITH ONE VIDEO PAIR ===")
    try:
        # Try to get a single video from model folder 1
        model_folder = '1'
        folder_path = os.path.join('/root/autodl-tmp', model_folder)
        if os.path.exists(folder_path):
            video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            if len(video_files) >= 2:  # Need at least 2 for better FVD
                # Using two videos improves FVD calculation
                print("Using two videos for more reliable FVD calculation")
                video_path1 = os.path.join(folder_path, video_files[0])
                video_path2 = os.path.join(folder_path, video_files[1])
                
                # Find ground truth for these videos
                gt_path1 = gt_mapping.get(video_path1)
                gt_path2 = gt_mapping.get(video_path2)
                
                if gt_path1 and gt_path2 and os.path.exists(gt_path1) and os.path.exists(gt_path2):
                    print("Testing FVD with batch approach (preferred)...")
                    # First test regular metrics with one video
                    print(f"Testing metrics on video: {os.path.basename(video_path1)}")
                    metrics = process_video(video_path1, gt_path1, inception_model, i3d_model, transform_pipeline, device)
                    
                    # Then try FVD with batch approach
                    if metrics['FVD'] is None:
                        print("\nAttempting batch-based FVD calculation with multiple videos...")
                        try:
                            # Extract frames from both videos
                            gen_frames1 = get_video_frames(video_path1)
                            gen_frames2 = get_video_frames(video_path2)
                            gt_frames1 = get_video_frames(gt_path1)
                            gt_frames2 = get_video_frames(gt_path2)
                            
                            # Ensure we have 16 frames for each
                            for frames_list in [gen_frames1, gen_frames2, gt_frames1, gt_frames2]:
                                if len(frames_list) < 16:
                                    black_frame = np.zeros_like(frames_list[0])
                                    while len(frames_list) < 16:
                                        frames_list.append(black_frame.copy())
                                # Take only 16 frames
                                frames_list = frames_list[:16]
                            
                            # Resize and convert to RGB
                            def prepare_frames(frames_list):
                                resized = []
                                for frame in frames_list:
                                    frame_resized = cv2.resize(frame, (224, 224))
                                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                                    resized.append(frame_rgb)
                                return resized
                            
                            gen_resized1 = prepare_frames(gen_frames1)
                            gen_resized2 = prepare_frames(gen_frames2)
                            gt_resized1 = prepare_frames(gt_frames1)
                            gt_resized2 = prepare_frames(gt_frames2)
                            
                            # Stack and create tensors
                            gen_np = np.stack([np.stack(gen_resized1), np.stack(gen_resized2)])
                            gt_np = np.stack([np.stack(gt_resized1), np.stack(gt_resized2)])
                            
                            gen_tensor = torch.from_numpy(gen_np).float() / 255.0
                            gt_tensor = torch.from_numpy(gt_np).float() / 255.0
                            
                            print(f"Batch tensor shapes: gen={gen_tensor.shape}, gt={gt_tensor.shape}")
                            
                            # Compute FVD directly
                            fvd_result = compute_our_fvd(gen_tensor, gt_tensor, i3d_model, device)
                            print(f"Batch FVD result: {fvd_result}")
                            
                            # Update metrics with the batch FVD result
                            metrics['FVD'] = fvd_result
                        except Exception as e:
                            print(f"Error in batch FVD calculation: {e}")
                    
                    # Save test results
                    test_results = [metrics]
                    save_to_csv(test_results, "test_results.csv")
                    print("Test results saved to FIDres/test_results.csv")
                else:
                    print(f"No valid ground truth found for test videos")
            elif video_files:
                # Fall back to single video test
                video_file = video_files[0]
                video_path = os.path.join(folder_path, video_file)
                key = video_path
                
                # Find ground truth for this video
                gt_path = gt_mapping.get(key)
                if gt_path and os.path.exists(gt_path):
                    print(f"Testing with single video: {video_path}")
                    print(f"Ground truth: {gt_path}")
                    
                    metrics = process_video(video_path, gt_path, inception_model, i3d_model, transform_pipeline, device)
                    print("\nTest results:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                    
                    # Save test results to a CSV
                    test_results = [metrics]
                    save_to_csv(test_results, "test_results.csv")
                    print("Test results saved to FIDres/test_results.csv")
                else:
                    print(f"No valid ground truth found for test video")
            else:
                print(f"No video files found in {folder_path}")
        else:
            print(f"Model folder {folder_path} not found")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== CONTINUE WITH FULL PROCESSING? (y/n) ===")
    proceed = input().strip().lower()
    if proceed != 'y':
        print("Exiting after test.")
        return
    
    # 2. Process videos in model folders (whole body)
    for model_folder in MODEL_FOLDERS:
        try:
            with open(progress_log, 'a') as f:
                f.write(f"Starting model folder: {model_folder}\n")
            
            results = process_model_folder(
                model_folder, inception_model, i3d_model, transform_pipeline, device
            )
            save_to_csv(results, f"{model_folder}-wholebody-score.csv")
            
            with open(progress_log, 'a') as f:
                f.write(f"Completed model folder: {model_folder} with {len(results)} videos\n")
        except Exception as e:
            msg = f"Error processing model folder {model_folder}: {e}"
            print(msg)
            failures.append(msg)
            with open(progress_log, 'a') as f:
                f.write(f"ERROR: {msg}\n")
    
    # 3. Process face videos
    face_dir = '/root/autodl-tmp/face_videos_v5'
    for model_prefix in MODEL_FOLDERS:
        try:
            with open(progress_log, 'a') as f:
                f.write(f"Starting face videos for model: {model_prefix}\n")
            
            results = process_video_group(
                face_dir, model_prefix, inception_model, i3d_model, transform_pipeline, device
            )
            save_to_csv(results, f"{model_prefix}-face-score.csv")
            
            with open(progress_log, 'a') as f:
                f.write(f"Completed face videos for model: {model_prefix} with {len(results)} videos\n")
        except Exception as e:
            msg = f"Error processing face videos for model {model_prefix}: {e}"
            print(msg)
            failures.append(msg)
            with open(progress_log, 'a') as f:
                f.write(f"ERROR: {msg}\n")
    
    # 4. Process hand videos
    hand_dir = '/root/autodl-tmp/hand_videos_v5'
    for model_prefix in MODEL_FOLDERS:
        try:
            with open(progress_log, 'a') as f:
                f.write(f"Starting hand videos for model: {model_prefix}\n")
            
            results = process_video_group(
                hand_dir, model_prefix, inception_model, i3d_model, transform_pipeline, device
            )
            save_to_csv(results, f"{model_prefix}-hand-score.csv")
            
            with open(progress_log, 'a') as f:
                f.write(f"Completed hand videos for model: {model_prefix} with {len(results)} videos\n")
        except Exception as e:
            msg = f"Error processing hand videos for model {model_prefix}: {e}"
            print(msg)
            failures.append(msg)
            with open(progress_log, 'a') as f:
                f.write(f"ERROR: {msg}\n")
    
    # Print final unmatched statistics
    total_unmatched = sum(1 for val in unmatched_mapping.values() if val == 1)
    print(f"Total unmatched videos excluded from processing: {total_unmatched}")
    with open(progress_log, 'a') as f:
        f.write(f"\nTotal unmatched videos excluded from processing: {total_unmatched}\n")
    
    # Report completion
    with open(progress_log, 'a') as f:
        f.write("\nProcessing completed!\n")
        if failures:
            f.write("\nErrors encountered:\n")
            for failure in failures:
                f.write(f"- {failure}\n")
        else:
            f.write("No errors encountered.\n")
    
    print("Video processing completed! Check FIDres directory for results.")
    if failures:
        print(f"Warning: {len(failures)} errors were encountered. Check {progress_log} for details.")

if __name__ == "__main__":
    main()
