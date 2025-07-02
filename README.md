# Whole-Body Benchmark Dataset
A Comprehensive Multi-Modal Benchmark for Whole-Body Animatable Avatar Generation
<p align="center">
  <img src="./sampleImg/logo.jpg" alt="Whole-Body Benchmark Logo" width="300px">
  <br>
  <em>A Comprehensive Multi-Modal Benchmark for Evaluating Whole-Body Animatable Avatar Generation</em>
  <br>
  <a href="#" target="_blank">GitHub</a> | 
  <a href="#" target="_blank">Paper</a> |
  <a href="#" target="_blank">Dataset</a>
</p>

## 📖 Introduction

The Whole-Body Benchmark Dataset addresses a critical gap in evaluating whole-body animatable avatar generation systems. Creating realistic, fully animatable whole-body avatars from a single static portrait remains challenging, as existing methods struggle to accurately capture:

- Subtle facial expressions
- Corresponding full-body movements
- Dynamic background changes
- Consistent identity preservation

Our benchmark is motivated by two primary needs:
1. Current metrics fail to adequately capture the full complexity involved in generating whole-body animatable avatars from a single image
2. A dedicated benchmark dataset can offer valuable insights to drive progress in high-quality animatable avatar generation

The Whole-Body Benchmark Dataset is a fully open-source, multi-modal benchmark specifically designed for whole-body animatable avatar generation. It provides comprehensive labels and a versatile evaluation framework to facilitate rigorous assessment of high-quality whole-body animatable avatar generation.

## 🎬 Key Features

Our benchmark provides several compelling features:

- **Multi-region Evaluation**: Specialized metrics for whole-body, face, and hand regions to provide comprehensive assessment
- **Detailed Multi-Modal Annotations**: Comprehensive labels that facilitate high-quality whole-body animatable avatar generation by providing fine-grained guidance
- **Versatile Evaluation Framework**: Enables rigorous assessment of whole-body animatable avatar quality across multiple dimensions
- **Specialized Metrics**: Both objective metrics (FID, E-FID, FVD, PSNR, SSIM, CSIM) and subjective metrics for holistic evaluation
- **Standardized Methodology**: Consistent testing protocol for fair comparison between different methods

## 📊 Evaluation Framework

Our benchmark provides multiple specialized evaluation metrics:

### Objective Metrics
- **FID (Fréchet Inception Distance)**: Measures visual quality similarity
- **E-FID (Edge-FID)**: Evaluates structural consistency using edge maps
- **FVD (Fréchet Video Distance)**: Assesses temporal coherence in generated videos
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantifies reconstruction accuracy
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity
- **CSIM (Cosine Similarity)**: Evaluates feature-level similarity

### Subjective Consistency Metrics
Six dimensions of subjective evaluation:
- Subject Consistency
- Background Consistency
- Motion Smoothness
- Dynamic Degree
- Aesthetic Quality
- Imaging Quality

## 🚀 Getting Started

### Installation

This benchmark uses MMPose for keypoint detection. For detailed installation instructions, please refer to the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).

We recommend following these steps to set up your environment:

#### Prerequisites

Our benchmark requires Python 3.7+, CUDA 9.2+ and PyTorch 1.8+.

**Step 0.** Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) from the official website.

**Step 1.** Create a conda environment and activate it:
```bash
conda create --name wholebody python=3.8 -y
conda activate wholebody
```

**Step 2.** Install PyTorch following official instructions:

For GPU platforms:
```bash
conda install pytorch torchvision -c pytorch
```

For CPU-only platforms:
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 3.** Install MMEngine and MMCV using MIM:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

**Step 4.** Install MMPose:
```bash
mim install "mmpose>=1.0.0"
```

**Step 5.** Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/yourusername/whole-body-benchmark.git
cd whole-body-benchmark

```

#### Alternative: Manual Installation from Source

If you prefer to install MMPose from source (recommended for development):

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -v -e .
# "-v" means verbose, and "-e" means installing in development mode
cd ..
```

#### Verify the Installation

To verify that MMPose and other dependencies are correctly installed:

```bash
# Enter Python interpreter
python

# Try importing key packages
>>> import torch
>>> import mmpose
>>> import mmcv
>>> import cv2
>>> print(torch.__version__)
>>> print(mmpose.__version__)
```

For troubleshooting, please refer to the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).

### Dataset Structure

```
whole-body-benchmark/
├── gt_test/                  # Ground truth test videos
├── 3-img/                    # Extracted image frames
├── face_videos_v5/           # Face region videos
├── hand_videos_v5/           # Hand region videos
├── FIDres/                   # FID evaluation results
├── SCres/                    # Subjective Consistency results
└── evaluation/               # Evaluation scripts
```

### Running Evaluations

```bash
# Process video regions extraction
python process_video_hands_videos.py

# Calculate FID, E-FID, FVD, PSNR, SSIM metrics
python FID_calculate.py

# Calculate Subjective Consistency scores
python SC_calculate.py
```

## 📊 Benchmark Results

Performance comparison of various methods on our benchmark:

### Whole Body Region
| Method | SC | BC | MS | DD | AQ | IQ | FID | FVD | SSIM | PSNR | E-FID | CSIM |
|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Step | 93.45% | 94.50% | 98.71% | 67.39% | 48.32% | 61.99% | 172.47 | 1356.49 | 0.573 | 17.74 | 187.58 | 0.760 |
| Hun | 92.55% | 94.30% | 98.50% | 99.34% | 43.17% | 63.54% | 208.75 | 1567.94 | 0.543 | 16.06 | 216.91 | 0.811 |
| Wan | 96.83% | 96.31% | 99.69% | 19.90% | 46.72% | 54.67% | 92.76 | 750.51 | 0.660 | 20.16 | 196.39 | 0.896 |
| OpenS | 69.18% | 70.00% | 72.86% | 10.53% | 33.88% | 43.88% | 205.82 | 1032.52 | 0.481 | 14.28 | 269.33 | 0.678 |
| Ha3/w | 20.43% | 20.15% | 20.84% | 0.00% | 10.13% | 10.00% | 491.43 | 2470.14 | 0.209 | 6.24 | 484.08 | 0.303 |
| Ha3/wo | 10.90% | 11.09% | 11.54% | 0.00% | 3.90% | 5.16% | 568.68 | 4366.39 | 0.016 | 0.83 | 634.60 | 0.053 |
| ecv2/w | 29.16% | 31.25% | 29.75% | 9.81% | 14.27% | 16.53% | 508.19 | 2519.77 | 0.216 | 6.17 | 492.17 | 0.299 |
| ecv2/wo | 7.08% | 6.49% | 7.14% | 0.09% | 2.52% | 3.29% | 572.14 | 3791.25 | 0.028 | 1.29 | 712.91 | 0.092 |

### Face Region
| Method | SC | BC | MS | DD | AQ | IQ | FID | FVD | SSIM | PSNR | E-FID | CSIM |
|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Step | 65.06% | 74.04% | 63.12% | 14.91% | 28.18% | 39.40% | 300.98 | 1811.92 | 0.372 | 13.54 | 339.18 | 0.736 |
| Hun | 55.07% | 68.05% | 56.78% | 4.15% | 21.26% | 34.19% | 291.21 | 1351.71 | 0.322 | 14.47 | 347.24 | 0.787 |
| Wan | 66.51% | 72.26% | 63.40% | 25.12% | 25.09% | 33.68% | 254.03 | 1591.22 | 0.417 | 16.99 | 273.97 | 0.866 |
| OpenS | 52.34% | 63.55% | 49.74% | 7.69% | 19.76% | 30.90% | 275.50 | 1221.85 | 0.388 | 16.29 | 359.85 | 0.863 |
| Ha3/w | 14.92% | 15.73% | 14.66% | 7.21% | 5.79% | 6.98% | 425.82 | 2794.85 | 0.127 | 6.75 | 451.76 | 0.385 |
| Ha3/wo | 12.23% | 15.15% | 12.15% | 0.00% | 2.99% | 4.60% | 720.22 | 3255.19 | 0.010 | 0.89 | 492.67 | 0.065 |
| ecv2/w | 14.79% | 18.16% | 12.73% | 2.42% | 5.71% | 8.49% | 437.27 | 2651.92 | 0.139 | 6.88 | 439.92 | 0.401 |
| ecv2/wo | 8.29% | 9.27% | 8.16% | 0.21% | 5.69% | 6.29% | 801.27 | 3047.27 | 0.027 | 1.249 | 513.58 | 0.062 |

### Hand Region
| Method | SC | BC | MS | DD | AQ | IQ | FID | FVD | SSIM | PSNR | E-FID | CSIM |
|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Step | 66.83% | 77.98% | 46.36% | 48.14% | 24.22% | 38.40% | 327.10 | 1597.65 | 0.316 | 12.98 | 277.47 | 0.681 |
| Hun | 48.15% | 64.98% | 49.65% | 52.85% | 19.02% | 27.69% | 301.94 | 1734.98 | 0.262 | 11.45 | 255.53 | 0.737 |
| Wan | 51.27% | 57.45% | 42.49% | 41.71% | 18.38% | 23.84% | 331.51 | 1561.58 | 0.367 | 13.09 | 240.48 | 0.796 |
| OpenS | 19.16% | 22.45% | 13.30% | 13.89% | 6.40% | 11.50% | 278.95 | 1534.50 | 0.342 | 15.50 | 264.98 | 0.794 |
| Ha3/w | 13.65% | 14.76% | 13.25% | 13.05% | 5.08% | 6.51% | 574.40 | 7750.48 | 0.098 | 6.13 | 445.59 | 0.366 |
| Ha3/wo | 2.13% | 2.56% | 0.49% | 0.91% | 0.39% | 0.89% | 921.16 | 10671.79 | 0.008 | 0.55 | 713.46 | 0.041 |
| ecv2/w | 14.19% | 18.21% | 9.27% | 9.52% | 5.08% | 10.24% | 588.72 | 8958.27 | 0.106 | 7.272 | 525.18 | 0.427 |
| ecv2/wo | 3.58% | 4.27% | 2.79% | 1.84% | 2.74% | 1.94% | 975.92 | 9472.28 | 0.019 | 0.492 | 741.03 | 0.152 |

*Legend: SC: Subject Consistency, BC: Background Consistency, MS: Motion Smoothness, DD: Dynamic Degree, AQ: Aesthetic Quality, IQ: Imaging Quality, FID: Fréchet Inception Distance, FVD: Fréchet Video Distance, SSIM: Structural Similarity Index Measure, PSNR: Peak Signal-to-Noise Ratio, E-FID: Enhanced Fréchet Inception Distance, CSIM: Cosine Similarity. ↑ indicates higher is better, ↓ indicates lower is better.*

## 🔧 Technical Details

### FID_calculate.py

This script calculates objective metrics:
- Extracts frames from generated and ground truth videos
- Computes FID using InceptionV3 features
- Calculates E-FID using edge maps
- Measures FVD using I3D network
- Calculates PSNR, SSIM, and CSIM

### SC_calculate.py

This script processes subjective consistency scores:
- Filters out unmatched videos
- Calculates average scores across dimensions
- Generates summary statistics

### process_video_hands_videos.py

This script handles video region extraction:
- Uses DWPose-onnx for face and hand region detection
- Processes video frames and extracts keypoints
- Creates cropped face and hand videos for specialized evaluation

## 📝 Citation

If you find our work useful in your research, please consider citing:

```
```

## 📬 Contact

For questions, issues, or collaboration opportunities:


## 🙏 Acknowledgements

We extend our gratitude to:
- [VBench](https://github.com/Vchitect/VBench) for their Comprehensive Benchmark Suite for Video Generative Models
- [DWPose-onnx](https://github.com/IDEA-Research/DWPose) for their keypoint detection framework
- Our research institution for providing computational resources
- Other contributors and funding sources
- The following researchers:
```
@article{huang2025step,
  title={Step-Video-TI2V Technical Report: A State-of-the-Art Text-Driven Image-to-Video Generation Model},
  author={Huang, Haoyang and Ma, Guoqing and Duan, Nan and Chen, Xing and Wan, Changyi and Ming, Ranchen and Wang, Tianyu and Wang, Bo and Lu, Zhiying and Li, Aojie and others},
  journal={arXiv preprint arXiv:2503.11251},
  year={2025}
}

@article{kong2024hunyuanvideo,
  title={Hunyuanvideo: A systematic framework for large video generative models},
  author={Kong, Weijie and Tian, Qi and Zhang, Zijian and Min, Rox and Dai, Zuozhuo and Zhou, Jin and Xiong, Jiangfeng and Li, Xin and Wu, Bo and Zhang, Jianwei and others},
  journal={arXiv preprint arXiv:2412.03603},
  year={2024}
}

@article{opensora2,
    title={Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k}, 
    author={Xiangyu Peng and Zangwei Zheng and Chenhui Shen and Tom Young and Xinying Guo and Binluo Wang and Hang Xu and Hongxin Liu and Mingyan Jiang and Wenjun Li and Yuhui Wang and Anbang Ye and Gang Ren and Qianran Ma and Wanying Liang and Xiang Lian and Xiwen Wu and Yuting Zhong and Zhuangyan Li and Chaoyu Gong and Guojun Lei and Leijun Cheng and Limin Zhang and Minghao Li and Ruijie Zhang and Silan Hu and Shijie Huang and Xiaokang Wang and Yuanheng Zhao and Yuqi Wang and Ziang Wei and Yang You},
    year={2025},
    journal={arXiv preprint arXiv:2503.09642},
}

@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}

@article{cui2024hallo3,
  title={Hallo3: Highly dynamic and realistic portrait image animation with diffusion transformer networks},
  author={Cui, Jiahao and Li, Hui and Zhan, Yun and Shang, Hanlin and Cheng, Kaihui and Ma, Yuqi and Mu, Shan and Zhou, Hang and Wang, Jingdong and Zhu, Siyu},
  journal={arXiv preprint arXiv:2412.00733},
  year={2024}
}

@article{meng2024echomimicv2,
  title={EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation},
  author={Meng, Rang and Zhang, Xingyu and Li, Yuming and Ma, Chenguang},
  journal={arXiv preprint arXiv:2411.10061},
  year={2024}
}
```
## 📄 License

This project is licensed under the [MIT License](LICENSE).
