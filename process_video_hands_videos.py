#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DWPose视频处理启动脚本 (第四版)
功能：
1. 处理gt_test文件夹中的视频
2. 直接使用DWPose-onnx/ControlNet模型检测脸部和手部区域
3. 将视频转换为图片序列并存储在3-img目录中
4. 对每个图片进行独立的脸部和手部裁剪
5. 将裁剪的图片保存为单帧图片和视频
"""

import os
import sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import csv

# 检查是否存在DWPose-onnx目录
if not os.path.exists('DWPose-onnx'):
    print("错误: 未找到DWPose-onnx目录!")
    print("请确保在运行此脚本前已下载DWPose-onnx项目")
    sys.exit(1)

# 定义检测失败记录CSV文件
DETECTION_FAIL_CSV = 'detection_fail_v5.csv'

class VideoProcessor:

    def __init__(self):
        """初始化视频处理器"""
        # 创建输出目录
        os.makedirs('3-img', exist_ok=True)
        os.makedirs('face_videos_v5', exist_ok=True)
        os.makedirs('hand_videos_v5', exist_ok=True)
        
        # 添加DWPose路径
        DWPOSE_PATH = os.path.join(os.getcwd(), 'DWPose-onnx')
        sys.path.append(DWPOSE_PATH)
        
        # 添加ControlNet路径
        controlnet_path = os.path.join(DWPOSE_PATH, 'ControlNet-v1-1-nightly')
        sys.path.append(controlnet_path)
        
        # 设置模型路径
        YOLOX_ONNX_PATH = 'DWPose-onnx/ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx'
        POSE_ONNX_PATH = 'DWPose-onnx/ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx'
        
        # 检查是否存在模型文件
        if not os.path.exists(YOLOX_ONNX_PATH):
            print(f"错误: 未找到YOLOX模型文件 {YOLOX_ONNX_PATH}!")
            sys.exit(1)
        
        if not os.path.exists(POSE_ONNX_PATH):
            print(f"错误: 未找到姿势估计模型文件 {POSE_ONNX_PATH}!")
            sys.exit(1)
        
        try:
            # 加载ONNX模型
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider']
            self.session_det = ort.InferenceSession(path_or_bytes=YOLOX_ONNX_PATH, providers=providers)
            self.session_pose = ort.InferenceSession(path_or_bytes=POSE_ONNX_PATH, providers=providers)
            
            # 导入DWPose相关模块
            from annotator.dwpose.onnxdet import inference_detector
            from annotator.dwpose.onnxpose import inference_pose
            self.inference_detector = inference_detector
            self.inference_pose = inference_pose
            print("成功加载DWPose模型")
                
        except Exception as e:
            print(f"错误: 加载模型失败 - {e}")
            print("请先安装必要的依赖项:")
            print("1. 对于DWPose: 确保DWPose-onnx目录存在且包含必要模型")
            print("2. 一般依赖: pip install opencv-python numpy tqdm onnxruntime-gpu")
            sys.exit(1)
            
        # 全局参数
        self.detection_fail_list = []
        
    def convert_video_to_images(self, video_path, output_folder):
        """将视频转换为图片序列"""
        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return None, 0
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {video_path}")
        print(f"  - 大小: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - 总帧数: {frame_count}")
        
        # 保存视频元数据
        metadata_path = os.path.join(output_folder, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"source: {video_path}\n")
            f.write(f"fps: {fps}\n")
            f.write(f"width: {width}\n")
            f.write(f"height: {height}\n")
            f.write(f"frame_count: {frame_count}\n")
        
        # 读取并保存每一帧
        frame_idx = 0
        saved_frames = []
        
        with tqdm(total=frame_count, desc=f"转换视频 {os.path.basename(video_path)}", unit="帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存帧
                frame_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
                
                frame_idx += 1
                pbar.update(1)
        
        # 释放视频
        cap.release()
        
        print(f"成功将视频转换为 {len(saved_frames)} 帧图片，保存在 {output_folder}")
        return saved_frames, fps
    
    def process_videos(self):
        """处理 gt_test 文件夹中的所有视频"""
        video_folders = sorted(glob('gt_test/*'))
        
        for folder in tqdm(video_folders, desc="处理视频文件夹"):
            # 检查是否是目录
            if not os.path.isdir(folder):
                continue
                
            folder_basename = os.path.basename(folder)
            video_paths = glob(f"{folder}/*.mp4")
            
            if not video_paths:
                print(f"警告: 在文件夹 {folder} 中未找到MP4文件")
                continue
            
            for video_path in video_paths:
                video_basename = os.path.basename(video_path)
                video_id = os.path.splitext(video_basename)[0]
                
                # 创建输出目录（使用3-img的格式）
                output_folder = f"3-img/{folder_basename}/{video_id}"
                os.makedirs(output_folder, exist_ok=True)
                
                # 转换视频为图片序列
                images, fps = self.convert_video_to_images(video_path, output_folder)
                
                if not images:
                    print(f"警告: 视频 {video_path} 转换失败，跳过处理")
                    continue
                
                # 确保输出目录存在
                os.makedirs(f"{output_folder}/faces_v5", exist_ok=True)
                os.makedirs(f"{output_folder}/hands_v5", exist_ok=True)
                
                # 处理转换好的图片
                self.process_images(output_folder, images, fps, video_path, folder_basename, video_id)
    
    def process_images(self, video_folder, images, fps, video_source, folder_name, video_id):
        """处理图片序列提取脸部和手部区域"""
        if not images:
            print(f"警告: 在 {video_folder} 中未找到图片，跳过处理")
            return
        
        # 检测统计
        face_detection_count = 0
        hand_detection_count = 0
        total_frames = len(images)
        
        # 保存面部和手部裁剪图像的尺寸
        face_sizes = []
        hand_sizes = []
        
        # 创建面部和手部帧索引到区域的映射
        face_frames = {}
        hand_frames = {}
        
        # 处理每张图片，记录面部和手部区域
        for img_path in tqdm(images, desc=f"处理 {video_folder} 中的图片", leave=False):
            frame_idx = int(os.path.basename(img_path).split('_')[1].split('.')[0])
            
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}，跳过")
                continue
            
            # 检测人体
            det_result = self.inference_detector(self.session_det, img)
            
            # 检测姿势
            keypoints, scores = self.inference_pose(self.session_pose, det_result, img)
            
            if len(keypoints) > 0:
                # 提取面部关键点 (24-92)
                face_keypoints = keypoints[0][24:92]
                valid_face_pts = face_keypoints[np.where(scores[0][24:92] > 0.3)]
                
                # 提取手部关键点 (92-113, 113-134)
                hand_keypoints_left = keypoints[0][92:113]
                hand_keypoints_right = keypoints[0][113:134]
                valid_hand_pts_left = hand_keypoints_left[np.where(scores[0][92:113] > 0.3)]
                valid_hand_pts_right = hand_keypoints_right[np.where(scores[0][113:134] > 0.3)]
                
                # 合并左右手关键点
                if len(valid_hand_pts_left) > 0 and len(valid_hand_pts_right) > 0:
                    valid_hand_pts = np.vstack([valid_hand_pts_left, valid_hand_pts_right])
                elif len(valid_hand_pts_left) > 0:
                    valid_hand_pts = valid_hand_pts_left
                elif len(valid_hand_pts_right) > 0:
                    valid_hand_pts = valid_hand_pts_right
                else:
                    valid_hand_pts = np.array([])
            else:
                valid_face_pts = np.array([])
                valid_hand_pts = np.array([])
            
            # 处理面部区域
            if len(valid_face_pts) > 0:
                min_x = max(0, int(np.min(valid_face_pts[:, 0])))
                min_y = max(0, int(np.min(valid_face_pts[:, 1])))
                max_x = min(img.shape[1], int(np.max(valid_face_pts[:, 0])))
                max_y = min(img.shape[0], int(np.max(valid_face_pts[:, 1])))
                
                # 扩大边界框以获得完整脸部
                width = max_x - min_x
                height = max_y - min_y
                min_x = max(0, min_x - int(width * 0.5))
                min_y = max(0, min_y - int(height * 0.5))
                max_x = min(img.shape[1], max_x + int(width * 0.5))
                max_y = min(img.shape[0], max_y + int(height * 0.5))
                
                # 确保最小尺寸
                if (max_x - min_x) < 50 or (max_y - min_y) < 50:
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    min_x = max(0, center_x - 25)
                    min_y = max(0, center_y - 25)
                    max_x = min(img.shape[1], center_x + 25)
                    max_y = min(img.shape[0], center_y + 25)
                
                # 记录该帧索引对应的面部区域
                face_frames[frame_idx] = (min_x, min_y, max_x, max_y)
                
                # 裁剪并保存面部图像
                face_crop = img[min_y:max_y, min_x:max_x]

                # 检查裁剪是否有效
                if face_crop.size > 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    face_crop_path = f"{video_folder}/faces_v5/face_{frame_idx:06d}.jpg"
                    try:
                        cv2.imwrite(face_crop_path, face_crop)
                        face_sizes.append((max_x - min_x, max_y - min_y))
                        face_detection_count += 1
                    except Exception as e:
                        print(f"警告: 保存面部图像 {face_crop_path} 失败 - {e}")
                else:
                    print(f"警告: 帧 {frame_idx} 的面部裁剪区域无效: {min_x},{min_y},{max_x},{max_y}")
            
            # 处理手部区域
            if len(valid_hand_pts) > 0:
                min_x = max(0, int(np.min(valid_hand_pts[:, 0])))
                min_y = max(0, int(np.min(valid_hand_pts[:, 1])))
                max_x = min(img.shape[1], int(np.max(valid_hand_pts[:, 0])))
                max_y = min(img.shape[0], int(np.max(valid_hand_pts[:, 1])))
                
                # 扩大边界框以获得手腕周围区域
                width = max_x - min_x
                height = max_y - min_y
                expansion = max(width, height) * 1.0  # 扩大区域以包含整个手
                
                # 根据手腕位置进行适当扩展
                if width < 20 or height < 20:  # 如果检测到的区域太小，扩大更多
                    min_x = max(0, min_x - int(expansion))
                    min_y = max(0, min_y - int(expansion))
                    max_x = min(img.shape[1], max_x + int(expansion))
                    max_y = min(img.shape[0], max_y + int(expansion))
                else:
                    min_x = max(0, min_x - int(width * 0.5))
                    min_y = max(0, min_y - int(height * 0.5))
                    max_x = min(img.shape[1], max_x + int(width * 0.5))
                    max_y = min(img.shape[0], max_y + int(height * 0.5))
                
                # 确保坐标有效
                min_x, min_y = max(0, min_x), max(0, min_y)
                max_x = min(img.shape[1], max_x)
                max_y = min(img.shape[0], max_y)
                
                # 检查坐标是否构成有效区域
                if min_x >= max_x or min_y >= max_y:
                    print(f"警告: 帧 {frame_idx} 的手部坐标无效: {min_x},{min_y},{max_x},{max_y}")
                    continue
                
                # 记录该帧索引对应的手部区域
                hand_frames[frame_idx] = (min_x, min_y, max_x, max_y)
                
                # 裁剪并保存手部图像
                hand_crop = img[min_y:max_y, min_x:max_x]

                # 检查裁剪是否有效
                if hand_crop.size > 0 and hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                    hand_crop_path = f"{video_folder}/hands_v5/hand_{frame_idx:06d}.jpg"
                    try:
                        cv2.imwrite(hand_crop_path, hand_crop)
                        hand_sizes.append((max_x - min_x, max_y - min_y))
                        hand_detection_count += 1
                    except Exception as e:
                        print(f"警告: 保存手部图像 {hand_crop_path} 失败 - {e}")
                else:
                    print(f"警告: 帧 {frame_idx} 的手部裁剪区域无效: {min_x},{min_y},{max_x},{max_y}")

        # 计算面部和手部区域的面积
        face_areas = [(max_x-min_x)*(max_y-min_y) for min_x,min_y,max_x,max_y in face_frames.values()]
        hand_areas = [(max_x-min_x)*(max_y-min_y) for min_x,min_y,max_x,max_y in hand_frames.values()]
        
        if face_areas:
            # 计算面部面积统计
            avg_face_area = np.mean(face_areas)
            min_face_area = np.min(face_areas)
            max_face_area = np.max(face_areas)
            print(f"面部区域面积统计: 平均={avg_face_area:.1f}, 最小={min_face_area:.1f}, 最大={max_face_area:.1f}")
        else:
            print("警告: 未找到有效的面部区域")
        
        if hand_areas:
            # 计算手部面积统计
            avg_hand_area = np.mean(hand_areas)
            min_hand_area = np.min(hand_areas)
            max_hand_area = np.max(hand_areas)
            print(f"手部区域面积统计: 平均={avg_hand_area:.1f}, 最小={min_hand_area:.1f}, 最大={max_hand_area:.1f}")
        else:
            print("警告: 未找到有效的手部区域")
        
        # 使用固定的经验阈值
        FACE_MIN_AREA = 2000
        FACE_MAX_AREA = 10000
        HAND_MIN_AREA = 2000
        HAND_MAX_AREA = 15000
        
        print(f"面部区域过滤范围: {FACE_MIN_AREA}-{FACE_MAX_AREA}")
        print(f"手部区域过滤范围: {HAND_MIN_AREA}-{HAND_MAX_AREA}")
        
        # 创建过滤后的裁剪图片文件路径列表
        filtered_face_paths = []
        too_small_face_count = 0
        too_big_face_count = 0
        
        # 根据面积过滤面部图片
        for frame_idx, (min_x,min_y,max_x,max_y) in face_frames.items():
            area = (max_x-min_x)*(max_y-min_y)
            face_path = f"{video_folder}/faces_v5/face_{frame_idx:06d}.jpg"
            
            # 判断面积是否在指定范围内
            if FACE_MIN_AREA <= area <= FACE_MAX_AREA:
                if os.path.exists(face_path):
                    filtered_face_paths.append(face_path)
            elif area < FACE_MIN_AREA:
                too_small_face_count += 1
            else:
                too_big_face_count += 1
        
        # 根据面积过滤手部图片
        filtered_hand_paths = []
        too_small_hand_count = 0
        too_big_hand_count = 0
        
        for frame_idx, (min_x,min_y,max_x,max_y) in hand_frames.items():
            area = (max_x-min_x)*(max_y-min_y)
            hand_path = f"{video_folder}/hands_v5/hand_{frame_idx:06d}.jpg"
            
            # 判断面积是否在指定范围内
            if HAND_MIN_AREA <= area <= HAND_MAX_AREA:
                if os.path.exists(hand_path):
                    filtered_hand_paths.append(hand_path)
            elif area < HAND_MIN_AREA:
                too_small_hand_count += 1
            else:
                too_big_hand_count += 1
        
        # 打印过滤结果
        print(f"面部图片过滤结果: 总帧数 {len(face_frames)}, 过滤后 {len(filtered_face_paths)} 帧, 移除了 {too_small_face_count} 个过小区域和 {too_big_face_count} 个过大区域")
        print(f"手部图片过滤结果: 总帧数 {len(hand_frames)}, 过滤后 {len(filtered_hand_paths)} 帧, 移除了 {too_small_hand_count} 个过小区域和 {too_big_hand_count} 个过大区域")
        
        # 如果过滤后没有图片，则使用所有图片
        if not filtered_face_paths and face_frames:
            print("警告: 面部过滤后没有图片，使用所有面部图片")
            filtered_face_paths = [f"{video_folder}/faces_v5/face_{frame_idx:06d}.jpg" 
                                   for frame_idx in face_frames.keys()
                                   if os.path.exists(f"{video_folder}/faces_v5/face_{frame_idx:06d}.jpg")]
        
        if not filtered_hand_paths and hand_frames:
            print("警告: 手部过滤后没有图片，使用所有手部图片")
            filtered_hand_paths = [f"{video_folder}/hands_v5/hand_{frame_idx:06d}.jpg"
                                  for frame_idx in hand_frames.keys()
                                  if os.path.exists(f"{video_folder}/hands_v5/hand_{frame_idx:06d}.jpg")]
        
        # 使用过滤后的图像路径列表创建视频
        self.create_video_from_crops(
            video_folder, 
            "faces_v5", 
            f"face_videos_v5/{folder_name}_{video_id}_face.mp4",
            fps,
            filtered_face_paths
        )
        
        self.create_video_from_crops(
            video_folder, 
            "hands_v5", 
            f"hand_videos_v5/{folder_name}_{video_id}_hand.mp4",
            fps,
            filtered_hand_paths
        )
        
        # 计算检测率
        face_detection_rate = face_detection_count / total_frames if total_frames > 0 else 0
        hand_detection_rate = hand_detection_count / total_frames if total_frames > 0 else 0
        
        # 计算过滤后的检测率
        filtered_face_rate = len(filtered_face_paths) / total_frames if total_frames > 0 else 0
        filtered_hand_rate = len(filtered_hand_paths) / total_frames if total_frames > 0 else 0
        
        # 打印统计信息
        print(f"视频 {video_id} 统计:")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 过滤前脸部检测数: {face_detection_count} ({face_detection_rate:.2%})")
        print(f"  - 过滤后脸部有效数: {len(filtered_face_paths)} ({filtered_face_rate:.2%})")
        print(f"  - 过滤前手部检测数: {hand_detection_count} ({hand_detection_rate:.2%})")
        print(f"  - 过滤后手部有效数: {len(filtered_hand_paths)} ({filtered_hand_rate:.2%})")
        
        if face_sizes:
            avg_face_width = sum(w for w, _ in face_sizes) / len(face_sizes)
            avg_face_height = sum(h for _, h in face_sizes) / len(face_sizes)
            print(f"  - 平均脸部尺寸: {avg_face_width:.1f}x{avg_face_height:.1f}")
        
        if hand_sizes:
            avg_hand_width = sum(w for w, _ in hand_sizes) / len(hand_sizes)
            avg_hand_height = sum(h for _, h in hand_sizes) / len(hand_sizes)
            print(f"  - 平均手部尺寸: {avg_hand_width:.1f}x{avg_hand_height:.1f}")
        
        # 记录检测率不足20%的视频
        if filtered_face_rate < 0.2:
            self.detection_fail_list.append({
                'video_path': video_source,
                'folder': folder_name,
                'video_id': video_id,
                'part_type': '脸部',
                'detection_rate': f"{filtered_face_rate:.2%}"
            })
            print(f"警告: 视频 {video_source} 中脸部有效检测率不足20%: {filtered_face_rate:.2%}")
        
        if filtered_hand_rate < 0.2:
            self.detection_fail_list.append({
                'video_path': video_source,
                'folder': folder_name,
                'video_id': video_id,
                'part_type': '手部',
                'detection_rate': f"{filtered_hand_rate:.2%}"
            })
            print(f"警告: 视频 {video_source} 中手部有效检测率不足20%: {filtered_hand_rate:.2%}")

    def create_video_from_crops(self, video_folder, crop_folder, output_path, fps, filtered_list=None):
        """从裁剪的图片创建视频"""
        # 获取所有裁剪图片
        crop_images = sorted(glob(f"{video_folder}/{crop_folder}/*.jpg"))
        if not crop_images:
            print(f"警告: 在 {video_folder}/{crop_folder} 中未找到裁剪图片，跳过视频生成")
            return
        
        # 如果提供了过滤列表，则直接使用过滤列表中的路径
        if filtered_list is not None and filtered_list:
            # 确保所有路径都存在
            filtered_images = [path for path in filtered_list if os.path.exists(path)]
            if filtered_images:
                print(f"使用过滤后的 {len(filtered_images)} 张图片生成视频")
                crop_images = filtered_images
            else:
                print(f"警告: 过滤后的图片路径都不存在，使用所有裁剪图片")
        
        # 确定尺寸（使用第一张图片的尺寸）
        first_img = None
        for img_path in crop_images:
            try:
                first_img = cv2.imread(img_path)
                if first_img is not None:
                    break
            except Exception as e:
                print(f"警告: 读取图片 {img_path} 失败: {e}")
                continue
                
        if first_img is None:
            print(f"警告: 无法读取任何图片，跳过视频生成 {output_path}")
            return
        
        height, width = first_img.shape[:2]
        
        # 创建视频写入器 - 直接生成MP4文件
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 使用mp4v编码器生成MP4文件
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            if not writer.isOpened():
                print(f"警告: 无法创建视频写入器 {output_path}，尝试使用其他编码器")
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    fps,
                    (width, height)
                )
            
            # 添加所有图片到视频
            frame_count = 0
            for img_path in tqdm(crop_images, desc=f"创建视频 {output_path}", leave=False):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告: 无法读取图片 {img_path}，跳过")
                        continue
                        
                    # 确保尺寸一致
                    if img.shape[:2] != (height, width):
                        img = cv2.resize(img, (width, height))
                    
                    # 写入帧
                    writer.write(img)
                    frame_count += 1
                except Exception as e:
                    print(f"警告: 处理图片 {img_path} 失败 - {e}")
            
            # 释放视频写入器
            writer.release()
            
            if frame_count > 0:
                print(f"成功生成视频: {output_path}，包含 {frame_count} 帧")
            else:
                print(f"警告: 未写入任何帧到视频 {output_path}")
            
        except Exception as e:
            print(f"错误: 创建视频失败 - {e}")
            import traceback
            traceback.print_exc()

    def save_detection_fail_report(self):
        """保存检测失败的视频信息到CSV文件"""
        if not self.detection_fail_list:
            print("所有视频的手部和脸部检测率均达标")
            return
            
        # 保存到CSV
        with open(DETECTION_FAIL_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_path', 'folder', 'video_id', 'part_type', 'detection_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in self.detection_fail_list:
                writer.writerow(record)
        
        print(f"已将检测率不足20%的视频记录保存到 {DETECTION_FAIL_CSV}")

    def run(self):
        """运行处理流程"""
        try:
            print("步骤 1: 处理视频并提取脸部和手部区域")
            self.process_videos()
            
            print("步骤 2: 保存检测失败报告")
            self.save_detection_fail_report()
            
            print("处理完成!")
        except Exception as e:
            import traceback
            print(f"错误: 处理过程中发生异常 - {e}")
            print("详细错误信息:")
            traceback.print_exc()
            print("尝试保存已处理的数据...")
            self.save_detection_fail_report()
            print("部分处理完成。")

if __name__ == "__main__":
    print("="*50)
    print("视频处理工具 v4 - 使用DWPose提取脸部和手部区域")
    print("="*50)
    
    # 创建VideoProcessor实例并运行
    try:
        processor = VideoProcessor()
        processor.run()
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)