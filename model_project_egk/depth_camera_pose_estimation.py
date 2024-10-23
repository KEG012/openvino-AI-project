import pyrealsense2 as rs
import numpy as np
import cv2
import time
import collections
import pandas as pd

from openvino.runtime import Core
import torch

from pose_estimation import process_results, draw_poses

Ie = Core()

model_path = "./model/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.xml"

# human_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./openvino-AI-project/human_detection.pt', _verbose=False, force_reload=True)
color_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./openvino-AI-project/color_detection.pt', _verbose=False, force_reload=True)
color_classes = ["BLACK", "YELLOW", "BROWN", "GREEN", "ORANGE", "PINK", "PURPLE", "RED", "WHITE", "BLUE"]

pose_model = Ie.read_model(model_path)
compiled_model = Ie.compile_model(pose_model, device_name = "CPU")

input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

height, width = list(input_layer.shape)[2:]

pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")

# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# 스트림 활성화 (Depth와 Color)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
profile = pipeline.start(config)

# depth camera의 센서 내부 파라미터를 가지고 오기 위한 작업
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 카메라 파라미터
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
camera_intrinsics = {'fx': intrinsics.fx,
                     'fy': intrinsics.fy,
                     'cx': intrinsics.ppx,
                     'cy': intrinsics.ppy}

# print("Camera Intrinsics: ", camera_intrinsics)

# 깊이 데이터를 RGB 프레임에 맞추기 위한 align 객체 생성
align_to = rs.stream.color
align = rs.align(align_to)

processing_times = collections.deque()

def get_keypoint_depths(depth_image, pose, point_score_threshold):
    keypoint_depths = []

    points = pose[:, :2].astype(np.int32)
    scores = pose[:, 2]
    
    left_shoulder_idx = 5
    right_shoulder_idx = 6
    left_elbow_idx = 7
    right_elbow_idx = 8
    left_hip_idx = 11
    right_hip_idx = 12
    
    keypoints = [left_shoulder_idx, right_shoulder_idx,
                 left_elbow_idx, right_elbow_idx,
                 left_hip_idx, right_hip_idx]
    
    for i in keypoints:
        x, y = points[i]
        v = scores[i]
        if v > point_score_threshold:
            if 0 < x < depth_image.shape[1] and 0 < y < depth_image.shape[0]:                
                z = depth_image[y,x] * depth_scale
                keypoint_depths.append((i,x,y,z))
                
    return keypoint_depths

def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return round(distance, 3)

def calculate_body_distances(keypoint_depths):
    keypoint_dict = {i: (x, y, z) for i, x, y, z in keypoint_depths}
    
    left_shoulder_idx = 5
    right_shoulder_idx = 6
    left_elbow_idx = 7
    right_elbow_idx = 8
    left_hip_idx = 11
    right_hip_idx = 12

    distances = {}
    
    if left_shoulder_idx in keypoint_dict and right_shoulder_idx in keypoint_dict:
        distances['shoulder_width'] = calculate_distance(keypoint_dict[left_shoulder_idx], keypoint_dict[right_shoulder_idx])
    
    if left_hip_idx in keypoint_dict and right_hip_idx in keypoint_dict:
        distances['hip_width'] = calculate_distance(keypoint_dict[left_hip_idx], keypoint_dict[right_hip_idx])
    
    if left_shoulder_idx in keypoint_dict and left_elbow_idx in keypoint_dict:
        distances['left_shoulder_to_left_elbow'] = calculate_distance(keypoint_dict[left_elbow_idx], keypoint_dict[left_shoulder_idx])
    
    if right_shoulder_idx in keypoint_dict and right_elbow_idx in keypoint_dict:
        distances['right_shoulder_to_right_elbow'] = calculate_distance(keypoint_dict[right_elbow_idx], keypoint_dict[right_shoulder_idx])
    
    if left_shoulder_idx in keypoint_dict and left_hip_idx in keypoint_dict:
        distances['left_shoulder_to_left_hip'] = calculate_distance(keypoint_dict[left_shoulder_idx], keypoint_dict[left_hip_idx])
    
    if right_shoulder_idx in keypoint_dict and right_hip_idx in keypoint_dict:
        distances['right_shoulder_to_right_hip'] = calculate_distance(keypoint_dict[right_shoulder_idx], keypoint_dict[right_hip_idx])
    
    return distances

def pixel_to_real(x,y,z,camera_parameter):
    fx = camera_parameter['fx']
    fy = camera_parameter['fy']
    cx = camera_parameter['cx']
    cy = camera_parameter['cy']
    
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    
    return X, Y, Z

distance_dataset = []
seen_people = set()

try:
    while True:
        # 프레임 받기
        frames = pipeline.wait_for_frames()
        # 프레임 정렬
        aligned_frames = align.process(frames)

        # 정렬된 depth와 color 프레임 가져오기
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # numpy 배열로 변환
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth_colormap 생성
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # pose estimation을 위한 input image 준비
        input_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_AREA)  # (width, height)
        input_image = input_image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        input_image = np.expand_dims(input_image, axis=0)  # (1, C, H, W)

        # processing pose estimation
        start_time = time.time()
        results = compiled_model([input_image])
        
        # output pafs and heatmaps in process
        pafs = results[pafs_output_key]
        heatmaps = results[heatmaps_output_key]
        
        # output poses using process results
        poses, scores = process_results(color_image, pafs, heatmaps)

        for person_idx, pose in enumerate(poses):
            keypoint_depths = get_keypoint_depths(aligned_depth_image, pose, 0.1)
        
            '''
            # pixel 좌표와 z좌표가 정확하게 나오는지 확인하기 위한 debug code
            for depth in keypoint_depths:
                i, x, y, z = depth
                print(f"{i}: {z:.2f} meters in {x}, {y}")
            '''
            
            # keypoint의 pixel 값을 real 값으로 변환
            real_world_keypoints = []
            
            for keypoint in keypoint_depths:
                i, x, y, z = keypoint
                real_world_point = pixel_to_real(x,y,z,camera_intrinsics)
                real_world_keypoints.append((i, *real_world_point))
        
            '''
            # realworld 좌표를 확인하기 위한 debug code
            for real_point in real_world_keypoints:
                i, rx, ry, rz = real_point
                print(f"{i}: {rz:.2f} meters in {rx:.2f}, {ry:.2f}")
            '''
            
            # distance를 real world coordinate로 계산 및 출력
            landmark_distances = calculate_body_distances(real_world_keypoints)
            
            # if person_idx not in seen_people:
            #     seen_people.add(person_idx)
            row = {'person_id': person_idx + 1,
                    'shoulder_width': landmark_distances.get('shoulder_width', 0),
                    'hip_width': landmark_distances.get('hip_width', 0),
                    'left_shoulder_to_left_elbow': landmark_distances.get('left_shoulder_to_left_elbow', 0),
                    'right_shoulder_to_right_elbow': landmark_distances.get('right_shoulder_to_right_elbow', 0),
                    'left_shoulder_to_left_hip': landmark_distances.get('left_shoulder_to_left_hip', 0),
                    'right_shoulder_to_right_hip': landmark_distances.get('right_shoulder_to_right_hip', 0)
                    }
            distance_dataset.append(row)
            
            # distance 값이 정확하게 나오는지 확인하기 위한 debug code
            print(f"Person {person_idx + 1}: ")
            for part, distance in landmark_distances.items():
                print(f"{part}: {distance:.2f} meters")
        
        # draw pose and covered in color image
        color_image = draw_poses(color_image, poses, 0.1)
        stop_time = time.time()
        
        # fps calculation
        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()
        
        # put the fps in the image
        _, f_width = color_image.shape[:2]
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(
            color_image,
            f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            (20,40),
            cv2.FONT_HERSHEY_COMPLEX,
            f_width / 1000,
            (0,0,255),
            1,
            cv2.LINE_AA
        )
        
        # color image and depth colormap merge on horizontal
        combined_image = np.hstack((color_image, depth_colormap))

        # 병합된 이미지 출력
        cv2.imshow('Color and Depth with Pose Estimation', combined_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 파이프라인 정리
    pipeline.stop()
    cv2.destroyAllWindows()
    
    distance_df = pd.DataFrame(distance_dataset)
    
    print(distance_df)
    
    distance_df.to_csv('landmark_distances.csv', index = False)