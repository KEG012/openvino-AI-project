from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import pickle

detect_name = "Unknown"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11s-pose.pt").to(device)
model_path = "/home/keg/workspace/openvino-AI-project/distance_model1.pkl"
with open(model_path, 'rb') as f:
    rf_model = pickle.load(f)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
camera_intrinsics = {'fx': intrinsics.fx,
                     'fy': intrinsics.fy,
                     'cx': intrinsics.ppx,
                     'cy': intrinsics.ppy}

def run_extract_box(color_img, results):
    
    box_info = []
        
    if not results:
        return box_info, color_img
    
    for result in results:
        try:
            boxes = result.boxes.cpu().numpy()
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                box_info.append((idx, (x1, y1, x2, y2)))

                label = f"person : {idx + 1}"
                
                cv2.rectangle(color_img, (x1,y1), (x2, y2), (0,255,0), 2, cv2.LINE_AA)
                
                cv2.putText(color_img, label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
        except IndexError:
            print("No boxs in here")
            
    return color_img, box_info

def run_extract_pose(color_img, depth_img, results):
    
    z,x_center = 0, 0
    color_ROI = []
    
    if not results:
        return color_img, color_ROI, z, x_center
    
    for result in results:
        try:
            keypoints = result.keypoints.cpu().numpy()
            for _, keypoint in enumerate(keypoints):
                left_shoulder = keypoint.xy[0][5][:2]
                right_shoulder = keypoint.xy[0][6][:2]
                left_elbow = keypoint.xy[0][7][:2]
                right_elbow = keypoint.xy[0][8][:2]
                left_hip = keypoint.xy[0][11][:2]
                right_hip = keypoint.xy[0][12][:2]
                
                if all(np.any(pt) for pt in [left_shoulder, right_shoulder, left_hip, right_hip]):
                    valid_points = [left_shoulder, right_shoulder, left_hip, right_hip]

                    x_min, y_min = np.min(valid_points, axis=0).astype(int)
                    x_max, y_max = np.max(valid_points, axis=0).astype(int)
                    
                    color_ROI.append((x_min, y_min, x_max, y_max))
                    
                    # 어깨와 골반의 최소 최대 좌표를 통하여 사각형 그림
                    cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max),
                                (0, 0, 255), 2)
                    
                    x_center = (x_max + x_min) // 2
                    y_center = (y_max + y_min) // 2
                    cv2.circle(color_img, (x_center, y_center), 3, (0, 0, 255), 2)
                    
                    if depth_scale != 0:
                        z = depth_img[y_center, x_center] * depth_scale
                    else:
                        z = depth_img[y_center, x_center] * 0.001
                    
                    depth_label = f"depth : {z:.1f}"
                    cv2.putText(color_img, depth_label, (x_center - 50, y_center - 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                # else:
                #     print("Can't detect all shoulder and hip")
        except IndexError:
            print("No landmarks in here")

    return color_img, color_ROI, z, x_center

def point_depth(depth_img, x, y):
    
    points = {}
    
    x,y = int(x), int(y)
    
    z = depth_img[y, x] * depth_scale
    points = (x,y,z)
    
    return points

def pixel_to_real(x,y,z,camera_parameter):
    
    fx = camera_parameter['fx']
    fy = camera_parameter['fy']
    cx = camera_parameter['cx']
    cy = camera_parameter['cy']
    
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    
    return X, Y, Z

def find_real_world_keypoint(results, depth_img):
    
    real_world_keypoint = []
    
    if not results:
        print("No keypoints detected.")
        return real_world_keypoint
    
    keypoints = results[0].keypoints.cpu().numpy()
    
    for idx, keypoint in enumerate(keypoints):
        try:
            left_shoulder = keypoint.xy[0][5][:2].astype(int)
            right_shoulder = keypoint.xy[0][6][:2].astype(int)
            left_elbow = keypoint.xy[0][7][:2].astype(int)
            right_elbow = keypoint.xy[0][8][:2].astype(int)
            left_hip = keypoint.xy[0][11][:2].astype(int)
            right_hip = keypoint.xy[0][12][:2].astype(int)
            
            person_keypoint = [("left_shoulder", left_shoulder),
                            ("right_shoulder", right_shoulder),
                            ("left_elbow", left_elbow),
                            ("right_elbow", right_elbow),
                            ("left_hip", left_hip),
                            ("right_hip", right_hip)]
            
            for name, (x, y) in person_keypoint:
                try:
                    point = point_depth(depth_img, x, y)
                    x,y,z = point
                    real_world_position = pixel_to_real(x,y,z,camera_intrinsics)
                    real_world_keypoint.append((name, *real_world_position))
                except Exception as e:
                    print("Index is wrong")
        except Exception as e:
            print("Error occured")
            
    return real_world_keypoint

def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return round(distance, 3)
    
def calculate_body_distances(real_world_keypoints):
    
    # all_distances = []
    
    distances = []
    
    keypoint_dict = {i: (x,y,z) for i,x,y,z in real_world_keypoints}

    left_shoulder = "left_shoulder"
    right_shoulder = "right_shoulder"
    left_elbow = "left_elbow"
    right_elbow = "right_elbow"
    left_hip = "left_hip"
    right_hip = "right_hip"
    
    new_keypoint_dict = list(keypoint_dict.keys())

    if left_shoulder in keypoint_dict and right_shoulder in keypoint_dict:
        # distances['shoulder_width'] = calculate_distance(keypoint_dict[left_shoulder], keypoint_dict[right_shoulder])
        distance = calculate_distance(keypoint_dict[left_shoulder], keypoint_dict[right_shoulder])
        distances.append(distance)
        
    if left_hip in keypoint_dict and right_hip in keypoint_dict:
        # distances['hip_width'] = calculate_distance(keypoint_dict[left_hip], keypoint_dict[right_hip])
        distance = calculate_distance(keypoint_dict[left_hip], keypoint_dict[right_hip])
        distances.append(distance)
    
    if left_shoulder in keypoint_dict and left_elbow in keypoint_dict:
        # distances['left_shoulder_to_left_elbow'] = calculate_distance(keypoint_dict[left_elbow], keypoint_dict[left_shoulder])
        distance = calculate_distance(keypoint_dict[left_elbow], keypoint_dict[left_shoulder])
        distances.append(distance)
        
    if right_shoulder in keypoint_dict and right_elbow in keypoint_dict:
        # distances['right_shoulder_to_right_elbow'] = calculate_distance(keypoint_dict[right_elbow], keypoint_dict[right_shoulder])
        distance = calculate_distance(keypoint_dict[right_elbow], keypoint_dict[right_shoulder])
        distances.append(distance)
    
    if left_shoulder in keypoint_dict and left_hip in keypoint_dict:
        # distances['left_shoulder_to_left_hip'] = calculate_distance(keypoint_dict[left_shoulder], keypoint_dict[left_hip])
        distance = calculate_distance(keypoint_dict[left_shoulder], keypoint_dict[left_hip])
        distances.append(distance)
        
    if right_shoulder in keypoint_dict and right_hip in keypoint_dict:
        # distances['right_shoulder_to_right_hip'] = calculate_distance(keypoint_dict[right_shoulder], keypoint_dict[right_hip])
        distance = calculate_distance(keypoint_dict[right_shoulder], keypoint_dict[right_hip])
        distances.append(distance)
    
    # all_distances.append(distances)
        
    return distances

def get_frames():
    
    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
    except RuntimeError as e:
        print(f"Frame sync error: {e}")
        pipeline.stop()
        time.sleep(1)  #restart after delay
        pipeline.start(config)
    
    return aligned_frames

def run_mainstream(use_popup=True):
    
    global detect_name
    
    try:
        while True:
            aligned_frames = get_frames()
            
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            depth_image = cv2.flip(aligned_depth_image, 1)
            color_image = cv2.flip(color_image, 1)
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            results: Results = pose_model(source=color_image, conf=0.3, verbose=False)
            
            color_image, box_info = run_extract_box(color_image, results)
            color_image, color_ROI, z, x_center = run_extract_pose(color_image, depth_image, results)

            real_world_keypoints = find_real_world_keypoint(results, depth_image)
            distances = calculate_body_distances(real_world_keypoints)
            distances = np.array(distances)
            distances = distances.reshape(1,-1)
            
            if distances.size > 0:
                distances = np.array(distances).reshape(1,-1)
                detect = rf_model.predict(distances)
                if detect == 1:
                    detect_name = f"Detect : KEG"
                if detect == 2:
                    detect_name = f"Detect : KEC"
                if detect == 3:
                    detect_name = f"Detect : KDH"
            else:
                print("No prediction")
            
            cv2.putText(color_image, detect_name, (420, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            
            f_height, f_width = color_image.shape[:2]

            # center line using center position to set the position in center ratio
            f_width_center = f_width // 2
            f_width_ratio = f_width // 20
            f_center_min = f_width_center - f_width_ratio
            f_center_max = f_width_center + f_width_ratio
            
            cv2.line(color_image, (f_center_min, 0),
                    (f_center_min, f_height), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(color_image, (f_center_max, 0),
                    (f_center_max, f_height), (255, 255, 255), 1, cv2.LINE_AA)
            
            combined_image = np.hstack((color_image, depth_colormap))

            if use_popup:
                cv2.imshow('Color and depth combined image', combined_image)

                # 'ESC'를 누르면 종료
                if cv2.waitKey(10) & 0xFF == 27:
                    break
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

run_mainstream(use_popup=True)