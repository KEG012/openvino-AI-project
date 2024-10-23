#%%

# import 코드 블록

from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deep_sort_realtime.deepsort_tracker import DeepSort
# from torchreid.reid.utils import FeatureExtractor
# from torchreid import utils
# from torchreid import reid_model_factory
import torchreid


#%%

# intel realsense data 수신을 위한 코드 블록

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11n-pose.pt").to(device)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

# datamanager = torchreid.data.ImageDataManager(
#     root = "reid-data",
#     sources = "market1501",
#     targets="market1501",
#     height=256,
#     width=128,
#     batch_size_train=32,
#     batch_size_test=100,
#     transforms=["random_flip", "random_crop"]
# )

# extractor = utils.FeatureExtractor(
#     model_name = 'osnet_x1_0',
#     device = device
# )

tracker = DeepSort(
    max_age = 30,
    n_init = 5,
    max_iou_distance = 0.6,
    nms_max_overlap=0.8,
    embedder = 'mobilenet',
    embedder_gpu=device =='cuda'
    )

mean_color_arry = [0, 0]

#%%

# hand gesture용 코드 블록

model_path = './gesture_recognizer.task'

#Hand land mark 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Gesture Recognizer 설정
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

global recognized_gesture
recognized_gesture = 'No Gesture'

def handGestueInit():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handGestureResult
    )
    return GestureRecognizer.create_from_options(options)

def handGestureResult(result: GestureRecognizerResult,output_image: mp.Image, timestamp_ms: int):
    global recognized_gesture
    if result.gestures and len(result.gestures) > 0:
        recognized_gesture = result.gestures[0][0].category_name
    else:
        recognized_gesture = 'No Gesture'

def draw_hand_landmarks(img,results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                if id == 0:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)# 초록색 원
            mp_drawing.draw_landmarks(img,handLms,mp_hands.HAND_CONNECTIONS)

def showHandGesture(recognizer,color_image):   
    # BGR 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # MediaPipe의 Image 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    #UI에 인식한 손의 landmark 표시
    results = hands.process(rgb_image)
    draw_hand_landmarks(color_image,results)
    
    recognizer.recognize_async(mp_image, frame_timestamp_ms)
    cv2.putText(color_image, recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
#%%

def set_distance(z):
    
    if z <= 1.5:
        print("Move Backward")
        return 0 #TODO: 후진방향으로 명령
    
    else:
        print("mover Foward")
        return 0 #TODO: 전진방향으로 명령

def move_center_position(x_center, f_center_min, f_center_max):
    
    if x_center < f_center_min:
        print("Turn Right")
        return 0 #TODO: 오른쪽으로 명령
    
    elif x_center > f_center_max:
        print("Turn Left")
        return 0 #TODO: 왼쪽으로 명령
    
    else: 
        print("Stay Center")
        return 0 #TODO: 그대로 명령

def mean_color(color_img, frame_ROI):
    
    if not frame_ROI:
        return [0, 0]
    
    for roi in frame_ROI:
        x1, y1, x2, y2 = roi

        cropped_img = color_img[y1:y2, x1:x2]
        cropped_HSV_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        
        if mean_color_arry[0] == 0 and mean_color_arry[1] == 0:
            mean_color_arry[0] = tuple(map(int, cv2.mean(cropped_img)[:3]))
            mean_color_arry[1] = tuple(map(int, cv2.mean(cropped_HSV_img)[:3]))
        else:
            print("")
    
    return mean_color_arry

def yolo_box_detector(results):
    detections = []
    
    if not results:
        return np.array(detections)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            clas = box.cls[0]
            detections.append([x1, y1, x2-x1, y2-y1, conf, clas])
    
    return np.array(detections)

def run_extract_pose(use_popup):
    recognizer = handGestueInit()
    try:
        while True:
            aligned_frames = get_frames()
        
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            flipped_color_image = cv2.flip(color_image, 1)

            results: Results = pose_model(source=flipped_color_image, conf=0.3, verbose=False)
            detections = yolo_box_detector(results)
            
            tracks = tracker.update_tracks(detections, frame=flipped_color_image)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                x, y, w, h = map(int, track.to_tlbr())

                # 박스와 ID 표시
                cv2.rectangle(flipped_color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(flipped_color_image, f"ID: {track_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 손 제스처 표시
            showHandGesture(recognizer, flipped_color_image)

            combined_image = np.hstack((flipped_color_image, cv2.applyColorMap(
                cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)))

            if use_popup:
                cv2.imshow('Color and Depth Combined Image', combined_image)

                if cv2.waitKey(10) & 0xFF == 27:  # ESC 키로 종료
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()                
                
'''
def calculate_box(color_img, depth_img, results):
    
    z, x_center = 0, 0
    color_ROI = []
    
    for result in results:
        try:
            keypoints = result.keypoints.cpu().numpy()
            boxes = result.boxes.cpu().numpy()
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                cv2.rectangle(color_img,
                              (x1, y1), (x2, y2),
                              (0,255,0), 2, cv2.LINE_AA)
                
                human_box_width_center = (x2 + x1) // 2
                person_box_idx = f"person {idx + 1}"
                
                cv2.putText(color_img, person_box_idx,
                            (human_box_width_center - 50, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,255,0), 2, cv2.LINE_AA)

            for _, keypoint in enumerate(keypoints):
                
                left_shoulder = keypoint.xy[0][5][:2]
                right_shoulder = keypoint.xy[0][6][:2]
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

                    z = depth_img[y_center, x_center] * depth_scale

                    depth_label = f"depth : {z:.1f}"
                    cv2.putText(color_img, depth_label, (x_center - 50, y_center - 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                
                else:
                    print("Can't detect all shoulder and hip")
                
        except IndexError:                    
            print("No landmarks in here")
    
    return color_img, z, color_ROI, x_center
'''


'''
def extract_depth(color_img ,depth_img, x, y):
    
    z = depth_img[x,y] * depth_scale
    
    return z
'''

def get_frames():
    
    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
    except RuntimeError as e:
        print(f"Frame sync error: {e}")
        pipeline.stop()
        time.sleep(1)  #restart after delay
        pipeline.start(config)
        return None
    
    return aligned_frames


'''
def run_extract_pose(use_popup):
    
    recognizer = handGestueInit()
    
    try:
        while True:

            aligned_frames = get_frames()
        
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            print("g")  
  
            flipped_depth_image = cv2.flip(aligned_depth_image, 1)
            flipped_color_image = cv2.flip(color_image, 1)
            
            flipped_depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(flipped_depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # start_time = time.time()
            results: Results = pose_model(source = flipped_color_image, conf = 0.3, verbose=False)
            
            flipped_color_image, z, color_ROI, x_center = calculate_box(flipped_color_image, flipped_depth_image, results)
            print(color_ROI)
            
            if np.any(color_ROI):
                mean_color_arry = mean_color(flipped_color_image, color_ROI)
                print(mean_color_arry)
            
            f_height, f_width = flipped_color_image.shape[:2]

            # center line using center position to set the position in center ratio
            f_width_center = f_width // 2
            f_width_ratio = f_width // 20
            f_center_min = f_width_center - f_width_ratio
            f_center_max = f_width_center + f_width_ratio
            
            cv2.line(flipped_color_image, (f_center_min, 0),
                    (f_center_min, f_height), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(flipped_color_image, (f_center_max, 0),
                    (f_center_max, f_height), (255, 255, 255), 1, cv2.LINE_AA)

            showHandGesture(recognizer, flipped_color_image)
                        
            if x_center is not 0:
                move_direction_xside = move_center_position(x_center, f_center_min, f_center_max)
            
            if z is not 0:
                move_direction_yside = set_distance(z)

            # if hand_gestrue == 0:
            #     mean_color_arry = [0, 0]
            
            combined_image = np.hstack((flipped_color_image, flipped_depth_colormap)) 
            
            if use_popup:
                cv2.imshow('Color and depth combined image', combined_image)

                # 'ESC'를 누르면 종료
                if cv2.waitKey(10) & 0xFF == 27:
                    break
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
'''


run_extract_pose(use_popup=True)
