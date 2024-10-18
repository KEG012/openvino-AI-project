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

#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11s-pose.pt").to(device)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

'''
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
camera_intrinsics = {'fx': intrinsics.fx,
                     'fy': intrinsics.fy,
                     'cx': intrinsics.ppx,
                     'cy': intrinsics.ppy}
'''

#%%

model_path = './gesture_recognizer.task'

# Hand land mark 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

previous_timestamp_ms = 0
gesture_start_time = None
GESTURE_HOLD_TIME = 3

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

def showHandGesture(recognizer,color_img, depth_img):   
    global previous_timestamp_ms
        
    # BGR 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # MediaPipe의 Image 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    # frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    frame_timestamp_ms = int(time.time() * 1000)
    
    if frame_timestamp_ms <= previous_timestamp_ms:
        frame_timestamp_ms = previous_timestamp_ms + 1 #frame_timestanp_ms가 더 커야 하는데 더 작을 경우에 대한 예외 처리
    
    previous_timestamp_ms = frame_timestamp_ms
    
    #UI에 인식한 손의 landmark 표시
    hand_results = hands.process(rgb_image)
    closest_hand = find_closest_hand(hand_results, depth_img) #가장 가까이 있는 사람의 손을 찾음 hand results 중에서 가장 가까운 hand results를 구함
    
    if closest_hand != None:
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        
        for landmarks in closest_hand.landmark:
            x = int(landmarks.x * color_img.shape[1])
            y = int(landmarks.y * color_img.shape[0])
            cv2.circle(color_img, (x,y), 5, (0,255,0), -1)
            
        cv2.putText(color_img, recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return closest_hand

def find_closest_hand(results, depth_img):
    
    if not results.multi_hand_landmarks:
        return None
    
    min_distance = float('inf')
    closest_hand = None
    
    for hand_landmarks in results.multi_hand_landmarks:
        wrist = hand_landmarks.landmark[0]
        if wrist.x > 1.0:
            wrist.x = 0.999
        if wrist.y > 1.0:
            wrist.y = 0.999
        x,y = int(wrist.x * depth_img.shape[1]), int(wrist.y * depth_img.shape[0])
        z = depth_img[y, x] * depth_scale

        if z < min_distance:
            min_distance = z
            closest_hand = hand_landmarks
                
    return closest_hand

def track_victory_gesture():
    global gesture_start_time
    
    if recognized_gesture == "Victory":
        if gesture_start_time is None:
            gesture_start_time = time.time()
            print(gesture_start_time)
        elif time.time() - gesture_start_time >= GESTURE_HOLD_TIME:
            return True
    else:
        gesture_start_time = None
    
    return False

#%%

def mean_color(color_img, person_color_roi):
    
    mean_color_arry = [0,0]
    
    if not person_color_roi:
        return [0, 0]

    rx1, ry1, rx2, ry2 = person_color_roi
    cropped_image = color_img[ry1:ry2, rx1:rx2]
    cropped_HSV_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    if mean_color_arry[0] == 0 and mean_color_arry[1] == 0:
        mean_color_arry[0] = tuple(map(int, cv2.mean(cropped_image)[:3]))
        mean_color_arry[1] = tuple(map(int, cv2.mean(cropped_HSV_image)[:3]))

    return mean_color_arry

def run_extract_box(color_img, results):
    
    box_info = []
        
    if not results:
        return box_info, color_img
    

    try:
        boxes = results[0].boxes.cpu().numpy()
        
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
    

    try:
        keypoints = results[0].keypoints.cpu().numpy()
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

def find_hand_person(closest_hand, box_info, color_img, depth_img):
    hand_person_box = None
    min_combined_metric = float('inf')

    if closest_hand is None:
        # print("Can't detect hands")
        return None
    
    wrist = closest_hand.landmark[0]
    wrist_x = int(wrist.x * color_img.shape[1])
    wrist_y = int(wrist.y * color_img.shape[0])
    wrist_depth = depth_img[wrist_y, wrist_x] * depth_scale

    for idx, box in box_info:
        x1, y1, x2, y2 = map(int, box)
        
        box_x_center = (x1 + x2) // 2
        box_y_center = (y1 + y2) // 2
        box_depth = depth_img[box_y_center, box_x_center] * depth_scale

        distance = np.sqrt((wrist_x - box_x_center) ** 2 + (wrist_y - box_y_center) ** 2)

        combined_metric = distance + abs(wrist_depth - box_depth)

        if combined_metric < min_combined_metric:
            min_combined_metric = combined_metric
            hand_person_box = (x1, y1, x2, y2)

    return hand_person_box

def find_the_hand_person_roi(color_ROI, hand_person_box):
    
    if hand_person_box is None:
        return None
    
    hand_person_roi = None
    
    x1, y1, x2, y2 = hand_person_box
    
    for idx, roi in enumerate(color_ROI):
        rx1, ry1, rx2, ry2 = roi
        rx_center = int((rx1 + rx2) // 2)
        ry_center = int((ry1 + ry2) // 2)
        if x1 <= rx_center <= x2 and y1 <= ry_center <= y2:
            hand_person_roi = (rx1, ry1, rx2, ry2)
    
    return hand_person_roi

#%%

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
            
            depth_image = cv2.flip(aligned_depth_image, 1)
            color_image = cv2.flip(color_image, 1)
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            results = pose_model(source=color_image, conf=0.3, verbose=False)  #Pose_model은 골반과 hip 좌표를 얻기 위하여
            
            color_image, box_info = run_extract_box(color_image, results) #run_extract_box 는 사람에게 박스를 쳐줌
            color_image, color_ROI, z, x_center = run_extract_pose(color_image, depth_image, results) #run_extract_pose 함수는 사람 내부 ROI를 따고, 좌표 값 추출
            closest_hand = showHandGesture(recognizer, color_image, depth_image) #showHandGesture 함수는 손에 랜드마크를 그려주고, hand gesture의 결과를 반환함
            
            hand_person_box = find_hand_person(closest_hand, box_info, color_image, depth_image)
            person_color_roi = find_the_hand_person_roi(color_ROI, hand_person_box)
            
            if track_victory_gesture():
                mean_rgb, mean_hsv = mean_color(color_image, person_color_roi)
                print(mean_rgb)
                print(mean_hsv)
            
            
            
            
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

