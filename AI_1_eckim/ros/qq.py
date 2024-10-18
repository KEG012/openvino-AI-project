import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socket
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch


# TCP 클라이언트 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.10.16.163', 5000))  # 서버의 IP 주소

def receive_data(sock):
    # 데이터 크기 수신
    data_size = int.from_bytes(sock.recv(4), byteorder='big')
    # 데이터 수신
    data = b""
    while len(data) < data_size:
        packet = sock.recv(data_size - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_data(sock, data):
    """서버로 데이터를 전송하는 함수"""
    data_bytes = data.encode()
    data_size = len(data_bytes)
    sock.sendall(data_size.to_bytes(4, byteorder='big'))  # 데이터 크기 전송
    sock.sendall(data_bytes)  # 데이터 전송

model_path = 'gesture_recognizer.task'

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
    
        
def showImage(color_image, depth_image):
    cv2.imshow("Color Image", color_image)
    cv2.imshow("Depth Image (Colormap)", depth_image)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11n-pose.pt").to(device)
depth_scale, color_data, depth_data = 0, 0, 0
mean_color_arry = [0, 0]
mean_color_arry_first = [0, 0]
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
        

        mean_color_arry[0] = tuple(map(int, cv2.mean(cropped_img)[:3]))
        mean_color_arry[1] = tuple(map(int, cv2.mean(cropped_HSV_img)[:3]))

    
    return mean_color_arry

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
                    
                    if depth_scale is not 0:
                        z = depth_img[y_center, x_center] * depth_scale
                    else:
                        z = depth_img[y_center, x_center] * 0.001
                    
                    depth_label = f"depth : {z:.1f}"
                    cv2.putText(color_img, depth_label, (x_center - 50, y_center - 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                
                else:
                    print("Can't detect all shoulder and hip")
                
        except IndexError:                    
            print("No landmarks in here")
    
    return color_img, z, color_ROI, x_center

def main_program():
    recognizer = handGestueInit()
    FirstmeanFlag=0
    try:  # 종료 조건을 루프 내부로 이동
        while True:
            # 컬러 이미지 수신
            color_data = receive_data(client_socket)
            if color_data is None:
                break
            
            # 깊이 이미지 수신
            depth_data = receive_data(client_socket)
            if depth_data is None:
                break
            
            # JPEG 디코딩
            color_image = cv2.imdecode(np.frombuffer(color_data, np.uint8), cv2.IMREAD_COLOR)
            # PNG 디코딩
            depth_image = cv2.imdecode(np.frombuffer(depth_data, np.uint16), cv2.IMREAD_UNCHANGED)
            
            
            # flipped_depth_image = cv2.flip(aligned_depth_image, 1)
            # flipped_color_image = cv2.flip(color_image, 1)
            
            depth_image_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            results = pose_model(source = color_image, conf = 0.3, verbose=False)
            color_image, z, color_ROI, x_center = calculate_box(color_image, depth_image, results)
            if np.any(color_ROI):
                if FirstmeanFlag==0:
                    mean_color_arry_first=mean_color(color_image, color_ROI)
                    FirstmeanFlag=1

                
                mean_color_arry = mean_color(color_image, color_ROI)
                print(mean_color_arry)

                cv2.putText(color_image, f"{mean_color_arry[0][0]}", (250, 250),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image,f"{mean_color_arry[0][1]}", (300, 250),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image, f"{mean_color_arry[0][2]}", (350, 250),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image, f"{mean_color_arry[1][0]}", (250, 300),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image, f"{mean_color_arry[1][1]}", (300, 300),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image,f"{mean_color_arry[1][2]}", (350, 300),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                
                cv2.putText(color_image, f"{mean_color_arry_first[0][0]}", (250, 350),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image,f"{mean_color_arry_first[0][1]}", (300, 350),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image, f"{mean_color_arry_first[0][2]}", (350, 350),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image, f"{mean_color_arry_first[1][0]}", (250, 400),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image, f"{mean_color_arry_first[1][1]}", (300, 400),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(color_image,f"{mean_color_arry_first[1][2]}", (350, 400),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        
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
            
            showHandGesture(recognizer,color_image)
            
            send_data(client_socket, recognized_gesture)
            # send_data(client_socket,str(x_center))
            # send_data(client_socket,str(z))
            combined_image = np.hstack((color_image, depth_image_color))
            cv2.imshow('Color and depth combined image', combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 소켓 종료 및 자원 해제
        client_socket.close()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main_program()