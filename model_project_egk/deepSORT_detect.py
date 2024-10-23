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
import socket

################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11s-pose.pt").to(device)

tracker = DeepSort(max_age=45,
                   n_init=5,
                   max_iou_distance = 0.7,
                   nn_budget=150,
                   nms_max_overlap=1.0,
                   max_cosine_distance=0.2,
                   embedder="mobilenet",
                   embedder_gpu = True,
                   half = False)

################################################################################

target_id = None
bot_mode_flag = 0
move_flag = 0
searching_flag = 0
stop_flag = 0
state_text = "PREPARE"
target_text = "None Target"

################################################################################

previous_depth_data = None     
#'192.168.0.20'
#'10.10.16.163'
# TCP 클라이언트 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.22', 5000))  # 서버의 IP 주소

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

def get_frame_from_data():
    global previous_depth_data
    
    color_data = receive_data(client_socket)        
    depth_data = receive_data(client_socket)
    
    # depth 프레임을 일시적으로 놓쳤을 경우 저장해둔 전의 프레임을 대신 보내줌
    if depth_data is None or len(depth_data) == 0:
        if previous_depth_data is not None:
            depth_data = previous_depth_data
        else:
            print('Error occur Null depth frame arrived...')
    previous_depth_data =depth_data
    
    color_image = cv2.imdecode(np.frombuffer(color_data, np.uint8), cv2.IMREAD_COLOR)
    depth_image = cv2.imdecode(np.frombuffer(depth_data, np.uint16), cv2.IMREAD_UNCHANGED)
    
    flipped_color_image = cv2.flip(color_image, 1)
    flipped_depth_image = cv2.flip(depth_image, 1)
    flipped_depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(flipped_depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    return flipped_color_image, flipped_depth_image, flipped_depth_colormap 
        
def send_data(sock, data):
    """서버로 데이터를 전송하는 함수"""
    data_bytes = data.encode()
    data_size = len(data_bytes)
    sock.sendall(data_size.to_bytes(4, byteorder='big'))  # 데이터 크기 전송
    sock.sendall(data_bytes)  # 데이터 전송

###############################################################################

model_path = './gesture_recognizer.task'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

previous_timestamp_ms = 0
global gesture_start_time_victory
gesture_start_time_victory = None
global gesture_start_time_thumbup
gesture_start_time_thumbup = None
global gesture_start_time_i_love_you
gesture_start_time_i_love_you = None
global gesture_start_time_closed_fist
gesture_start_time_closed_fist = None
GESTURE_HOLD_TIME = 1
GESTURE_HOLD_TIME_SHORT=2

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

def showHandGesture(recognizer,color_img):   
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
    draw_hand_landmarks(color_img,hand_results)
    
    recognizer.recognize_async(mp_image, frame_timestamp_ms)
    cv2.putText(color_img, recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def track_victory_gesture():
    global gesture_start_time_victory
    
    if recognized_gesture == "Victory":
        if gesture_start_time_victory is None:
            gesture_start_time_victory = time.time()
            print(gesture_start_time_victory)
        elif time.time() - gesture_start_time_victory >= GESTURE_HOLD_TIME:
            return True
    else:
        gesture_start_time_victory = None
    
    return False

def track_i_love_you_gesture():
    global gesture_start_time_i_love_you
    
    if recognized_gesture == "ILoveYou":
        if gesture_start_time_i_love_you is None:
            gesture_start_time_i_love_you = time.time()
            print(gesture_start_time_i_love_you)
        elif time.time() - gesture_start_time_i_love_you >= GESTURE_HOLD_TIME:
            return True
    else:
        gesture_start_time_i_love_you = None
    
    return False

def track_open_palm():
    global gesture_start_time_closed_fist
    
    if recognized_gesture == "Open_Palm":
        if gesture_start_time_closed_fist is None:
            gesture_start_time_closed_fist = time.time()
            print(gesture_start_time_closed_fist)
        elif time.time() - gesture_start_time_closed_fist >= GESTURE_HOLD_TIME_SHORT:
            return True
    else:
        gesture_start_time_closed_fist = None
    
    return False

def track_thump_up():
    global gesture_start_time_thumbup
    
    if recognized_gesture == "Thumb_Up":
        if gesture_start_time_thumbup is None:
            gesture_start_time_thumbup = time.time()
            print(gesture_start_time_thumbup)
        elif time.time() - gesture_start_time_thumbup >= GESTURE_HOLD_TIME_SHORT:
            return True
    else:
        gesture_start_time_thumbup = None
    
    return False

####################################################################################

def matching_degree_back(cropped_image):
    global mean_back_rgb, mean_back_hsv
    mean_back_rgb_now, mean_front_hsv_now = mean_color(cropped_image)
    matching_degree_back_=3
    matching_limit=10
    
    if(mean_back_rgb[0]-matching_limit<mean_back_rgb_now[0]<mean_back_rgb[0]+matching_limit):
        matching_degree_back_+=1
    if(mean_back_rgb[1]-matching_limit<mean_back_rgb_now[1]<mean_back_rgb[1]+matching_limit):
        matching_degree_back_+=1
    if(mean_back_rgb[2]-matching_limit<mean_back_rgb_now[2]<mean_back_rgb[2]+matching_limit):
        matching_degree_back_+=1
    
    return matching_degree_back_

def matching_degree_front(cropped_image):
    global mean_front_rgb, mean_front_hsv
    mean_front_rgb_now, mean_front_hsv_now=mean_color(cropped_image)
    matching_degree_front_=3
    matching_limit=10

    if(mean_front_rgb[0]-matching_limit<mean_front_rgb_now[0]<mean_front_rgb[0]+matching_limit):
        matching_degree_front_+=1
    if(mean_front_rgb[1]-matching_limit<mean_front_rgb_now[1]<mean_front_rgb[1]+matching_limit):
        matching_degree_front_+=1
    if(mean_front_rgb[2]-matching_limit<mean_front_rgb_now[2]<mean_front_rgb[2]+matching_limit):
        matching_degree_front_+=1

    print(matching_degree_front_)
    return matching_degree_front_

def mean_color(cropped_image):
    
    mean_color_arry = [0,0]
    cropped_HSV_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    if mean_color_arry[0] == 0 and mean_color_arry[1] == 0:
        mean_color_arry[0] = tuple(map(int, cv2.mean(cropped_image)[:3]))
        mean_color_arry[1] = tuple(map(int, cv2.mean(cropped_HSV_image)[:3]))

    return mean_color_arry

####################################################################################

def calculate_box(color_img, depth_img, results):
    global move_flag
    upper_body_ROI = []
    box_info = []
    
    if not results:
        return color_img, box_info, upper_body_ROI

    try:
        # boxes = results[0].boxes.data
        boxes = results[0].boxes.data
        keypoints = results[0].keypoints.cpu().numpy()
        for idx, keypoint in enumerate(keypoints):
            left_shoulder = keypoint.xy[0][5][:2]
            right_shoulder = keypoint.xy[0][6][:2]
            left_hip = keypoint.xy[0][11][:2]
            right_hip = keypoint.xy[0][12][:2]
            
            if all(np.any(pt) for pt in [left_shoulder, right_shoulder, left_hip, right_hip]):
                valid_points = [left_shoulder, right_shoulder, left_hip, right_hip]

                x_min, y_min = np.min(valid_points, axis=0).astype(int)
                x_max, y_max = np.max(valid_points, axis=0).astype(int)
                
                upper_body_ROI.append((x_min, y_min, x_max, y_max))

                for box in boxes:
                    x1, y1, x2, y2, score, class_id = box.tolist()
                    if int(class_id) == 0:
                        if score > 0.5:
                            width = x2 - x1
                            height = y2 - y1
                            box_info.append(([x1, y1, width, height], score, int(class_id)))

    except IndexError:
        print("No boxs in here")
        move_flag = 0

    return color_img, box_info, upper_body_ROI

def deepsort_tracker(box_info, color_image):
    
    info = []
    
    tracks = tracker.update_tracks(box_info, frame = color_image)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        
        id_text = f"Person {track_id}"
        
        info.append((track_id, (x1, y1, x2, y2)))
                
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(color_image, id_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    return color_image, info

def stand_by_mode(color_image, human_box, recognizer):
    
    global target_id, bot_mode_flag, state_text, target_text
    
    if len(human_box) == 1:
        track_id, box = human_box[0]
        x1, y1, x2, y2 = box
        cropped_image = color_image[y1:y2, x1:x2]
        state_text = "STAND_BY"
        
        if cropped_image is not None and cropped_image.size > 0:
            showHandGesture(recognizer, cropped_image)
        else:
            showHandGesture(recognizer, color_image)
        
        if track_victory_gesture():
            target_id = track_id
            bot_mode_flag = 1
            state_text = "MOVE"
            target_text = f"Target ID : {target_id}"
            print(f"Tracking person ID : {target_id}")


def auto_mode(color_image, depth_image, recognizer, human_box, upper_body_ROI):
    
    global target_id
    global bot_mode_flag
    global move_flag
    global searching_flag
    global stop_flag
    global state_text
    global target_text
    z, box_x_center = 0, 0
    cropped_human_image = None
    searching_flag=0
    
    for idx, box in human_box:
        x1, y1, x2, y2 = box
        
        if target_id == idx:
            searching_flag = 1
            cropped_human_image = color_image[y1:y2, x1:x2]
            
            box_x_center = (x1 + x2) // 2
            box_y_center = (y1 + y2) // 2
            
            # if box_x_center >= 640 or box_y_center >= 480:
            #     box_x_center = 319
            #     box_y_center = 239
            
            z = depth_image[box_y_center, box_x_center] * 0.001
            depth_label = f"distance : {z:.2f} m"
            
            cv2.circle(color_image, (box_x_center, box_y_center), 3, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(color_image, depth_label, (box_x_center - 50, box_y_center - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)
            
        break

    if searching_flag == 1 and stop_flag == 0:
        move_flag = 1
        print("serching", move_flag)
        
    elif searching_flag == 0:
        move_flag = 0
        print("can't search", move_flag)
        target_text = "None Target"
        bot_mode_flag = 0
    
    if cropped_human_image is not None and cropped_human_image.size > 0:
        showHandGesture(recognizer, cropped_human_image)
    else:
        showHandGesture(recognizer, color_image)
        
    if track_i_love_you_gesture():
        print("reset")
        bot_mode_flag = 0
        move_flag = 0
        target_id = None
        state_text = "RESET"
        target_text = "None Target"
        print("target Id reset")
        
    if track_open_palm():
        print("stop")
        move_flag = 0
        stop_flag = 1
        state_text = "STOP"
        
    if track_thump_up():
        print("move")
        move_flag = 0
        stop_flag = 0
        state_text = "MOVE"

    return z, box_x_center

############################################################################################

def main():
    global bot_mode_flag
    global move_flag
    global state_text
    global target_text
    recognizer = handGestueInit()
    x_center, z = 0, 0
    
    cv2.namedWindow('Color and depth combined image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Color and depth combined image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        while True:
            
            color_image, depth_image, depth_colormap = get_frame_from_data()
            
            results = pose_model(source = color_image, conf=0.4, verbose = False)      
            color_image, box_info, upper_body_ROI = calculate_box(color_image, depth_image, results)
            
            color_image, human_box = deepsort_tracker(box_info, color_image)
            
            if bot_mode_flag == 0:
                stand_by_mode(color_image, human_box, recognizer)
                
            if bot_mode_flag == 1:
                z, x_center = auto_mode(color_image, depth_image, recognizer, human_box, upper_body_ROI)
                    
            f_height, f_width = color_image.shape[:2]

            cv2.putText(color_image, target_text, (5, 475), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(color_image, state_text, (450, 475), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            
            # center line using center position to set the position in center ratio
            f_width_center = f_width // 2
            # f_width_ratio = f_width0 // 5
            f_width_ratio = 100
            f_center_min = f_width_center - f_width_ratio
            f_center_max = f_width_center + f_width_ratio
            
            cv2.line(color_image, (f_center_min, 0),
                    (f_center_min, f_height), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(color_image, (f_center_max, 0),
                    (f_center_max, f_height), (255, 255, 255), 1, cv2.LINE_AA)
            
            send_data(client_socket, str(move_flag))
            send_data(client_socket, str(x_center))
            send_data(client_socket, str(z))
            
            combined_image = np.hstack((color_image, depth_colormap))
            c_height, c_width = combined_image.shape[:2]
            resized_combine_image = cv2.resize(combined_image, (c_width*2, c_height*2),
                                               interpolation=cv2.INTER_LINEAR)

            if 'out' not in locals():  # 비디오 저장 객체가 아직 초기화되지 않았을 때만 실행
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
                out = cv2.VideoWriter('output.mp4', fourcc, 15 , (c_width*2, c_height*2))  # 해상도를 c_width*2, c_height*2로 설정

            out.write(resized_combine_image)  # 프레임 저장
            
            cv2.imshow('Color and depth combined image', resized_combine_image)

            # 'ESC'를 누르면 종료
            if cv2.waitKey(10) & 0xFF == 27:
                break
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        out.release()
        
if __name__ == "__main__":
    main()
            
            