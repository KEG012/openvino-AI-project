import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socket
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import time

previous_timestamp_ms = 0
global gesture_start_time_victory
gesture_start_time_victory = None
global gesture_start_time_thumbup
gesture_start_time_thumbup = None
global gesture_start_time_i_love_you
gesture_start_time_i_love_you = None
global gesture_start_time_closed_fist
gesture_start_time_closed_fist = None
GESTURE_HOLD_TIME = 5
GESTURE_HOLD_TIME_SHORT=3
global previous_depth_data
previous_depth_data = None     
#'192.168.0.28'
#'10.10.16.163'
# TCP 클라이언트 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.30', 5000))  # 서버의 IP 주소

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
    if depth_data  == b"":
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

model_path = 'gesture_recognizer.task'

#Hand land mark 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = 0.5)
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
    global previous_timestamp_ms
    # BGR 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # MediaPipe의 Image 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    frame_timestamp_ms = int(time.time() * 1000)
    
    if frame_timestamp_ms <= previous_timestamp_ms:
        frame_timestamp_ms = previous_timestamp_ms + 1 #frame_timestanp_ms가 더 커야 하는데 더 작을 경우에 대한 예외 처리
    
    previous_timestamp_ms = frame_timestamp_ms

    #UI에 인식한 손의 landmark 표시
    results = hands.process(rgb_image)
    draw_hand_landmarks(color_image,results)
    
    recognizer.recognize_async(mp_image, frame_timestamp_ms)
    cv2.putText(color_image, recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11n-pose.pt").to(device)
depth_scale= 0

global Back_start_time
Back_start_time=0 
global back_flag
back_flag=0 
global bot_mode_flag # 0: standbymode 1:automode
global move_flag
global mean_front_rgb , mean_back_rgb
# global mean_front_hsv , mean_back_hsv

#sound_flag
global camera_in_flag
camera_in_flag=0
global front_scan_flag
front_scan_flag=0
global back_scan_flag
back_scan_flag=0
global please_one_flag
please_one_flag=0
global dudu_flag
dudu_flag=0
global termination_flag
termination_flag=0
global start_flag
start_flag=0
global pause_flag
pause_flag=0
def track_victory_gesture():
    global gesture_start_time_victory
    
    if recognized_gesture == "Victory":
        if gesture_start_time_victory is None:
            gesture_start_time_victory = time.time()
            print(gesture_start_time_victory)
        elif time.time() - gesture_start_time_victory >= GESTURE_HOLD_TIME:
            gesture_start_time_victory=time.time()
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
            gesture_start_time_i_love_you= time.time()
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
            gesture_start_time_closed_fist=time.time()
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
            gesture_start_time_thumbup=time.time()
            return True
    else:
        gesture_start_time_thumbup = None
    
    return False

def matching_degree_back(cropped_image):
    global mean_back_rgb, mean_back_hsv
    mean_back_rgb_now, mean_front_hsv_now = mean_color(cropped_image)
    matching_degree_back_=3
    matching_limit=20
    
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
    matching_limit=20

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

def calculate_box(color_img, depth_img, results):
    color_ROI = []
    y_center=0
    x_center=0
    z = 0
    z_list=[]
    box_info = []
    x_center_list = []
    y_center_list = []
    global move_flag
    global camera_in_flag
    if not results:
        return color_img, color_ROI, z_list, x_center_list
    

    try:
        boxes = results[0].boxes.cpu().numpy()
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
                
                color_ROI.append((x_min, y_min, x_max, y_max))
                
                cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max),
                            (0, 0, 255), 2)
                
                x_center = (x_max + x_min) // 2
                x_center_list.append((x_center))
                y_center = (y_max + y_min) // 2
                y_center_list.append((y_center))
                cv2.circle(color_img, (x_center, y_center), 3, (0, 0, 255), 2)
                
                if depth_scale != 0:
                    z = depth_img[y_center, x_center] * depth_scale
                    z_list.append((z))
                else:
                    z = depth_img[y_center, x_center] * 0.001
                    z_list.append((z))
                depth_label = f"depth : {z:.1f}"
                cv2.putText(color_img, depth_label, (x_center - 50, y_center - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

                x1, y1, x2, y2 = boxes[idx].xyxy[0].astype(int)
                box_info.append(((x1, y1, x2, y2)))

                label = f"person : {idx + 1}"
                
                cv2.rectangle(color_img, (x1,y1), (x2, y2), (0,255,0), 2, cv2.LINE_AA)
                
                cv2.putText(color_img, label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
    except IndexError:
        print("No boxs in here")
        move_flag = 0
        camera_in_flag=1
        #bell_ring
    return color_img, box_info, color_ROI, x_center_list, z_list

def stand_by_mode(color_image, depth_image,recognizer):
    global Back_start_time
    global back_flag
    global mean_front_rgb , mean_back_rgb
    # global mean_front_hsv , mean_back_hsv
    global bot_mode_flag
    global front_scan_flag
    global back_scan_flag
    global please_one_flag
    results = pose_model(source=color_image, conf=0.3, verbose=False)  
    color_image, box_info, color_ROI, x_center_list, z_list = calculate_box(color_image, depth_image, results) 

  

    if len(color_ROI)==1:
        x1, y1, x2, y2 = box_info[0]
        x_ROI_1,y_ROI_1,x_ROI_2,y_ROI_2=color_ROI[0]
        cropped_image=color_image[y1:y2,x1:x2]
        cropped_image_ROI=color_image[y_ROI_1:y_ROI_2,x_ROI_1:x_ROI_2]
        showHandGesture(recognizer, cropped_image)
        
        if track_victory_gesture() and back_flag==0:
            mean_front_rgb, mean_front_hsv = mean_color(cropped_image_ROI)
            Back_start_time=time.time() 
            back_flag=1
            # print(mean_front_hsv)
            print("뒤도세요")
            front_scan_flag=1
            #bell_ringing(앞 스캔 완료)
        

        if ((time.time()-Back_start_time)>GESTURE_HOLD_TIME) and back_flag==1:
            mean_back_rgb, mean_back_hsv= mean_color(cropped_image_ROI)
            # print(mean_back_rgb)
            # print(mean_back_hsv)
            print("successed")
            back_flag=0
            bot_mode_flag=1
            back_scan_flag=1
            #bell_ringing(뒤 스캔 완료, 주행을 시작합니다.)
            
    elif len(color_ROI)>1:
        #bell_ring(한명만 나와주세용)
        please_one_flag=1
        print("gg")
        
            
def auto_mode(color_image, depth_image, recognizer):
    global mean_front_rgb, mean_front_hsv
    global mean_back_rgb, mean_back_hsv
    global bot_mode_flag
    success_flag_back=0
    success_flag_front=0# 0: standby 1: front 2: back
    global move_flag
    global move_pause_flag
    global dudu_flag
    global termination_flag
    global start_flag
    global pause_flag
    results = pose_model(source=color_image, conf=0.3, verbose=False) 
    color_image, box_info, color_ROI, x_center_list, z_list = calculate_box(color_image, depth_image, results) 
    idx_back=None
    idx_front=None
    for idx_back, ROI_back in enumerate(color_ROI):
        cropped_image_back=color_image[ROI_back[1]:ROI_back[3],ROI_back[0]:ROI_back[2]]
        matching_degree_back_num = matching_degree_back(cropped_image_back)
        
        if matching_degree_back_num==6:
            success_flag_back=1
            ROI_back_human=box_info[idx_back]
            print("등 일치")
            #bell_ring_dudu
            # dudu_flag=1
            cropped_image_back_human=color_image[ROI_back_human[1]:ROI_back_human[3],ROI_back_human[0]:ROI_back_human[2]]
            break
        else: 
            move_flag = 0
            break
            
    for idx_front, ROI_front in enumerate(color_ROI):
        cropped_image_front=color_image[ROI_front[1]:ROI_front[3],ROI_front[0]:ROI_front[2]]
        matching_degree_front_num = matching_degree_front(cropped_image_front)
        
        if matching_degree_front_num==6:
            success_flag_front=1
            ROI_front_human=box_info[idx_front]
            print("앞 일치")
            cropped_image_front_human=color_image[ROI_front_human[1]:ROI_front_human[3],ROI_front_human[0]:ROI_front_human[2]]
            break
        else : 
            move_flag = 0
            break
            
    if success_flag_back==1 and move_pause_flag==0:
        move_flag = 1
        # start()
        print("주행중")

    if success_flag_front==1:
        showHandGesture(recognizer, cropped_image_front_human) 
        if track_i_love_you_gesture(): # 해지
            print("해지")
            #bell_ringing(해지 되었습니다.)
            termination_flag=1
            bot_mode_flag=0
            move_flag = 0
        if track_thump_up(): #주행 시작
            print("주행 시작")
            # move_flag=1
            #bell_ringing(주행을 시작합니다.)
            start_flag=1
            move_pause_flag=0
        if track_open_palm(): #정지
            print("정지")
            move_flag=0
            pause_flag=1
            move_pause_flag=1
            #bell_ringing(일시 정지)


    if idx_back is None or idx_front is None:
        return 300,1.4

    return x_center_list[idx_back], z_list[idx_back]

def main_program():
    global bot_mode_flag
    bot_mode_flag=0
    global move_flag
    move_flag = 0
    global move_pause_flag
    move_pause_flag=0
    recognizer = handGestueInit()
    x_center, z = 0, 0

    global sound_num
    sound_num=0
    global camera_in_flag
    global front_scan_flag
    global back_scan_flag
    global please_one_flag
    global dudu_flag
    global termination_flag
    global start_flag
    global pause_flag

    try:  # 종료 조건을 루프 내부로 이동
        while True:
            
            color_image, depth_image, depth_colormap = get_frame_from_data()
            
            
            if bot_mode_flag==0:
                stand_by_mode(color_image, depth_image, recognizer)
                
            # person_dtection_flag=1

            if bot_mode_flag==1:
                x_center , z = auto_mode(color_image, depth_image, recognizer)
                
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

            if camera_in_flag==1:
                sound_num=1
            elif front_scan_flag==1:
                sound_num=2
            elif back_scan_flag==1:
                sound_num=3
            elif please_one_flag==1:
                sound_num=4
            elif dudu_flag==1:
                sound_num=5
            elif termination_flag==1:
                sound_num=6
            elif start_flag==1:
                sound_num=7
            elif pause_flag==1:
                sound_num=8
            send_data(client_socket,str(sound_num)) 

            #print(move_flag)
            send_data(client_socket, str(move_flag))
            
            send_data(client_socket,str(x_center))
            
            send_data(client_socket,str(z))


            sound_num=0
            camera_in_flag=0
            front_scan_flag=0
            back_scan_flag=0
            please_one_flag=0
            dudu_flag=0
            termination_flag=0
            start_flag=0
            pause_flag=0

            

            
            combined_image = np.hstack((color_image, depth_colormap))
            cv2.imshow('Color and depth combined image', combined_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 소켓 종료 및 자원 해제
        client_socket.close()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main_program()