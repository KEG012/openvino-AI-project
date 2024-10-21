from ultralytics import YOLO
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import Hand
from Hand import Hand_Instrument
############################################################   Model & gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = YOLO("yolo11s-pose.pt").to(device)
Hand_Instrument_object=Hand_Instrument(args=None)
############################################################   Model & gpu


############################################################   realsense-turtlebot
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

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


def frame_return():
    color_image=None
    depth_image=None
    depth_colormap=None
    frame_arrived_flag=False
    aligned_frames = get_frames()
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        return color_image, depth_image, depth_colormap, frame_arrived_flag
    frame_arrived_flag=True
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = cv2.flip(aligned_depth_image, 1)
    color_image = cv2.flip(color_image, 1)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return color_image, depth_image, depth_colormap, frame_arrived_flag
############################################################   realsense-turtlebot


############################################################   내가 준 roi의 평균값 반환
def mean_color(cropped_image):
    
    mean_color_arry = [0,0]
    cropped_HSV_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    if mean_color_arry[0] == 0 and mean_color_arry[1] == 0:
        mean_color_arry[0] = tuple(map(int, cv2.mean(cropped_image)[:3]))
        mean_color_arry[1] = tuple(map(int, cv2.mean(cropped_HSV_image)[:3]))

    return mean_color_arry
############################################################   내가 준 roi의 평균값 반환

############################################################   탐지된 모든 사람의 phisical data 반환
def run_extract_pose(color_img, depth_img, results):
    color_ROI = []
    ycenter=0
    xcenter=0
    z=[]
    box_info = []
    x_center = []
    y_center = []
    if not results:
        return color_img, color_ROI, z, x_center
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
                
                # 어깨와 골반의 최소 최대 좌표를 통하여 사각형 그림
                cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max),
                            (0, 0, 255), 2)
                
                xcenter = (x_max + x_min) // 2
                x_center.append((xcenter))
                ycenter = (y_max + y_min) // 2
                y_center.append((ycenter))
                cv2.circle(color_img, (xcenter, ycenter), 3, (0, 0, 255), 2)
                
                if depth_scale != 0:
                    z1 = depth_img[ycenter, xcenter] * depth_scale
                    z.append((z1))
                else:
                    z1 = depth_img[ycenter, xcenter] * 0.001
                    z.append((z1))
                depth_label = f"depth : {z1:.1f}"
                cv2.putText(color_img, depth_label, (xcenter - 50, ycenter - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

                x1, y1, x2, y2 = boxes[idx].xyxy[0].astype(int)
                box_info.append(((x1, y1, x2, y2)))

                label = f"person : {idx + 1}"
                
                cv2.rectangle(color_img, (x1,y1), (x2, y2), (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(color_img, label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
    except IndexError:
        print("No boxs in here")

    return color_img, box_info, color_ROI, x_center, z

############################################################   탐지된 모든 사람의 phisical data 반환


############################################################   가장 가까이서 gesture를 취한 사람의 index 반환


############################################################   가장 가까이서 gesture를 취한 사람의 index 반환
global back_flag #전역 변수는 local에서도 선언해 줘야 로컬에서 전역 변수로의 접근이 가능하다.
back_flag=0
global Back_start_time
Back_start_time=0
global mean_front_rgb, mean_front_hsv
global mean_back_rgb, mean_back_hsv
global person_detection_flag

def matching_degree_back(cropped_image):
    global mean_back_rgb, mean_back_hsv
    mean_back_rgb_now, mean_back_hsv_now=mean_color(cropped_image)
    matching_degree_back_=3
    matching_limit=10

    if(mean_back_rgb[0]-matching_limit<mean_back_rgb_now[0]<mean_back_rgb[0]+matching_limit):
        matching_degree_back_+=1
    if(mean_back_rgb[1]-matching_limit<mean_back_rgb_now[1]<mean_back_rgb[1]+matching_limit):
        matching_degree_back_+=1
    if(mean_back_rgb[2]-matching_limit<mean_back_rgb_now[2]<mean_back_rgb[2]+matching_limit):
        matching_degree_back_+=1
    # if(mean_back_hsv[0]-matching_limit<mean_back_hsv_now[0]<mean_back_hsv[0]+matching_limit):s
    #     matching_degree_back_+=1
    # if(mean_back_hsv[1]-matching_limit<mean_back_hsv_now[1]<mean_back_hsv[1]+matching_limit):
    #     matching_degree_back_+=1
    # if(mean_back_hsv[2]-matching_limit<mean_back_hsv_now[2]<mean_back_hsv[2]+matching_limit):
    #     matching_degree_back_+=1
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
    # if(mean_front_hsv[0]-matching_limit<mean_front_hsv_now[0]<mean_front_hsv[0]+matching_limit):
    #     matching_degree_front_+=1
    # if(mean_front_hsv[1]-matching_limit<mean_front_hsv_now[1]<mean_front_hsv[1]+matching_limit):
    #     matching_degree_front_+=1
    # if(mean_front_hsv[2]-matching_limit<mean_front_hsv_now[2]<mean_front_hsv[2]+matching_limit):
    #     matching_degree_front_+=1

    print(matching_degree_front_)
    return matching_degree_front_





def person_detection_func(color_image, depth_image,recognizer):
    global Back_start_time
    global back_flag
    global mean_front_rgb, mean_front_hsv
    global mean_back_rgb, mean_back_hsv
    global person_detection_flag
    results = pose_model(source=color_image, conf=0.3, verbose=False)  #Pose_model은 골반과 hip 좌표를 얻기 위하여, 이 때 모든 사람의 results가 나옴
    color_image, box_info, color_ROI, x_center, z = run_extract_pose(color_image, depth_image, results) #run_extract_pose 함수는 사람 내부 ROI를 따고, 좌표 값 추출


    if len(color_ROI)==1:
        x1, y1, x2, y2 = box_info[0]
        x_ROI_1,y_ROI_1,x_ROI_2,y_ROI_2=color_ROI[0]
        cropped_image=color_image[y1:y2,x1:x2]
        cropped_image_ROI=color_image[y_ROI_1:y_ROI_2,x_ROI_1:x_ROI_2]
        Hand_Instrument_object.showHandGesture(cropped_image) #showHandGesture 함수는 손에 랜드마크를 그려주고, hand gesture의 결과를 반환함, 여기서 gesture 시작
        if Hand_Instrument_object.track_victory_gesture() and back_flag==0:
            mean_front_rgb, mean_front_hsv = mean_color(cropped_image_ROI)
            Back_start_time=time.time() #Back_start_time은 뒤를 돌아서 앞으로 보는 데까지 시간을 주기 위해서
            back_flag=1
            print("뒤도세요")
            #bell_ringing
            

        if ((time.time()-Back_start_time)>Hand_Instrument_object.GESTURE_HOLD_TIME) and back_flag==1:
            mean_back_rgb, mean_back_hsv= mean_color(cropped_image_ROI)
            print("successed")
            back_flag=0
            person_detection_flag=1
            #bell_ringing
    

global move_flag
move_flag=1        
def automode(color_image, depth_image, recognizer):
    global mean_front_rgb, mean_front_hsv
    global mean_back_rgb, mean_back_hsv
    global person_detection_flag
    success_flag_back=0
    success_flag_front=0
    global move_flag
    results = pose_model(source=color_image, conf=0.3, verbose=False)  #Pose_model은 골반과 hip 좌표를 얻기 위하여, 이 때 모든 사람의 results가 나옴
    color_image, box_info, color_ROI, x_center, z = run_extract_pose(color_image, depth_image, results) #run_extract_pose 함수는 사람 내부 ROI를 따고, 좌표 값 추출
    for idx_back, ROI_back in enumerate(color_ROI):
        cropped_image_back=color_image[ROI_back[1]:ROI_back[3],ROI_back[0]:ROI_back[2]]
        matching_degree_back_num = matching_degree_back(cropped_image_back)
        if matching_degree_back_num==6:
            success_flag_back=1
            ROI_back_human=box_info[idx_back]
            print("등 일치")
            cropped_image_back_human=color_image[ROI_back_human[1]:ROI_back_human[3],ROI_back_human[0]:ROI_back_human[2]]
            break
    #위의 for 문을 통해 1. 원하는 사람을 찾았다는 flag On 2. 그 사람의 cropped된 image 3. 그 사람의 index와 cropped 된 image 추출

    for idx_front, ROI_front in enumerate(color_ROI):
        cropped_image_front=color_image[ROI_front[1]:ROI_front[3],ROI_front[0]:ROI_front[2]]
        matching_degree_front_num = matching_degree_front(cropped_image_front)
        if matching_degree_front_num==6:
            success_flag_front=1
            ROI_front_human=box_info[idx_front]
            print("앞 일치")
            cropped_image_front_human=color_image[ROI_front_human[1]:ROI_front_human[3],ROI_front_human[0]:ROI_front_human[2]]
            break
    #위의 for 문을 통해 1. 원하는 사람을 찾았다는 flag On 2. 그 사람의 cropped된 image 3. 그 사람의 index와 cropped 된 image 추출
    if success_flag_back==1 and move_flag==1:
        #자율주행
        # start()
        print("주행중")
    
    elif success_flag_back==1 and move_flag==0:
        print("정지")


    if success_flag_front==1:
        Hand_Instrument_object.showHandGesture(cropped_image_front_human) #showHandGesture 함수는 손에 랜드마크를 그려주고, hand gesture의 결과를 반환함, 여기서 gesture 시작
        if Hand_Instrument_object.track_i_love_you_gesture(): # 해지
            print("해지")
            #bell_ringing
            person_detection_flag=0
        if Hand_Instrument_object.track_thump_up(): #주행 시작
            # start()
            print("주행 시작")
            move_flag=1
            #bell_ringing

        if Hand_Instrument_object.track_closed_fist(): #정지
            # stop()
            print("정지")
            move_flag=0
            #bell_ringing




def main(args=None):

    global person_detection_flag
    person_detection_flag=0
    while True:
        color_image, depth_image, depth_colormap, frame_arrived_flag=frame_return()
        if frame_arrived_flag==False:
            continue

        if person_detection_flag==0:
            person_detection_func(color_image, depth_image, Hand_Instrument_object.recognizer)
        if person_detection_flag==1:
            automode(color_image, depth_image, Hand_Instrument_object.recognizer) 


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
        cv2.imshow('Color and depth combined image', combined_image)

        # 'ESC'를 누르면 종료
        if cv2.waitKey(10) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()

    
