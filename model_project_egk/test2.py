import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# RealSense 스트림 시작
pipeline.start(config)

# Align 설정 (Color와 Depth 정렬)
align = rs.align(rs.stream.color)
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

# 2. YOLO 모델과 DeepSORT 초기화
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 모델 로드 (사람 감지 용도)
tracker = DeepSort(max_age=120,
                   n_init=3,
                   max_iou_distance = 0.6,
                   nn_budget=100,
                   nms_max_overlap=0.7,
                   max_cosine_distance=0.18,
                   embedder="clip_RN50x16",
                   embedder_gpu = True,
                   half = False)   # 객체 추적을 위한 DeepSORT 초기화

def get_depth_at_point(depth_image, x, y):
    """특정 좌표 (x, y)에서의 깊이 값을 반환."""
    width ,height = depth_image.shape[:2]
    
    if x < 0 or x >= width or y < 0 or y >= height:
        depth_value = 0
    else:    
        depth_value = depth_image[x, y] * depth_scale
    
    return depth_value

# 3. 메인 루프 (RealSense로 영상 처리 및 사람 추적)
try:
    while True:
        # 프레임 수신 및 정렬
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        # 유효한 프레임 확인
        if not color_frame or not depth_frame:
            continue

        # Color 이미지와 Depth 이미지 가져오기
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        flip_color_image = cv2.flip(color_image, 1)
        flip_depth_image = cv2.flip(depth_image, 1)
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(flip_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 4. YOLO로 사람 감지 수행
        results = yolo_model(flip_color_image)
        detections = []

        # YOLO 감지 결과 처리
        for result in results[0].boxes.data:
            x1, y1, x2, y2, score, class_id = result.tolist()
            if int(class_id) == 0:  # class_id가 0이면 '사람'
                if score > 0.50:
                    width = x2 - x1
                    height = y2 - y1
                    detections.append(([x1, y1, width, height], score, int(class_id)))

        # 5. DeepSORT로 객체 추적
        tracks = tracker.update_tracks(detections, frame=flip_color_image)

        # 추적된 객체 표시
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_tlbr())

            # 사람의 중앙 좌표 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 6. 해당 좌표의 Depth 값 추출 및 거리 계산
            distance = get_depth_at_point(depth_image, center_x, center_y)

            # 추적된 객체의 정보와 거리 표시
            cv2.rectangle(flip_color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                flip_color_image, f"ID: {track_id} Dist: {distance:.2f}m",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        combined_image = np.hstack((flip_color_image, depth_colormap))
        
        # 결과 영상 표시
        cv2.imshow("RealSense YOLO + DeepSORT", combined_image)

        # 'q'를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 파이프라인 종료 및 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()
