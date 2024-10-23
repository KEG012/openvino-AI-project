import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# 스트림 활성화 (Depth와 Color)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)

# 깊이 데이터를 RGB 프레임에 맞추기 위한 align 객체 생성
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 프레임 받기
        frames = pipeline.wait_for_frames()

        # 프레임 정렬
        aligned_frames = align.process(frames)

        # 정렬된 depth와 color 프레임 가져오기
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        infrared_frame = aligned_frames.get_infrared_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # numpy 배열로 변환
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 이미지 시각화
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        combined_image = np.hstack((color_image, depth_colormap))

        # 이미지 출력
        cv2.imshow('Aligned Color and Depth', combined_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 파이프라인 정리
    pipeline.stop()
    cv2.destroyAllWindows()
