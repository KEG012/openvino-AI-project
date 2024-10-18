import pyrealsense2 as rs
import numpy as np
import socket
import pygame
import cv2
import behavior_Class_test
from behavior_Class_test import behavior
import time
# RealSense 카메라 설정


def main(args=None):
    
    try:
        print("hi")
        print(np.__version__)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        pipeline.start(config)

        # TCP 서버 소켓 설정

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', 4000))
        server_socket.listen(1)
        conn, addr = server_socket.accept()

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            
            # 컬러 이미지를 numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 깊이 이미지를 컬러맵으로 변환
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 이미지를 JPEG로 압축
            _, encoded_color = cv2.imencode('.jpg', color_image)
            _, encoded_depth = cv2.imencode('.jpg', depth_colormap)

            # 이미지 크기 전송 (4바이트)
            conn.sendall(len(encoded_color).to_bytes(4, byteorder='big'))
            conn.sendall(encoded_color)
            
            conn.sendall(len(encoded_depth).to_bytes(4, byteorder='big'))
            conn.sendall(encoded_depth)
    except KeyboardInterrupt:
        mainCon.orderFifth()



    # finally:
    #     # pipeline.stop()
    #     # conn.close()
    #     # server_socket.close()
    #     mainCon.terminate()

if __name__ == '__main__':
    main()