
import pyrealsense2 as rs
import numpy as np
import socket
import cv2
import behavior_class
from behavior_class import behavior 

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps = 15):
        # RealSense 카메라 설정
        #print(np.__version__)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
    def start_pipeline(self):    
        self.pipeline.start(self.config)
        #depth scale ratio
        #self.depth_sensor = self.pipeline.start(self.config).get_device().first_depth_sensor()
        #self.depth_scale = self.depth_sensor.get_depth_scale()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        return color_frame, depth_frame

    def close(self):
        self.pipeline.stop()

class SocketServer:
    # TCP 서버 소켓 설정
    def __init__(self, host = '0.0.0.0', port = 5000):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"Listening on {host}:{port}")
    
    def accept_connection(self):    
        self.conn, self.addr = self.server_socket.accept()
        print(f"Connection from {self.addr}")
    
    def send_frame_data(self,data):
        self.conn.sendall(len(data).to_bytes(4, byteorder='big'))
        self.conn.sendall(data)
    
    def send_data(self,data):
        self.conn.sendall(data)
    
    def receive_data(self):
        # 데이터 크기 수신 (4바이트)
        data_size = int.from_bytes(self.conn.recv(4), byteorder='big')
        
        # 데이터 수신
        data = b""
        while len(data) < data_size:
            packet = self.conn.recv(data_size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def close(self):
        self.conn.close()
        self.server_socket.close()

class byteToFloat:
    
    def __init__(self, byte_data):
        self.byte_data = byte_data
    
    def byte_to_float(self):
        self.str_data = self.byte_data.decode('utf-8')
        self.float_data = float(self.byte_data)
        
        return self.float_data
    
def main(args=None):
    try:
        camera = RealSenseCamera()
        camera.start_pipeline()
    
        socket_server = SocketServer()
        socket_server.accept_connection()
        
        mainCon = behavior(args)
        
        while True:
            color_frame, depth_frame = camera.get_frames()

            if not color_frame or not depth_frame:
                continue

            # 컬러 이미지를 numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 이미지를 JPEG로 압축
            _, encoded_color = cv2.imencode('.jpg', color_image)
            _, encoded_depth = cv2.imencode('.png', depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # 이미지 크기 전송 (4바이트)
            socket_server.send_frame_data(encoded_color)
            socket_server.send_frame_data(encoded_depth)
            
            #socket_server.send_data(str(camera.depth_scale).encode())
            sound_num = socket_server.receive_data()            
            move_flag = socket_server.receive_data()

            
            # x_center,z data 
            x_center_byte = socket_server.receive_data()
            z_byte = socket_server.receive_data()
            
            x_converter = byteToFloat(x_center_byte)
            z_converter = byteToFloat(z_byte)
            
            x_center = x_converter.byte_to_float()
            z = z_converter.byte_to_float()
            

            if sound_num==b'1':
                mainCon.sound_camera_in()
            elif sound_num==b'2':
                mainCon.sound_front_scan()
            elif sound_num==b'3':
                mainCon.sound_back_scan()
            elif sound_num==b'4':
                mainCon.sound_please_one()
            elif sound_num==b'5':
                mainCon.sound_dudu()
            elif sound_num==b'6':
                mainCon.sound_termination()
            elif sound_num==b'7':
                mainCon.sound_start()
            elif sound_num==b'8':
                mainCon.sound_pause()
            
            


            #이벤트처리#
            #chasing mode
            if move_flag == b'1':
                mainCon.move_center_position(x_center,z)
            #stop
            if move_flag == b'0':
                mainCon.orderFifth()        

    finally:
        camera.close()
        socket_server.close()
        # ROS open Node kill
        mainCon.terminate()
        
if __name__ == "__main__":
    main()