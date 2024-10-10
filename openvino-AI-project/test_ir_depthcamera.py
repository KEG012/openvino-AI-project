import cv2
import pyrealsense2 as rs
import numpy as np
import threading

class Depth_Camera():

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.align_to = None
        self.running = True

        context = rs.context()
        connect_device = None
        if context.devices[0].get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device = context.devices[0].get_info(rs.camera_info.serial_number)

        print(" > Serial number : {}".format(connect_device))
        self.config.enable_device(connect_device)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # FPS 조정
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # FPS 조정

        self.depth_frame = None
        self.color_frame = None

        # Start camera in a separate thread
        self.thread = threading.Thread(target=self.run_camera)
        self.thread.start()

    def run_camera(self):
        try:
            self.pipeline.start(self.config)
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

            while self.running:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                self.depth_frame = aligned_frames.get_depth_frame()
                self.color_frame = aligned_frames.get_color_frame()

        except Exception as e:
            print("Error in camera thread:", e)

    def __del__(self):
        self.running = False
        self.pipeline.stop()
        print("Collecting process is done.\n")

    def execute(self):
        print('Collecting depth information...')
        x, y = 400, 120

        while True:
            if self.color_frame is not None:
                color_image = np.asanyarray(self.color_frame.get_data())
                depth_info = self.depth_frame.as_depth_frame()
                print("Depth : ", round((depth_info.get_distance(x, y) * 100), 2), "cm")
                color_image = cv2.circle(color_image, (x, y), 2, (0, 0, 255), -1)
                cv2.imshow('RealSense', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print("┌──────────────────────────────────────┐")
        print('│ Collecting of depth info is stopped. │')
        print("└──────────────────────────────────────┘")

if __name__ == "__main__":
    depth_camera = Depth_Camera()
    depth_camera.execute()

cv2.release()
cv2.destroyAllWindows()