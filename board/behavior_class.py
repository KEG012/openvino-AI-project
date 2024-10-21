import ros_control
from ros_control import Turmain
import time
from playsound import playsound
class behavior(Turmain):
    def __init__(self, args):
        super().__init__(args)  # 부모 클래스의 초기화 메서드에 args 전달
        self.sound_file_camera_in = "please_in_camera.m4a"
        self.sound_file_front_scan = "front_scan_complete.m4a"
        self.sound_file_back_scan = "back_scan_complete.m4a"
        self.sound_file_please_one = "please_one.m4a"
        self.sound_file_dudu = "question.wav"
        self.sound_file_termination = "termination_complete.m4a"
        self.sound_file_start = "Start.m4a"
        self.sound_file_pause = "pause.m4a"

    
    def orderOne(self):
        self.controller.go_forward()
    def orderTwo(self):
        self.controller.go_back()
    def orderThree(self):
        self.controller.turn_left()
    def orderFour(self):
        self.controller.turn_right()
    def orderFifth(self):
        self.controller.stop()
    
    def set_distance(self,z):
        if z <= 1.35:
            print("Move Backward")
            self.controller.go_back()
    
        elif z >= 1.45:
            print("mover Foward")
            self.controller.go_forward()
        
        else :
            print("stop")
            self.controller.stop()
            
            
    def move_center_position(self, x_center,z, f_center_min = 256, f_center_max = 384):
    
        if x_center < f_center_min:
            print("Turn Right")
            self.controller.turn_right()
    
        elif x_center > f_center_max:
            print("Turn Left")
            self.controller.turn_left()
    
        else: 
            print("Stay Center")
            self.controller.stop()
            self.set_distance(z)
    
    def terminate(self):
        super().terminate()

    def sound_camera_in(self):
        playsound(self.sound_file_camera_in)
    def sound_front_scan(self):
        playsound(self.sound_file_front_scan)
    def sound_back_scan(self):
        playsound(self.sound_file_back_scan)
    def sound_please_one(self):
        playsound(self.sound_file_please_one)
    def sound_dudu(self):
        playsound(self.sound_file_dudu)
    def sound_termination(self):
        playsound(self.sound_file_termination)
    def sound_start(self):
        playsound(self.sound_file_start)
    def sound_pause(self):
        playsound(self.sound_file_pause)


def main(args=None):
    # TurtleBot3 제어 노드 생성
    mainCon=behavior(args)

    try:
        # TurtleBot3을 전진, 후진, 좌회전, 우회전 명령 테스트
        mainCon.orderOne()
        time.sleep(1)
        mainCon.orderFifth()
        time.sleep(1)
        mainCon.sound_order_1()
        mainCon.sound_order_2()
        mainCon.sound_order_3()

    except KeyboardInterrupt:
        mainCon.orderFifth()

    # 노드 종료
    mainCon.terminate()


if __name__ == '__main__':
    main()


