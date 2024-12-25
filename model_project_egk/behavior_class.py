import roscontrol
from roscontrol import Turmain
import time
from playsound import playsound
class behavior(Turmain):
    def __init__(self, args):
        super().__init__(args)  # 부모 클래스의 초기화 메서드에 args 전달
        self.sound_file1 = "information3.wav"
        self.sound_file2 = "outgoing.wav"
        self.sound_file3 = "question.wav"

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
        if z <= 1.3:
            print("Move Backward")
            self.controller.go_back()
    
        elif z >= 1.4:
            print("mover Foward")
            self.controller.go_forward()
        
        else :
            print("stop")
            self.controller.stop()
            
            
    def move_center_position(self, x_center,z, f_center_min = 220, f_center_max = 420):
    
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

    def sound_order_1(self):
        playsound(self.sound_file1)

    def sound_order_2(self):
        playsound(self.sound_file2)

    def sound_order_3(self):
        playsound(self.sound_file3)

        

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


