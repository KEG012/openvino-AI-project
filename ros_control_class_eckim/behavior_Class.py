import rosControl
from rosControl import Turmain
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

    def terminate(self):
        super().terminate()

    def sound_order_1(self):
        playsound(self.sound_file1)

    def sound_order_2(self):
        playsound(self.sound_file2)

    def sound_order_3(self):
        playsound(self.sound_file3)

    def bot_order_out(self,num):
        if num==1:
            self.orderOne() #go_forward
        elif num==2:
            self.orderTwo() #go_back
        elif num==3:
            self.orderThree() #turn_left
        elif num==4:
            self.orderFour() #turn_right
        elif num==5:
            self.orderFifth() #stop

    def sound_order_out(self,num):
        if num==1:
            self.sound_order_1()
        elif num==2:
            self.sound_order_2()
        elif num==3:
            self.sound_order_3()


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

