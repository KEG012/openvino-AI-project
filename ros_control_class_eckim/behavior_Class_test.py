import behavior_Class
from behavior_Class import behavior
import time

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