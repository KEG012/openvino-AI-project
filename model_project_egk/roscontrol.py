import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import subprocess
import time
import gc
import os
import signal


class TurtleBot3Controller(Node):
    def __init__(self, args=None):
        super().__init__('turtlebot3_controller')
        # cmd_vel 토픽 퍼블리셔 설정
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # Twist 메시지 생성
        self.twist = Twist()
        print("Waiting for bringup to complete...")
        

    def go_forward(self):
        # 모터 속도 설정 (전진)
        self.twist.linear.x = 0.2
        self.twist.angular.z = 0.0
        self.get_logger().info('go_forward')
        self.publisher_.publish(self.twist)

    def go_back(self):
        # 모터 속도 설정 (후진)
        self.twist.linear.x = -0.2
        self.twist.angular.z = 0.0
        self.get_logger().info('go_back')
        self.publisher_.publish(self.twist)

    def turn_left(self):
        # 좌회전 (각속도 설정)
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.4  # 양의 z 값은 시계 반대 방향 회전
        self.get_logger().info('turn_left')
        self.publisher_.publish(self.twist)

    def turn_right(self):
        # 우회전 (각속도 설정)
        self.twist.linear.x = 0.0
        self.twist.angular.z = -0.4  # 음의 z 값은 시계 방향 회전
        self.get_logger().info('turn_right')
        self.publisher_.publish(self.twist)

    def stop(self):
        # 정지
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.get_logger().info('Stopped.')
        self.publisher_.publish(self.twist)


class Turmain():
    def __init__(self, args):
        # ROS 2 노드 초기화
        rclpy.init(args=args)
        
        # 런치 파일 서브 프로세스 실행
        self.bringup_process = subprocess.Popen(['ros2', 'launch', 'turtlebot3_bringup', 'robot.launch.py'])
        time.sleep(10)  # 런치 파일이 완전히 실행되기까지 10초 대기 (필요시 시간 조정)
        self.controller = TurtleBot3Controller(args=args)

    def terminate(self):
        # ROS 노드 종료
        if self.controller:
            self.controller.destroy_node()
            self.controller = None  # 객체를 None으로 설정하여 해제
            gc.collect()  # 가비지 컬렉션 실행

        rclpy.shutdown()

        # 런치 프로세스 종료
        if self.bringup_process:
            os.kill(self.bringup_process.pid, signal.SIGINT)  # 서브 프로세스에 SIGINT (CTRL+C) 보내기
            time.sleep(10)
            self.bringup_process.terminate()  # 종료 시도
            self.bringup_process.wait()  # 프로세스가 완전히 종료될 때까지 대기
        
        

def main(args=None):
    # TurtleBot3 제어 노드 생성
    mainCon=Turmain(args)

    try:
        # TurtleBot3을 전진, 후진, 좌회전, 우회전 명령 테스트
        mainCon.controller.go_forward()
        time.sleep(1)
        mainCon.controller.go_back()
        time.sleep(1)
        mainCon.controller.turn_left()
        time.sleep(1)
        mainCon.controller.turn_right()
        time.sleep(1)
        mainCon.controller.stop()

    except KeyboardInterrupt:
        mainCon.controller.get_logger().info('Keyboard interrupt, stopping the robot.')
        mainCon.controller.stop()

    # 노드 종료
    mainCon.terminate()
if __name__ == '__main__':
    main()
