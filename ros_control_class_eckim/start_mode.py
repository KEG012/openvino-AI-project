import behavior_Class
from behavior_Class import behavior

mainCon=behavior(args=None)


# num=1 : go_forward
# num=2 : go_back
# num=3 : turn_left
# num=4 : turn_right
# num=5 : stop
standard_distance=1

#center_X frame의 중앙 값
#human_X 인식된 사람의 센터 값 (인식되지 않았을 경우 None)
#boundary_X 기준으로 정해준 경계의 좌측 x좌표와 우측 x좌표의 tuple 형태, boundary[0] 좌측 boundary[1] 우측
#human_distance 인식된 사람과의 거리

def startMode(center_X=None, human_X=None, boundary_X=None, human_distance=None):
    if human_X==None:  #인식 X
        mainCon.bot_order_out(5)
    else:              #인식 O
        if human_X>boundary_X[0] & human_X<boundary_X[1]: #boundary 안 O
            if human_distance>standard_distance:    #사람 거리가 일정 거리 밖에 있다
                mainCon.bot_order_out(1)
            else:                                   #사람 거리가 일정 거리 안에 있다
                mainCon.bot_order_out(5)

        else:                                             #boundary 안 X
            if human_X<boundary_X[0]: #화면 기준 좌측에 사람 존재
                mainCon.bot_order_out(3)
            else:                     #화면 기준 우측에 사람 존재
                mainCon.bot_order_out(4)




        
