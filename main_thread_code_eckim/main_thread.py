#!/usr/bin/env python3

import os
import threading
import queue
from queue import Empty, Queue
from time import sleep
import cv2
# from openvino.inference_engine import IECore

from iotdemo import FactoryController

FORCE_STOP = False

text1="VIDEO:Cam1 live"
text2="VIDEO:Cam2 live"
def thread_cam1(q):
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        q.put((text1,frame))
    q.put(('DONE', None))
    exit()


def thread_cam2(q):

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

    q.put(('DONE', None))
    exit()

def main():
    global FORCE_STOP

    q=Queue()
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()
    while not FORCE_STOP:
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
        name,frame=q.get()
        if name == 'DONE':
            FORCE_STOP = True
        q.task_done()
    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()