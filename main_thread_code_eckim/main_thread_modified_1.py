#!/usr/bin/env python3

import os
import threading
import queue
from queue import Empty, Queue
from time import sleep


FORCE_STOP = False


def thread1(q):
    while not FORCE_STOP:
    exit()


def thread2(q):

    while not FORCE_STOP:
    exit()

def main():
    global FORCE_STOP
    q=Queue()
    t1 = threading.Thread(target=thread1, args=(q,))
    t2 = threading.Thread(target=thread2, args=(q,))
    t1.start()
    t2.start()
    while not FORCE_STOP:
        
    try:
        threadNum = q.get(timeout=2)  # 2초 동안만 대기
    except queue.Empty:
        print("큐가 비어 있습니다!")


        q.task_done()

    t1.join()
    t2.join()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()