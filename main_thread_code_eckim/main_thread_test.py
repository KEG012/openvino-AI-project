#!/usr/bin/env python3

import os
import threading
import queue
from queue import Empty, Queue
from time import sleep
import keyboard
import cv2

FORCE_STOP = False


def thread1(q):
    while not FORCE_STOP:
        q.put((1))
        # sleep(1)
    return


def thread2(q):

    while not FORCE_STOP:
        q.put((2))
        # sleep(1)

    return

def main():
    global FORCE_STOP
    q=Queue()
    t1 = threading.Thread(target=thread1, args=(q,))
    t2 = threading.Thread(target=thread2, args=(q,))
    t1.start()
    t2.start()
    while not FORCE_STOP:
        if keyboard.is_pressed('esc'):  
            print("ESC 키 눌림")
            FORCE_STOP=True

        try:
            threadNum = q.get_nowait()
            q.task_done()
            print(threadNum)
            # sleep(1)
        except queue.Empty:
            continue

       

    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
            print(threadNum)
        except queue.Empty:
            continue

    t1.join()
    t2.join()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()