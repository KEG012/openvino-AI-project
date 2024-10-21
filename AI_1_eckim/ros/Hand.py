import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Hand_Instrument():
    def __init__(self, args):
        self.model_path = './gesture_recognizer.task'

        self.mp_hands = mp.solutions.hands
        self.hands= self.mp_hands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5)
        self.mp_drawing= mp.solutions.drawing_utils

        self.recognized_gesture = 'No Gesture'
        
        self.gesture_start_time_victory = None
        self.gesture_start_time_thumbup = None
        self.gesture_start_time_i_love_you = None
        self.gesture_start_time_closed_fist = None
        self.GESTURE_HOLD_TIME = 5
        self.GESTURE_HOLD_TIME_SHORT=3

        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.recognizer=self.handGestueInit()



    def handGestueInit(self):
        options = self.GestureRecognizerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.handGestureResult
        )
        return self.GestureRecognizer.create_from_options(options)

    def handGestureResult(self, result,output_image: mp.Image, timestamp_ms: int):
        if result.gestures and len(result.gestures) > 0:
            self.recognized_gesture = result.gestures[0][0].category_name
        else:
            self.recognized_gesture = 'No Gesture'

    def draw_hand_landmarks(self, img,results):
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)# 초록색 원
                self.mp_drawing.draw_landmarks(img,handLms,self.mp_hands.HAND_CONNECTIONS)

    def showHandGesture(self ,color_img):       
        # BGR 이미지를 RGB로 변환
        rgb_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # MediaPipe의 Image 형식으로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        frame_timestamp_ms = int(time.time() * 1000)
        #UI에 인식한 손의 landmark 표시
        hand_results = self.hands.process(rgb_image)
        self.draw_hand_landmarks(color_img,hand_results)

        self.recognizer.recognize_async(mp_image, frame_timestamp_ms)
        cv2.putText(color_img, self.recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    def track_victory_gesture(self):

        if self.recognized_gesture == "Victory":
            if self.gesture_start_time_victory is None:
                self.gesture_start_time_victory = time.time()
            elif time.time() - self.gesture_start_time_victory >= self.GESTURE_HOLD_TIME:
                return True
        else:
            self.gesture_start_time_victory = None
        
        return False

    def track_i_love_you_gesture(self):
        if self.recognized_gesture == "ILoveYou":
            if self.gesture_start_time_i_love_you is None:
                self.gesture_start_time_i_love_you = time.time()
            elif time.time() - self.gesture_start_time_i_love_you >= self.GESTURE_HOLD_TIME:
                return True
        else:
            self.gesture_start_time_i_love_you = None
        
        return False

    def track_closed_fist(self):
        if self.recognized_gesture == "Closed_Fist":
            if self.gesture_start_time_closed_fist is None:
                self.gesture_start_time_closed_fist = time.time()
            elif time.time() - self.gesture_start_time_closed_fist >= self.GESTURE_HOLD_TIME_SHORT:
                return True
        else:
            self.gesture_start_time_closed_fist = None
        
        return False

    def track_thump_up(self):
        if self.recognized_gesture == "Thumb_up":
            if self.gesture_start_time_thumbup is None:
                self.gesture_start_time_thumbup = time.time()
            elif time.time() - self.gesture_start_time_thumbup >= self.GESTURE_HOLD_TIME_SHORT:
                return True
        else:
            self.gesture_start_time_thumbup = None
        
        return False
    
def main(args=None):
    print("ㅎㅎ")
if __name__ == '__main__':
    main()