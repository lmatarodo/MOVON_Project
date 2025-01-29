# drowsiness_detection.py

import os
import sys
import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
sys.path.append(current_dir)
import make_train_data as mtd


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def init_message():
    print("init_message")
    alarm.sound_alarm("init_sound.mp3")

def run_drowsiness_detection(shared_data=None):
    """
    졸음 감지 메인 루프를 실행하는 함수.
    `shared_data`가 주어지면, 예: shared_data['is_drowsy'] 등으로
    외부에서 졸음 상태를 실시간으로 공유받을 수 있게 함.
    """
    
    # (1) 기존 코드에 있던 변수들
    OPEN_EAR = 0
    EAR_THRESH = 0
    EAR_CONSEC_FRAMES = 20
    COUNTER = 0
    closed_eyes_time = []
    TIMER_FLAG = False
    ALARM_FLAG = False
    ALARM_COUNT = 0
    RUNNING_TIME = 0
    PREV_TERM = 0
    
    print("loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("starting video stream for drowsiness detection...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # both_ear는 init thread에서 계속 읽어야 하므로 nonlocal 스코프로 둠
    both_ear = 0

    def init_open_ear():
        nonlocal OPEN_EAR, both_ear
        time.sleep(5)
        print("open init time sleep")
        ear_list = []
        th_message1 = Thread(target=init_message)
        th_message1.daemon = True
        th_message1.start()
        for _ in range(7):
            ear_list.append(both_ear)
            time.sleep(1)
        OPEN_EAR = sum(ear_list) / len(ear_list)
        print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")

    def init_close_ear():
        nonlocal OPEN_EAR, EAR_THRESH, both_ear
        time.sleep(2)
        th_open.join()
        time.sleep(5)
        print("close init time sleep")
        ear_list = []
        th_message2 = Thread(target=init_message)
        th_message2.daemon = True
        th_message2.start()
        time.sleep(1)
        for _ in range(7):
            ear_list.append(both_ear)
            time.sleep(1)
        CLOSE_EAR = sum(ear_list) / len(ear_list)
        EAR_THRESH = (OPEN_EAR - CLOSE_EAR) / 2 + CLOSE_EAR
        print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR)
        print("The last EAR_THRESH's value :", EAR_THRESH, "\n")

    # init thread 실행
    th_open = Thread(target=init_open_ear)
    th_open.daemon = True
    th_open.start()

    th_close = Thread(target=init_close_ear)
    th_close.daemon = True
    th_close.start()

    # (2) 메인 루프
    while True:
        frame = vs.read()
        if frame is None:
            continue
        
        frame = imutils.resize(frame, width=400)
        L, gray = lr.light_removing(frame)
        
        rects = detector(gray, 0)
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            both_ear = (leftEAR + rightEAR) * 500  # 범위확대를 위한 *500

            # 졸음 여부 판별
            if both_ear < EAR_THRESH:
                if not TIMER_FLAG:
                    start_closing = timeit.default_timer()
                    TIMER_FLAG = True
                COUNTER += 1

                if COUNTER >= EAR_CONSEC_FRAMES:
                    mid_closing = timeit.default_timer()
                    closing_time = round((mid_closing - start_closing), 3)
                    if closing_time >= RUNNING_TIME:
                        if RUNNING_TIME == 0:
                            CUR_TERM = timeit.default_timer()
                            OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM), 3)
                            PREV_TERM = CUR_TERM
                            RUNNING_TIME = 1.75
                        
                        RUNNING_TIME += 2
                        ALARM_FLAG = True
                        ALARM_COUNT += 1

                        print(f"{ALARM_COUNT}st ALARM")
                        print("Eyes open time before alarm:", OPENED_EYES_TIME)
                        print("closing time:", closing_time)
                        
                        # shared_data에 졸음 플래그 저장
                        if shared_data is not None:
                            shared_data['is_drowsy'] = True

                        # 알람 울리는 부분
                        result = mtd.run([OPENED_EYES_TIME, closing_time * 10])

                        t = Thread(target=alarm.select_alarm, args=(result,))
                        t.daemon = True
                        t.start()

            else:
                COUNTER = 0
                TIMER_FLAG = False
                RUNNING_TIME = 0

                if ALARM_FLAG:
                    end_closing = timeit.default_timer()
                    closed_eyes_time.append(round((end_closing - start_closing), 3))
                    print("The time eyes were being offed:", closed_eyes_time)

                ALARM_FLAG = False
                if shared_data is not None:
                    shared_data['is_drowsy'] = False

            cv2.putText(frame, f"EAR : {both_ear:.2f}", (300, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
        
        cv2.imshow("Drowsiness Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    vs.stop()
    print("Drowsiness detection stopped.")
