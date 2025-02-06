import os
import sys
import time
import timeit
import numpy as np
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import light_remover as lr
import ringing_alarm as alarm
from threading import Thread

import serial

try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # 시리얼 포트 안정화 대기
    print("✅ Arduino 연결 성공!")
except serial.SerialException as e:
    print(f"⚠️ Arduino 연결 실패: {e}")
    arduino = None  # 오류 발생 시, arduino 변수를 None으로 설정

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
sys.path.append(current_dir)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def init_message():
    print("init_message")
    #alarm.sound_alarm("init_sound.mp3")

def run_drowsiness_detection(shared_data=None):

    OPEN_EAR = 0
    EAR_THRESH = 0
    both_ear = 500.0  
    TIMER_FLAG = False
    start_closing = 0
    ARDUINO_SENT = False
    is_drowsy = False

    # 눈 감김 상태 초기화
    if shared_data is not None:
        shared_data['ear_initialized'] = False  
    
    print("loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    print("starting video stream for drowsiness detection...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # EAR 값 초기화 함수 정의
    def init_open_ear():
        nonlocal OPEN_EAR, both_ear
        time.sleep(2)
        print("🟢 Open EAR 측정 시작...")
        ear_list = []
        th_message1 = Thread(target=init_message)
        th_message1.daemon = True
        th_message1.start()

        for _ in range(7):
            frame = vs.read()
            if frame is None:
                continue

            frame = imutils.resize(frame, width=400)
            _, gray = lr.light_removing(frame)
            rects = detector(gray, 0)

            if len(rects) == 0:
                print("⚠️ 얼굴 인식 실패, 다시 측정...")
                continue  # 얼굴이 감지되지 않으면 측정하지 않음

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                both_ear = (leftEAR + rightEAR) * 500

            print(f"📏 EAR 측정 중 (Open): {both_ear:.2f}")
            ear_list.append(both_ear)
            time.sleep(1)

        OPEN_EAR = sum(ear_list) / len(ear_list)
        print(f"✅ 측정된 OPEN_EAR: {OPEN_EAR:.2f}")

    def init_close_ear():
        nonlocal OPEN_EAR, EAR_THRESH, both_ear
        time.sleep(2)
        th_open.join()
        print("🟠 Close EAR 측정 시작...")
        ear_list = []
        th_message2 = Thread(target=init_message)
        th_message2.daemon = True
        th_message2.start()

        for _ in range(7):
            frame = vs.read()
            if frame is None:
                continue

            frame = imutils.resize(frame, width=400)
            _, gray = lr.light_removing(frame)
            rects = detector(gray, 0)

            if len(rects) == 0:
                print("⚠️ 얼굴 인식 실패, 다시 측정...")
                continue  # 얼굴이 감지되지 않으면 측정하지 않음

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                both_ear = (leftEAR + rightEAR) * 500

            print(f"📏 EAR 측정 중 (Close): {both_ear:.2f}")
            ear_list.append(both_ear)
            time.sleep(1)

        CLOSE_EAR = sum(ear_list) / len(ear_list)
        EAR_THRESH = (OPEN_EAR - CLOSE_EAR) / 2 + CLOSE_EAR
        print(f"✅ 측정된 CLOSE_EAR: {CLOSE_EAR:.2f}")
        print(f"📉 최종 EAR_THRESH: {EAR_THRESH:.2f}")

    th_open = Thread(target=init_open_ear)
    th_open.daemon = True
    th_open.start()
    th_open.join()  

    th_close = Thread(target=init_close_ear)
    th_close.daemon = True
    th_close.start()
    th_close.join()  

    # 🎯 EAR 측정 완료 후 YOLO 실행 가능하도록 플래그 설정
    shared_data['ear_initialized'] = True
    print("✅ EAR 측정 완료. YOLO 실행 가능!")

    # 이후 졸음 감지 루프 실행
    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame = imutils.resize(frame, width=400)
        _, gray = lr.light_removing(frame)
        rects = detector(gray, 0)

        both_ear = 500.0  # 기본값 설정

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            both_ear = (leftEAR + rightEAR) * 500

        #print(f"📡 실시간 EAR: {both_ear:.2f}")

        # 눈 감김 상태 판단
        if both_ear < EAR_THRESH:
            if shared_data is not None:
                shared_data['is_closed'] = True

            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
                ARDUINO_SENT = False  

            # 눈 감긴 후 경과 시간
            closing_time = timeit.default_timer() - start_closing
            if closing_time >= 5.0 and not ARDUINO_SENT:
                print(f"🚨 눈 감은 상태가 5초 이상 지속됨: {closing_time:.2f}s")
                print("🔔 아두이노 신호 전송 (부저)")
                if arduino:
                    arduino.write(b'B')  
                
                ARDUINO_SENT = True
                is_drowsy = True
                if shared_data is not None:
                    shared_data['is_drowsy'] = True

        else:
            if shared_data is not None:
                shared_data['is_closed'] = False

            TIMER_FLAG = False
            ARDUINO_SENT = False

            if is_drowsy:
                print("✅ 졸음 해제! 부저 OFF")
                if arduino:
                    arduino.write(b'N')  
                is_drowsy = False
                shared_data['is_drowsy'] = False

        cv2.putText(frame, f"EAR: {both_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()
    print("Drowsiness detection stopped.")
