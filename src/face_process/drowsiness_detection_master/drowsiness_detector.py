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

# (아두이노와 시리얼 통신을 위한 pyserial 임포트, 필요시 주석 해제)
# import serial

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
sys.path.append(current_dir)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def run_drowsiness_detection(shared_data=None):
    """
    눈 감긴 상태(is_closed)를 실시간으로 업데이트하고,
    5초 이상 감긴 상태가 지속되면 졸음(is_drowsy=True)으로 판정한 뒤,
    아두이노로 신호(부저)를 전송하는 예시 코드

    아두이노 부분은 주석 처리. 실제 연결 시 주석 해제 후 포트 확인.
    """
    
    # --- (아두이노 시리얼 연결, 필요시 주석 해제) ---
    # arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    # time.sleep(2)  # 시리얼 포트 초기화 대기

    EAR_THRESH = 200.0  # 눈 감김 임계값 (예: 200)
    TIMER_FLAG = False  # 눈 감기 타이머 동작 중인지 여부
    start_closing = 0.0   # 눈 감기 시작 시점
    ARDUINO_SENT = False  # 5초 초과 시 아두이노 신호를 한 번만 보내기 위한 플래그

    # 공유 데이터에 기본값 설정
    if shared_data is not None:
        shared_data['is_closed'] = False  # 현재 눈 감긴 상태
        shared_data['is_drowsy'] = False  # 5초 이상 감긴 상태(졸음)

    print("loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("starting video stream for drowsiness detection...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame = imutils.resize(frame, width=400)
        _, gray = lr.light_removing(frame)
        
        rects = detector(gray, 0)
        
        # 기본값
        both_ear = 500.0  # 혹시 감지 안 될 경우 대비

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # 👇 EAR 값에 500을 곱함 (기존 로직)
            both_ear = (leftEAR + rightEAR) * 500

            # 디버깅: 눈 윤곽 표시
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        # 1) 눈 감긴 상태 판단
        if both_ear < EAR_THRESH:
            # 👇 "눈 감김" 상태
            if shared_data is not None:
                shared_data['is_closed'] = True

            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
                ARDUINO_SENT = False  # 다시 감기 시작하면 아두이노 전송 가능 상태로 초기화

            # 눈 감긴 후 경과 시간
            closing_time = timeit.default_timer() - start_closing
            if closing_time >= 5.0 and not ARDUINO_SENT:
                # 5초 이상 눈 감김 ⇒ 졸음 판정
                print(f"🚨 눈 감은 상태가 5초 이상 지속됨: {closing_time:.2f}s")
                print("🔔 아두이노 신호 전송 (부저)")

                # if arduino:
                #     arduino.write(b'1')  # 아두이노에 '1' 전송 → 부저 울림
                
                ARDUINO_SENT = True
                if shared_data is not None:
                    shared_data['is_drowsy'] = True

        else:
            # 👇 "눈 뜸" 상태
            if shared_data is not None:
                shared_data['is_closed'] = False
                shared_data['is_drowsy'] = False

            TIMER_FLAG = False
            ARDUINO_SENT = False

        # EAR 값 표시
        cv2.putText(frame, f"EAR: {both_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    vs.stop()

    # if arduino:
    #     arduino.close()
    print("Drowsiness detection stopped.")
