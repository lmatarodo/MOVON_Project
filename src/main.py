# main.py

import sys
import os
import multiprocessing
import time

# 1) 졸음 감지 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "face_process/drowsiness_detection_master"))

from face_process.drowsiness_detection_master.drowsiness_detector import run_drowsiness_detection

# 2) YOLO 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "lane_process/YOLOPv2"))

from lane_process.YOLOPv2.demo import make_parser, run_demo, detect

def run_yolo():
    parser = make_parser()
    opt = parser.parse_args([
        "--source", "/home/dylan/MOVON_Project/src/lane_process/YOLOPv2/data/hipass_drive.mp4",
        "--weights", "/home/dylan/MOVON_Project/src/lane_process/YOLOPv2/data/weights/yolopv2.pt",
        "--img-size", "640",
        "--conf-thres", "0.3",
        "--iou-thres", "0.45",
    ])
    print("[MAIN] YOLO options:", opt)

    # YOLO 실행 (멀티프로세싱)
    run_demo(opt)


def main():
    try:
        # 공유 딕셔너리
        shared_data = multiprocessing.Manager().dict({
            'is_drowsy': False,
            'lane_departure': False,
            'stop': False,
        })

        # 1) 졸음 감지 프로세스 시작
        drowsiness_process = multiprocessing.Process(target=run_drowsiness_detection, args=(shared_data,))
        drowsiness_process.start()
        print("[MAIN] Drowsiness detection process started.")

        # 2) YOLO 프로세스 실행 (멀티프로세싱)
        yolo_process = multiprocessing.Process(target=run_yolo)
        yolo_process.start()

        # 3) 모든 프로세스 종료 대기
        drowsiness_process.join()
        yolo_process.join()
        print("[MAIN] All processes joined. Exiting.")

    except Exception as e:
        import traceback
        print("🚨 오류 발생! 프로그램이 중지되었습니다.")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
