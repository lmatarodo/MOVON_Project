import sys
import os
import multiprocessing
import time
import traceback

# 1) 졸음 감지 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "face_process/drowsiness_detection_master"))
from face_process.drowsiness_detection_master.drowsiness_detector import run_drowsiness_detection

# 2) YOLO 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "lane_process/YOLOPv2"))
from lane_process.YOLOPv2.demo import make_parser, run_demo


def run_yolo(shared_data):
    """YOLO Pv2를 실행하는 프로세스 함수 (차선 감지)"""
    try:
        parser = make_parser()
        opt = parser.parse_args([
            "--source", "/home/dylan/MOVON_Project/src/lane_process/YOLOPv2/data/hipass_drive.mp4",
            "--weights", "/home/dylan/MOVON_Project/src/lane_process/YOLOPv2/data/weights/yolopv2.pt",
            "--img-size", "640",
            "--conf-thres", "0.3",
            "--iou-thres", "0.45",
        ])
        print("[YOLO PROCESS] YOLO started.")

        # YOLO 실행 (차선 이탈 감지)
        run_demo(opt, shared_data)

        print("[YOLO PROCESS] YOLO finished.")
    except Exception as e:
        print("🚨 YOLO 프로세스에서 오류 발생!")
        print(traceback.format_exc())

    shared_data['stop'] = True  # YOLO가 종료되면 전체 프로그램 종료


def main():
    try:
        manager = multiprocessing.Manager()
        shared_data = manager.dict({
            'is_drowsy': False,       # 졸음 감지 상태
            'lane_departure': False,  # 차선 이탈 상태
            'is_closed': False,       # 눈 감김 상태
            'stop': False,            # 프로그램 종료 플래그
            'ear_initialized': False, # 눈 감김 초기화 여부
        })

        # 1) 졸음 감지 프로세스 실행
        drowsiness_process = multiprocessing.Process(
            target=run_drowsiness_detection, args=(shared_data,)
        )
        drowsiness_process.start()
        print("[MAIN] Drowsiness detection process started.")

        # 눈 감김 측정이 완료될 때까지 대기
        while not shared_data["ear_initialized"]:
            print("⏳ EAR 측정 중... YOLO 실행 대기")
            time.sleep(1)

        print("✅ EAR 측정 완료. YOLO 실행 시작!")


        # 2) YOLO 차선 감지 프로세스 실행
        yolo_process = multiprocessing.Process(
            target=run_yolo, args=(shared_data,)
        )
        yolo_process.start()
        print("[MAIN] YOLO process started.")

        # 3) 메인 프로세스에서 상태 모니터링e
        while not shared_data['stop']:
            print(f"[STATUS] 눈 감김: {shared_data['is_closed']} "
              f"| 졸음 감지: {shared_data['is_drowsy']} "
              f"| 차선 이탈: {shared_data['lane_departure']}")
            time.sleep(1)

        print("[MAIN] Stopping all processes...")

        # 4) 프로세스 종료 대기
        drowsiness_process.join()
        yolo_process.join()
        print("[MAIN] All processes joined. Exiting.")

    except Exception as e:
        print("🚨 프로그램에서 오류 발생!")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()