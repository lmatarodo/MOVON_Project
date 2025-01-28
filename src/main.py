# main.py

import sys
import os
import argparse
from lane_process.YOLOPv2.demo import make_parser, run_demo

# 현재 main.py의 위치를 기준으로 상대 경로를 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
yolopv2_path = os.path.join(current_dir, 'lane_process', 'YOLOPv2')

# Python 경로에 YOLOPv2 디렉토리 추가
sys.path.append(yolopv2_path)

# 옵션 설정
parser = make_parser()
opt = parser.parse_args([
    "--source", "/home/dylan/MOVON_Project/src/lane_process/YOLOPv2/data/summer_drive.mp4",
    "--weights", "/home/dylan/MOVON_Project/src/lane_process/YOLOPv2/data/weights/yolopv2.pt",
    "--img-size", "640",
    "--conf-thres", "0.3",
    "--iou-thres", "0.45"
])

# demo.py 실행
run_demo(opt)
