# lane_process/YOLOPv2/demo.py

import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import serial

try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # 시리얼 포트 안정화 대기
    print("✅ Arduino 연결 성공!")
except serial.SerialException as e:
    print(f"⚠️ Arduino 연결 실패: {e}")
    arduino = None  # 오류 발생 시, arduino 변수를 None으로 설정

# 필요한 함수들 임포트 (상대 경로)
from .utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, LoadImages
)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/tunnel_drive.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

def run_demo(opt, shared_data):
    with torch.no_grad():
        detect(opt, shared_data)

def detect(opt, shared_data):
    # -------------------------------------------------
    # 1) 기본 설정 및 디렉토리
    # -------------------------------------------------
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    lane_departure_start = None
    lane_departure_duration = 0.5  # 차선 이탈로 판단할 지속 시간 (초)

    # -------------------------------------------------
    # 2) 모델 로드
    # -------------------------------------------------
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device).eval()
    if half:
        model.half()  # to FP16

    # -------------------------------------------------
    # 3) 데이터 로더 준비
    # -------------------------------------------------
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # GPU Warmup (if CUDA)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    # -------------------------------------------------
    # 4) 탑뷰(Perspective Transform) 좌표 정의 
    # -------------------------------------------------
    #  (사용 환경에 맞게 변경 필요)
    src_points = np.float32([
        [500, 600],   # 좌측 상단
        [700, 600],   # 우측 상단
        [900, 700],  # 우측 하단
        [300, 700],   # 좌측 하단
    ])
    dst_width, dst_height = 600, 700
    dst_points = np.float32([
        [0, 0],
        [dst_width, 0],
        [dst_width, dst_height],
        [0, dst_height],
    ])
    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # -------------------------------------------------
    # 5) 메인 추론 루프
    # -------------------------------------------------
    for path, img, im0s, vid_cap in dataset:
        # 5.1) 전처리
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # [0,255] → [0.0,1.0]

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # -------------------------------------------------
        # 5.2) 모델 추론
        # -------------------------------------------------
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        # -------------------------------------------------
        # 5.3) 세그멘테이션 결과 (주행가능영역 / 차선)
        # -------------------------------------------------
        da_seg_mask = driving_area_mask(seg)  # (H,W), 0/1
        ll_seg_mask = lane_line_mask(ll)      # (H,W), 0/1

        # -------------------------------------------------
        # 5.4) 검출된 객체들 후처리
        # -------------------------------------------------
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            s += '%gx%g ' % img.shape[2:]

            if len(det):
                # scale coords
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_img:
                        plot_one_box(xyxy, im0, line_thickness=3)

            # 추론 시간 출력
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            #------------------------------------------------
            # 6) 원본에 세그멘테이션 표시
            #------------------------------------------------
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            #------------------------------------------------
            # 7) 차선 마스크를 별도 윈도우에 (0/255) 시각화
            #------------------------------------------------
            # ll_seg_mask는 0/1 텐서이므로 0/255로 변환
            if isinstance(ll_seg_mask, torch.Tensor):
                ll_mask_np = ll_seg_mask[0].cpu().numpy()  # batch 차원 주의. 필요 시 [0] 사용
            else:
                ll_mask_np = ll_seg_mask
            # 0/1 → 0/255
            lane_mask_vis = (ll_mask_np * 255).astype(np.uint8)
            #cv2.imshow('lane_mask_only', lane_mask_vis) # 차선 마스크만
            top_view_img = cv2.warpPerspective(im0, M, (dst_width, dst_height))

 
            lane_3ch = np.dstack([lane_mask_vis, lane_mask_vis, lane_mask_vis])
            lane_top_view = cv2.warpPerspective(lane_3ch, M, (dst_width, dst_height))

            # 차량 중심과 임계값 범위
            car_x = dst_width // 2  # 차량 중심 (탑뷰 이미지의 너비 중앙)
            threshold_x = 100  # 가로 방향 임계값
            threshold_y_top = 500  # 세로 방향 상단 제한
            threshold_y_bottom = 700  # 세로 방향 하단 제한

            # 중심선과 임계값 범위 표시
            cv2.line(lane_top_view, (car_x, 0), (car_x, dst_height), (255, 0, 0), 2)  # 파란색 중심선
            cv2.rectangle(lane_top_view,
                        (car_x - threshold_x, threshold_y_top),  # 왼쪽 상단
                        (car_x + threshold_x, threshold_y_bottom),  # 오른쪽 하단
                        (0, 255, 255), 2)  # 노란색 테두리 (임계값 범위)

            # 관심 영역 내 차선 마스크 확인
            mask_roi = lane_top_view[threshold_y_top:threshold_y_bottom, car_x - threshold_x:car_x + threshold_x]  # 제한된 ROI
            white_pixels = np.count_nonzero(mask_roi)  # 흰색 픽셀(차선) 개수

            # 디버깅용 출력
            #print("ROI Pixel Values:", np.unique(mask_roi))  # 고유 픽셀 값 확인
            #print("Number of White Pixels in ROI:", white_pixels)

            prev_lane_status = shared_data['lane_departure']
            lane_departure_detected = False
            # 이탈 여부 판단
            if white_pixels > 2973:  # 임계값 범위 내에 흰색 픽셀이 있다면
                if lane_departure_start is None:
                    lane_departure_start = time.time()
                
                elapsed_time = time.time() - lane_departure_start
                if elapsed_time >= lane_departure_duration:
                    lane_departure_detected = True
                    if prev_lane_status != lane_departure_detected:
                        arduino.write(b'L')  # 아두이노에 'L' 전송 → LED 켜짐
                        print(f"🚨 차선 벗어남")
                        print("🔔 아두이노 신호 전송 (LED 켜짐)")
                    #print("⚠️ 차량이 차선 중앙에서 이탈했습니다!")
                    cv2.putText(lane_top_view, "WARNING: Lane Departure", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                lane_departure_start = None
                if prev_lane_status != lane_departure_detected:
                    print(f"🚨 차선 돌아옴")
                    print("🔔 아두이노 신호 전송 (LED 꺼짐)")
                    arduino.write(b'l') # 아두이노에 'l' 전송 → LED 꺼짐
                #print("✅ 차량이 차선 중앙에 있습니다.")
                cv2.putText(lane_top_view, "Lane Centered", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #if prev_lane_status != lane_departure_detected:
                    #print("✅ [YOLO] 차선 이탈 상태 변경: {lane_departure_detected}")
            shared_data['lane_departure'] = lane_departure_detected



            # 디버깅: ROI와 탑뷰 출력
            # cv2.imshow('ROI', mask_roi)
            cv2.imshow('lane_top_view_with_visuals', lane_top_view)  # 차선 마스크 탑뷰

            cv2.imshow('lane_view', im0)           # 원본
            #cv2.imshow('top_view', top_view_img)  # 원본 탑뷰

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("작업을 종료합니다.")
                break  # q 키를 누르면 중단

            #------------------------------------------------
            # 10) 결과 저장
            #------------------------------------------------
            # if save_img:
            #     p = Path(p)  # Path 객체
            #     save_path = str(save_dir / p.name)
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w, h = im0.shape[1], im0.shape[0]
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path,
            #                                          cv2.VideoWriter_fourcc(*'mp4v'),
            #                                          fps, (w, h))
            #         vid_writer.write(im0)

        # 바깥 for문용, 'q'로 완전 종료
        if key == ord('q'):
            break

    if arduino:
        arduino.close()    

    # -------------------------------------------------
    # 11) 시간 출력
    # -------------------------------------------------
    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

    # 모든 창 닫기
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = make_parser().parse_args()
    #print(opt)
    run_demo(opt)
    
