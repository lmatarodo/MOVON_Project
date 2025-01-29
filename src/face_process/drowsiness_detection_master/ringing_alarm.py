
# coding: utf-8

# In[ ]:

import os
import pygame

def select_alarm(result):
    if result == 0:
        sound_alarm("power_alarm.wav")
    elif result == 1:
        sound_alarm("nomal_alarm.wav")
    else:
        sound_alarm("short_alarm.mp3")

def sound_alarm(filename):
    pygame.mixer.init()

    # 현재 파일(`ringing_alarm.py`)이 위치한 폴더 기준으로 사운드 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)

    # 파일이 존재하는지 확인 후 실행
    if not os.path.exists(path):
        print(f"⚠️ 경고: 사운드 파일을 찾을 수 없습니다: {path}")
        return  # 파일이 없으면 함수 종료
    
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

