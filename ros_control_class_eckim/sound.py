import pygame
import time

# Pygame 초기화
pygame.mixer.init()

# OGG 파일 경로
sound_file1 = "/home/eckim/Sound/information3.wav"
sound_file2 = "/home/eckim/Sound/information3.wav"
sound_file3 = "/home/eckim/Sound/information3.wav"

# 사운드 로드 및 재생
pygame.mixer.music.load(sound_file)
pygame.mixer.music.play()
# 사운드가 재생되는 동안 대기
while pygame.mixer.music.get_busy():
    time.sleep(1)

print("Sound playback finished.")
