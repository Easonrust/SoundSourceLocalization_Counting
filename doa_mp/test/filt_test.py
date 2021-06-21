import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import find_peaks
import time
from gcc_phat import GCC_Phat
from gcc_phat import gcc_phat
from utils import enframe
from vad import vad
from scipy import signal


def Filt(x, l, t, fs):
    l = 2.0*l/fs
    t = 2.0*t/fs
    # print(l)
    # print(t)
    b, a = signal.butter(6, [l, t], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
    # b, a = signal.firwin(7, [l, t], pass_zero=False)
    filtedData = signal.filtfilt(b, a, x)  # data为要过滤的信号
    return filtedData


fs, audio = wavfile.read("data/my_recording.wav")

plt.plot(audio)
plt.show()

audio = Filt(audio, 300, 3400, fs)

plt.plot(audio)
plt.show()
