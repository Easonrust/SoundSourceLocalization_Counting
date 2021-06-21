import math
import numpy as np


def enframe(wavData, frameSize, step):
    coeff = 0.97  # 预加重系数
    wlen = len(wavData)
    # step = frameSize - overlap
    frameNum: int = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum))

    hamwin = np.hamming(frameSize)

    for i in range(frameNum):
        singleFrame = wavData[np.arange(
            i * step, min(i * step+frameSize, wlen))]
        singleFrame = np.append(
            singleFrame[0], singleFrame[:-1] - coeff*singleFrame[1:])  # 预加重
        frameData[:len(singleFrame), i] = singleFrame
        frameData[:, i] = hamwin * frameData[:, i]  # 加窗
    # print(frameData.shape)
    return frameData


def Filt(x, l, t, fs):
    l = 2.0*l/fs
    t = 2.0*t/fs
    b, a = signal.butter(6, [l, t], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, x)  # data为要过滤的信号
    return filtedData
