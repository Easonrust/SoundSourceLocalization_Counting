import numpy as np
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt
import utils

# TODO 修正能量阈值


# 短时过零率
# def ZCR(frameData):
#     frameNum = frameData.shape[1]
#     frameSize = frameData.shape[0]
#     zcr = np.zeros((frameNum, 1))

#     for i in range(frameNum):
#         singleFrame = frameData[:, i]
#         temp = singleFrame[:frameSize-1]*singleFrame[1:frameSize]
#         temp = np.sign(temp)
#         zcr[i] = np.sum(temp < 0)
#     return zcr

# # Short Time Energy


# def STE(frameData):
#     frameNum = frameData.shape[1]

#     ener = np.zeros((frameNum, 1))

#     for i in range(frameNum):
#         singleframe = frameData[:, i]
#         ener[i] = sum(singleframe * singleframe)
#     return ener

def STEn(frame_audio):
    """
    计算短时能量函数
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = frame_audio
    s = np.multiply(X, X)
    return np.sum(s, axis=1)


def STZcr(frame_audio):
    """
    计算短时过零率
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = frame_audio

    X1 = X[:, :-1]
    X2 = X[:, 1:]
    s = np.multiply(X1, X2)
    sgn = np.where(s < 0, 1, 0)
    return np.sum(sgn, axis=1)


def vad(fs, frame_audio):
    # frame_audio = frame_audio.T

    print(frame_audio.shape)

    # 常数设置

    maxsilence = 10  # 语音段允许的最大静音长度
    minlen = 5  # 语音段的最短长度

    status = 0
    count = 0
    silence = 0

    # 计算短时过零率
    zcr = STZcr(frame_audio)
    ste = STEn(frame_audio)

    # 调整能量门限
    amp1 = np.max(ste)/10
    amp2 = np.max(ste)/12
    zcr2 = np.max(zcr)/3
    # amp2 = 80000
    # amp1 = 150000
    # zcr2 = 455

    print(amp2)
    print(amp1)
    print(zcr2)

    # plt.plot(zcr)
    # plt.show()

    # plt.plot(ste)
    # plt.show()

    print(len(zcr))
    print(len(ste))

    x1 = 0
    for n in range(len(zcr)):
        goto = 0
        if status == 0 or status == 1:
            if ste[n] > amp1:
                x1 = np.max((n-count-1, 1))
                status = 2
                silence = 0
                count = count + 1
            elif ste[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count = count+1
            else:
                status = 0
                count = 0
        elif status == 2:
            if ste[n] > amp2 or zcr[n] > zcr2:
                count = count+1
            else:
                silence = silence + 1
                if silence < maxsilence:
                    count = count + 1
                elif count < minlen:
                    status = 0
                    silence = 0
                    count = 0
                else:
                    status = 3

        elif status == 3:
            break
    # count = count-silence/2
    x2 = x1 + count

    result = [int(x1), int(x2)]

    return result


if __name__ == '__main__':
    time_start = time.time()
    fs, audio = wavfile.read("data/speech.wav")
    # 归一化
    audio = audio / np.max(np.abs(audio))
    audio = audio[:7*fs]
    frameSize_time = 0.032
    step_time = 0.016
    frameSize = round(frameSize_time*fs)
    step = round(step_time*fs)

    frame_audio = utils.enframe(audio, frameSize, step).T
    [x11, x12] = vad(fs, frame_audio)
    print(x11)
    print(x12)
    plt.plot(audio)
    plt.plot(int(x11*step), 1, '.k')
    plt.plot(int(x12*step), 1, 'or')
    plt.show()
    time_end = time.time()
    print('program time cost', time_end-time_start, 's')
