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

    # print(frame_audio.shape)

    fn = frame_audio.shape[1]

    # 常数设置

    maxsilence = 20  # 语音段允许的最大静音长度
    minlen = 30  # 语音段的最短长度

    status = 0

    # 计算短时过零率
    zcr = STZcr(frame_audio)
    ste = STEn(frame_audio)

    # 调整能量门限
    amp2 = 15*np.mean(ste[:10])
    amp1 = 30*np.mean(ste[:10])
    zcr2 = 10*np.mean(zcr[:10])
    # amp2 = 80000
    # amp1 = 150000
    # zcr2 = 455

    # print(amp2)
    # print(amp1)
    # print(zcr2)

    # plt.plot(zcr)
    # plt.show()

    # plt.plot(ste)
    # plt.show()

    # print(len(zcr))
    # print(len(ste))

    xn = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(len(zcr)):
        goto = 0
        if status == 0 or status == 1:
            if ste[n] > amp1:
                x1[xn] = np.max((n-count[xn]-1, 1))
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif ste[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        elif status == 2:
            if ste[n] > amp2 or zcr[n] > zcr2:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]

        elif status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    # count = count-silence/2
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
    voiceseg = findSegment(np.where(SF == 1)[0])

    return voiceseg


def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if len(express):

        if express[0] == 0:
            voiceIndex = np.where(express)
        else:
            voiceIndex = express
        d_voice = np.where(np.diff(voiceIndex) > 1)[0]
        voiceseg = {}
        if len(d_voice) > 0:
            for i in range(len(d_voice) + 1):
                seg = {}
                if i == 0:
                    st = voiceIndex[0]
                    en = voiceIndex[d_voice[i]]
                elif i == len(d_voice):
                    st = voiceIndex[d_voice[i - 1] + 1]
                    en = voiceIndex[-1]
                else:
                    st = voiceIndex[d_voice[i - 1] + 1]
                    en = voiceIndex[d_voice[i]]
                seg['start'] = st
                seg['end'] = en
                seg['duration'] = en - st + 1
                voiceseg[i] = seg
    else:
        voiceseg = 0
    print("voiceseg: ", voiceseg)
    return voiceseg


if __name__ == '__main__':
    time_start = time.time()
    fs, audio = wavfile.read("real_data/channel-0.wav")
    # 归一化
    audio = audio / np.max(np.abs(audio))
    # audio = audio[:7*fs]
    frameSize_time = 0.032
    step_time = 0.016
    frameSize = round(frameSize_time*fs)
    step = round(step_time*fs)

    frame_audio = utils.enframe(audio, frameSize, step).T
    voiceseg = vad(fs, frame_audio)
    print(voiceseg)

    if voiceseg:
        plt.plot(audio)
        for i in range(len(voiceseg.keys())):
            print(voiceseg[i]['start'])
            print(voiceseg[i]['end'])
            plt.plot(int(voiceseg[i]['start']*step), 1, '.k')
            plt.plot(int(voiceseg[i]['end']*step), 1, 'or')
            # plt.savefig('images/TwoThr.png')
            # plt.close()

        plt.show()
    else:
        print("cannot find any segment")
