import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import time
from gcc_phat import gcc_phat
from utils import enframe
from vad import vad
from scipy import signal
from scipy.stats import gaussian_kde
from itertools import permutations
from sklearn import cluster
from sklearn.neighbors import KernelDensity

import socket
import struct
import pickle
import json
from multiprocessing import Process, Queue, Pool


def gcc_in_range(frame_s1, frame_s2, fs, MAX_TDOA):
    # 每一帧计算DOA
    Tau = []
    for i in range(frame_s1.shape[0]):

        sig = frame_s1[i, :]
        ref = frame_s2[i, :]

        tau, _, peaks_num = gcc_phat(sig ,
                                     ref,
                                     fs=fs,
                                     max_tau=MAX_TDOA,)

        # a = tau/MAX_TDOA
        # if a > 1:
        #     a = 1
        # elif a < -1:
        #     a = -1

        # Tau = np.arcsin(a) * 180 / np.pi
        # Tau = int(round(Tau))
        Tau.append(tau)

    return Tau


def preprocess(x, fs, sQueue, frameQueue, nQueue, preprocessedDataQueue):
    frameSize_time = 0.025
    step_time = 0.01
    frameSize = round(frameSize_time*fs)
    step = round(step_time*fs)

    s1 = x[0, :]
    s2 = x[1, :]
    s3 = x[2, :]
    s4 = x[3, :]

    # nQueue.put([s1.shape[0], s2.shape[0], s3.shape[0], s4.shape[0]])

    # 归一化
    s1 = s1 / np.max(np.abs(s1))
    s2 = s2 / np.max(np.abs(s2))
    s3 = s3 / np.max(np.abs(s3))
    s4 = s4 / np.max(np.abs(s4))
    sQueue.put([s1, s2, s3, s4])

    # 分帧加窗
    frame_s1 = enframe(s1, frameSize, step).T
    frame_s2 = enframe(s2, frameSize, step).T
    frame_s3 = enframe(s3, frameSize, step).T
    frame_s4 = enframe(s4, frameSize, step).T
    # print(frame_s1.shape)   # 16,400

    # 语音端点检测
    # [x11, x12] = vad(fs, frame_s4)
    # frame_s1 = frame_s1[x11:x12]
    # frame_s2 = frame_s2[x11:x12]
    # frame_s3 = frame_s3[x11:x12]
    # frame_s4 = frame_s4[x11:x12]
    voice_seg = vad(fs, frame_s1)
    if voice_seg != 0:
        for i in range(len(voice_seg.keys())):
            frame_s1_ = frame_s1[voice_seg[i]['start']:voice_seg[i]['end']]
            frame_s2_ = frame_s2[voice_seg[i]['start']:voice_seg[i]['end']]
            frame_s3_ = frame_s3[voice_seg[i]['start']:voice_seg[i]['end']]
            frame_s4_ = frame_s4[voice_seg[i]['start']:voice_seg[i]['end']]
            frameQueue.put([frame_s1_, frame_s2_, frame_s3_, frame_s4_])
            preprocessedDataQueue.put([frame_s1_, frame_s2_, frame_s3_, frame_s4_])
    # frameQueue.put([frame_s1, frame_s2, frame_s3, frame_s4])
    # print(x11)
    # print(x12)

    # frame_num = x12 - x11

    # return [frame_s1, frame_s2, frame_s3, frame_s4], frame_num
    return [frame_s1, frame_s2, frame_s3, frame_s4]


def postprocess(Tau):
    # 中值滤波平滑
    Tau = signal.medfilt(Tau)

    # KDE(核密度估计)
    sample_range = np.nanmax(Tau) - np.nanmin(Tau)

    # 生成样点
    ind = np.linspace(
        np.nanmin(Tau) - 0.5 * sample_range,
        np.nanmax(Tau) + 0.5 * sample_range,
        1000,
    )
    X_plot = ind[:, np.newaxis]

    # 核密度估计
    Tau = Tau[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.000003).fit(
        Tau)

    # 生成概率密度曲线
    log_dens = kde.score_samples(X_plot)
    pdf = np.exp(log_dens)

    # 概率密度曲线的归一化
    pdf = (pdf-np.min(pdf))/(np.max(pdf)-np.min(pdf))

    # 峰值搜索，prominence 为突出程度限制
    peaks, properties = signal.find_peaks(
        pdf, prominence=0.1, height=0.99)
    source_num = len(peaks)
    tau_result = ind[peaks]

    return tau_result


def mp_preprocess(fs, rawDataQueue, preprocessedDataQueue, frameNumQueue, sQueue, frameQueue, nQueue):
    while True:
        while preprocessedDataQueue.qsize() > 2:
            pass
        data = rawDataQueue.get()
        for i in range(20):
            data = np.concatenate([data, rawDataQueue.get()], axis=1)
        # print(data.shape)
        preprocess(data, fs, sQueue, frameQueue, nQueue, preprocessedDataQueue)
        # preprocessedDataQueue.put(x)
        # frameNumQueue.put(frame_num)
        print("preprocessedDataQueue: ", preprocessedDataQueue.qsize())


def mp_postprocess(gccedDataQueue1, gccedDataQueue2, tauResultQueue1, tauResultQueue2):
    pool = Pool(processes=2)
    while True:
        Theta1 = gccedDataQueue1.get()
        Theta2 = gccedDataQueue2.get()
        # while (gccedDataQueue1.qsize() >= 1 and gccedDataQueue2.qsize() >= 1):
        #     Theta1 = gccedDataQueue1.get()
        #     Theta2 = gccedDataQueue2.get()

        a = pool.apply_async(
            postprocess, (Theta1, ))
        b = pool.apply_async(
            postprocess, (Theta2, ))

        a.wait()
        b.wait()

        result1 = a.get()
        result2 = b.get()

        tauResultQueue1.put(result1)
        tauResultQueue2.put(result2)
        # print("gccedDataQueue1: ", gccedDataQueue1.qsize())
        print("tauResultQueue1: ", tauResultQueue1.qsize())


def mp_gcc(mic_distance, fs, preprocessedDataQueue, gccedDataQueue1, gccedDataQueue2):
    # 常数设置
    SOUND_SPEED = 340.0
    MAX_TDOA = mic_distance / float(SOUND_SPEED)

    pool = Pool(processes=4)

    while True:

        x = preprocessedDataQueue.get()
        frame_s1 = x[0]
        frame_s2 = x[1]
        frame_s3 = x[2]
        frame_s4 = x[3]
        point = frame_s1.shape[0] // 2

        a = pool.apply_async(
            gcc_in_range, (frame_s1[0:point, :], frame_s2[0:point, :], fs, MAX_TDOA))
        b = pool.apply_async(
            gcc_in_range, (frame_s1[point:, :], frame_s2[point:, :], fs, MAX_TDOA))

        c = pool.apply_async(
            gcc_in_range, (frame_s3[0:point, :], frame_s4[0:point, :], fs, MAX_TDOA))
        d = pool.apply_async(
            gcc_in_range, (frame_s3[point:, :], frame_s4[point:, :], fs, MAX_TDOA))

        # a.wait()
        # b.wait()

        Theta1 = a.get() + b.get()
        gccedDataQueue1.put(Theta1)

        # c.wait()
        # d.wait()

        Theta2 = c.get() + d.get()
        gccedDataQueue2.put(Theta2)

        # print("gccedDataQueue1: ", gccedDataQueue1.qsize())


def calculate_center(fs, MAX_TDOA, tauResultQueue1, tauResultQueue2, frameNumQueue, frameQueue, nQueue, centerQueue):
    frameSize_time = 0.025
    frameSize = round(frameSize_time*fs)

    while True:

        tau_result1 = tauResultQueue1.get()
        tau_result2 = tauResultQueue2.get()
        # n = nQueue.get()
        frame = frameQueue.get()

        # while (tauResultQueue1.qsize() >= 1 and tauResultQueue2.qsize() >= 1):
        #     tau_result1 = tauResultQueue1.get()
        #     tau_result2 = tauResultQueue2.get()
        #     n = nQueue.get()
        #     frame = frameQueue.get()

        # frame_num = frameNumQueue.get()

        # 声源个数
        source_num = np.max((len(tau_result1), len(tau_result2)))

        # 输出声源的个数
        print("source_num: ", source_num)
        print("len(tau_result1): ", len(tau_result1))
        print("len(tau_result2): ", len(tau_result2))

        if len(tau_result1) != len(tau_result2):
            print("cannot do the Location")
        else:
            # n1 = n[0]
            # n2 = n[1]
            # n3 = n[2]
            # n4 = n[3]

            frame_s1 = frame[0]
            frame_s2 = frame[1]
            frame_s3 = frame[2]
            frame_s4 = frame[3]

            frame_num = frame_s1.shape[0]

            Location = []
            P_result = []

            start = time.time()

            # 为了节省计算时间，截取了最中间的几帧进行配对统计
            # 每一帧获得一个位置结果，最后对所有的位置结果做一个K-means聚类
            # for i in range(frame_num):
            for i in range(int(frame_num/2-1), int(frame_num/2)):
                # for i in range(frame_num):
                fft_s1 = np.fft.rfft(frame_s1[int(i), :], n=frameSize)
                fft_s2 = np.fft.rfft(frame_s2[int(i), :], n=frameSize)
                fft_s3 = np.fft.rfft(frame_s3[int(i), :], n=frameSize)
                fft_s4 = np.fft.rfft(frame_s4[int(i), :], n=frameSize)

                # 固定第一个麦克风阵列不动，第二个麦克风阵列做全排列，计算每种排列配对产生的RMS，取RMS最小的那一个
                p = list(permutations(range(len(tau_result2))))

                ISP1_array = []
                ISP2_array = []
                for i in range(source_num):
                    ISP1 = calculate_ISP(
                        tau_result1[i], fft_s1, fft_s2, fs, frameSize)
                    ISP2 = calculate_ISP(
                        tau_result2[i], fft_s3, fft_s4, fs, frameSize)
                    ISP1_array.append(ISP1)
                    ISP2_array.append(ISP2)

                RMS = []
                for i in range(len(p)):
                    rms = 0
                    for j in range(source_num):
                        ISP1 = ISP1_array[j]
                        ISP2 = ISP2_array[p[i][j]]
                        rms = rms + calculate_sd(ISP1, ISP2)
                    RMS.append(rms)

                # 每一帧配对之后，获得一组位置结果
                m_RMS = np.argmin(RMS)
                p_result = p[m_RMS]
                P_result.append(p_result)
                for i in range(len(p_result)):
                    theta1 = calculate_theta(tau_result1[i], MAX_TDOA)
                    theta2 = calculate_theta(
                        tau_result2[p_result[i]], MAX_TDOA)
                    x = 3.97*np.tan(theta1)/(np.tan(theta1)+np.tan(theta2))
                    y = 3.97/(np.tan(theta1)+np.tan(theta2))

                    Location.append((x, y))

            end = time.time()
            # print("time cost1: ", end-start, "s")

            start = time.time()

            k_means = cluster.KMeans(n_clusters=source_num)
            k_means.fit(Location)
            center = k_means.cluster_centers_

            end = time.time()
            # print("time cost2: ", end-start, "s")

            # 输出最后确定的位置坐标
            print("center: ", json.dumps([{"id": 0, "center": center.tolist()}]))
            centerQueue.put(center)


def calculate_ISP(tau, fft_s1, fft_s2, fs, frameSize):
    ISP = []
    for i in range(fft_s1.shape[0]):
        A = np.mat([1, np.exp(1j*2*np.pi*i*fs/frameSize*tau)]
                   ).reshape((-1, 1))

        X = np.mat([fft_s1[i], fft_s2[i]]).reshape((-1, 1))
        phi = np.dot(X, X.H)

        isp = np.dot(A.H, phi)
        isp = np.dot(isp, A)
        ISP.append(np.abs(isp[0, 0]))

    return ISP


def calculate_sd(ISP1, ISP2):
    RMS = 0
    for i in range(len(ISP1)):
        RMS = RMS + np.log((ISP1[i]+1e-10)/(ISP2[i]+1e-10)) * \
            np.log((ISP1[i]+1e-10)/(ISP2[i]+1e-10))

    RMS = RMS/len(ISP1)

    return RMS


def calculate_theta(tau, MAX_TDOA):
    a = tau/MAX_TDOA
    if a > 1:
        a = 1
    elif a < -1:
        a = -1

    theta = np.arcsin(a)
    return theta
