import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import time
from gcc_phat import gcc_phat
from utils import enframe
from vad import vad
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

import socket
import struct
import pickle
from multiprocessing import Process, Queue, Pool


def gcc_in_range(frame_s1, frame_s2, fs, MAX_TDOA):
    # 每一帧计算TDOA
    Tau = []
    for i in range(frame_s1.shape[0]):

        sig = frame_s1[i, :]
        ref = frame_s2[i, :]

        tau, _, peaks_num = gcc_phat(sig,
                                     ref,
                                     fs=fs,
                                     max_tau=MAX_TDOA,)

        Tau.append(tau)

    return Tau


def preprocess(x, fs, preprocessedDataQueue):
    frameSize_time = 0.032
    step_time = 0.016
    frameSize = round(frameSize_time*fs)
    step = round(step_time*fs)

    s1 = x[0, :]
    s2 = x[1, :]

    # 归一化
    s1 = s1 / np.max(np.abs(s1))
    s2 = s2 / np.max(np.abs(s2))
    
    # 分帧加窗
    frame_s1 = enframe(s1, frameSize, step).T
    frame_s2 = enframe(s2, frameSize, step).T
    # print(frame_s1.shape)   # 16,400

    # 语音端点检测
    # [x11, x12] = vad(fs, frame_s1)
    # frame_s1 = frame_s1[x11:x12]
    # frame_s2 = frame_s2[x11:x12]
    # print(x11)
    # print(x12)
    voice_seg = vad(fs, frame_s1)
    if voice_seg != 0:
        for i in range(len(voice_seg.keys())):
            frame_s1_ = frame_s1[voice_seg[i]['start']:voice_seg[i]['end']]
            frame_s2_ = frame_s2[voice_seg[i]['start']:voice_seg[i]['end']]
            preprocessedDataQueue.put([frame_s1_, frame_s2_])

    # return [frame_s1, frame_s2]


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
    kde = KernelDensity(kernel='gaussian', bandwidth=0.000006).fit(
        Tau)

    # 生成概率密度曲线
    log_dens = kde.score_samples(X_plot)
    pdf = np.exp(log_dens)

    # 概率密度曲线的归一化
    pdf = (pdf-np.min(pdf))/(np.max(pdf)-np.min(pdf))

    # 峰值搜索，prominence 为突出程度限制
    peaks, properties = signal.find_peaks(
        pdf, prominence=0.1, height=0.7)
    source_num = len(peaks)
    tau_result = ind[peaks]
    # peak = np.argmax(peaks)
    # tau_result = ind[peak]
    # tau_result = np.array([tau_result])

    # print("tau result length: ", len(tau_result))
    # if (len(tau_result) > 2):
    # plt.plot(ind, pdf)
    # plt.show()
    # while True:
    #     pass

    return tau_result


def calculate_theta(tau, MAX_TDOA):
    a = tau/MAX_TDOA

    if a > 1:
        a = 1
    elif a < -1:
        a = -1

    theta = np.arcsin(a)
    return theta


def mp_preprocess(fs, rawDataQueue, preprocessedDataQueue):
    while True:
        data = rawDataQueue.get()
        for i in range(36):
            data = np.concatenate([data, rawDataQueue.get()], axis=1)
        print(data.shape)
        preprocess(data, fs, preprocessedDataQueue)
        # print("preprocessedDataQueue: ", preprocessedDataQueue.qsize())


def mp_postprocess(gccedDataQueue, mic_distance, fs):
    # 常数设置
    SOUND_SPEED = 340.0
    MAX_TDOA = mic_distance / float(SOUND_SPEED)

    while True:
        Tau = gccedDataQueue.get()
        if len(Tau) == 0:
            print("warning: len(Tau) equals to zero !!!")
        else:
            tau_result = postprocess(Tau)
            for i in range(tau_result.shape[0]):
                theta = calculate_theta(tau_result[i], MAX_TDOA)
                print("theta: ", theta)
        # print("gccedDataQueue: ", gccedDataQueue.qsize())


def mp_gcc(mic_distance, fs, preprocessedDataQueue, gccedDataQueue):
    # 常数设置
    SOUND_SPEED = 340.0
    MAX_TDOA = mic_distance / float(SOUND_SPEED)

    pool = Pool(processes=4)

    while True:

        x = preprocessedDataQueue.get()
        frame_s1 = x[0]
        frame_s2 = x[1]
        # point = frame_s1.shape[0] // 4

        # a = pool.apply_async(
        #     gcc_in_range, (frame_s1[0:point, :], frame_s2[0:point, :], fs, MAX_TDOA))
        # b = pool.apply_async(
        #     gcc_in_range, (frame_s1[point:2*point, :], frame_s2[point:2*point, :], fs, MAX_TDOA))
        # c = pool.apply_async(
        #     gcc_in_range, (frame_s1[2*point:3*point, :], frame_s2[2*point:3*point, :], fs, MAX_TDOA))
        # d = pool.apply_async(
        #     gcc_in_range, (frame_s1[3*point:, :], frame_s2[3*point:, :], fs, MAX_TDOA))

        # a.wait()
        # b.wait()
        # c.wait()
        # d.wait()

        # Tau = a.get() + b.get() + c.get() + d.get()

        Tau = gcc_in_range(frame_s1, frame_s2, fs, MAX_TDOA)
        gccedDataQueue.put(Tau)
