import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import time
from gcc_phat import gcc_phat
from utils import enframe
from vad import vad
from multiple_vad import multi_vad
from scipy import signal
from itertools import permutations
from sklearn import cluster
from collections import Counter
from sklearn.neighbors import KernelDensity

# 常数设置
SOUND_SPEED = 340.0
MIC_DISTANCE = 0.1
MAX_TDOA = MIC_DISTANCE / float(SOUND_SPEED)


def calculate_frame_tdoa_between_two_signal(frame_s1, frame_s2, mic_distance, fs):
    # 每一帧计算TDOA
    time_mg_start = time.time()
    Tau = []
    for i in range(frame_s1.shape[0]):

        sig = frame_s1[i, :]
        ref = frame_s2[i, :]

        tau, _, peaks_num = gcc_phat(sig,
                                     ref,
                                     fs=fs,
                                     max_tau=MAX_TDOA,)

        # 根据需要可显示每一帧的互相关函数图
        # plt.plot(_)
        # plt.show()
        Tau.append(tau)
    time_mg_end = time.time()
    print('multiple gcc cost', time_mg_end-time_mg_start, 's')

    time_kde_start = time.time()

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
        pdf, prominence=0.1, height=0.2)
    source_num = len(peaks)
    tau_result = ind[peaks]
    time_kde_end = time.time()
    print('kde cost', time_kde_end-time_kde_start, 's')

    # 根据需要绘制最后的概率密度曲线
    # plt.plot(ind, pdf)
    # plt.show()
    return tau_result


def simulate():
    room_dim = [10, 10]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio1 = wavfile.read("data/speech1.wav")
    fs, audio2 = wavfile.read("data/speech2.wav")
    fs, audio3 = wavfile.read("data/speech3.wav")
    fs, audio4 = wavfile.read("data/speech4.wav")

    m = pra.Material(energy_absorption="fibre_absorber_2")

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=m
    )

    # place the source in the room
    room.add_source([4, 4], signal=audio1, delay=0)
    # room.add_source([4, 8], signal=audio2, delay=0)
    # room.add_source([6, 4], signal=audio3, delay=0)
    # room.add_source([30, 59], signal=audio1, delay=0)
    # room.add_source([1, 4], signal=audio4, delay=0)

    # room.add_source([10, 15.5], signal=audio, delay=6)

    # define the locations of the microphones
    # 0、1组成麦克风阵列A，2、3组成麦克风阵列B
    mic_locs = np.c_[
        [0, 0], [0.1, 0], [10, 0], [9.9, 0]
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)
    room.simulate()

    # 信号获取
    s1 = room.mic_array.signals[0, :]
    s2 = room.mic_array.signals[1, :]
    s3 = room.mic_array.signals[2, :]
    s4 = room.mic_array.signals[3, :]

    # 归一化
    s1 = s1 / np.max(np.abs(s1))
    s2 = s2 / np.max(np.abs(s2))
    s3 = s3 / np.max(np.abs(s3))
    s4 = s4 / np.max(np.abs(s4))

    # 分帧加窗
    frameSize_time = 0.032
    step_time = 0.016
    frameSize = round(frameSize_time*fs)
    step = round(step_time*fs)
    frame_s1 = enframe(s1, frameSize, step).T
    frame_s2 = enframe(s2, frameSize, step).T
    frame_s3 = enframe(s3, frameSize, step).T
    frame_s4 = enframe(s4, frameSize, step).T

    # 语音端点检测
    time_vad_start = time.time()
    [x11, x12] = vad(fs, frame_s4)
    time_vad_end = time.time()
    print('vad cost', time_vad_end-time_vad_start, 's')
    frame_s1 = frame_s1[x11:x12]
    frame_s2 = frame_s2[x11:x12]
    frame_s3 = frame_s3[x11:x12]
    frame_s4 = frame_s4[x11:x12]

    frame_num = x12 - x11

    # plt.plot(s1)
    # plt.plot(int(x11*step), 1, '.k')
    # plt.plot(int(x12*step), 1, 'or')
    # plt.show()

    # 两麦克风阵列分别计算TDOA

    time_gcc_start = time.time()

    tau_result1 = calculate_frame_tdoa_between_two_signal(
        frame_s1, frame_s2, MIC_DISTANCE, fs)

    tau_result2 = calculate_frame_tdoa_between_two_signal(
        frame_s3, frame_s4, MIC_DISTANCE, fs)

    time_gcc_end = time.time()
    print('gcc cost', time_gcc_end-time_gcc_start, 's')

    # 声源个数
    source_num = np.max((len(tau_result1), len(tau_result2)))

    # 输出声源的个数
    print(source_num)
    Location = []

    if len(tau_result1) != len(tau_result2):
        print("cannot do the Location")
    elif len(tau_result1) == 1:

        # 单声源部分
        theta1 = calculate_theta(tau_result1[0])
        theta2 = calculate_theta(tau_result2[0])
        x = 10*np.tan(theta1)/(np.tan(theta1)+np.tan(theta2))
        y = 10/(np.tan(theta1)+np.tan(theta2))
        print((x, y))
    else:
        n1 = s1.shape[0]
        n2 = s2.shape[0]
        n3 = s3.shape[0]
        n4 = s4.shape[0]

        P_result = []

        # 为了节省计算时间，截取了最中间的几帧进行配对统计
        # 每一帧获得一个位置结果，最后对所有的位置结果做一个K-means聚类
        # for i in range(frame_num):
        time3_start = time.time()
        for i in range(int(frame_num/2-frame_num/4), int(frame_num/2+frame_num/4)):

            time1_start = time.time()
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
                theta1 = calculate_theta(tau_result1[i])
                theta2 = calculate_theta(tau_result2[p_result[i]])
                x = 10*np.tan(theta1)/(np.tan(theta1)+np.tan(theta2))
                y = 10/(np.tan(theta1)+np.tan(theta2))

                Location.append((x, y))

            time1_end = time.time()
            print('time cost', time1_end-time1_start, 's')
        time3_end = time.time()
        print('gcc cost', time3_end-time3_start, 's')

        # # 众数法
        # p_result = Counter(P_result).most_common(1)
        # p_result = p_result[0][0]
        # result = []
        # for i in range(len(p_result)):
        #     theta1 = calculate_theta(tau_result1[i])
        #     theta2 = calculate_theta(tau_result2[p_result[i]])
        #     x = 10*np.tan(theta1)/(np.tan(theta1)+np.tan(theta2))
        #     y = 10/(np.tan(theta1)+np.tan(theta2))

        #     Location.append((x, y))
        #     result.append((x, y))
        # print(result)

        k_means = cluster.KMeans(n_clusters=source_num)
        k_means.fit(Location)
        center = k_means.cluster_centers_

        # 输出最后确定的位置坐标
        print(center)

        # 聚类法图
        # x1 = [x[0] for x in Location]
        # y1 = [x[1] for x in Location]
        # plt.scatter(x1, y1, c="blue")

        # x2 = [x[0] for x in center]
        # y2 = [x[1] for x in center]
        # plt.scatter(x2, y2, c="red")

        # plt.xlim(0, 10)  # limited the length of axis
        # plt.ylim(0, 10)
        # plt.show()


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


def calculate_theta(tau):
    a = tau/MAX_TDOA
    if a > 1:
        a = 1
    elif a < -1:
        a = -1

    theta = np.arcsin(a)
    return theta


def test():
    fs, s1 = wavfile.read("log/channel-0.wav")
    fs, s2 = wavfile.read("log/channel-1.wav")

    # 归一化
    s1 = s1 / np.max(np.abs(s1))
    s2 = s2 / np.max(np.abs(s2))

    # 分帧加窗
    frameSize_time = 0.032
    step_time = 0.016
    frameSize = round(frameSize_time*fs)
    step = round(step_time*fs)
    frame_s1 = enframe(s1, frameSize, step).T
    frame_s2 = enframe(s2, frameSize, step).T

    voiceseg1 = multi_vad(fs, frame_s1)
    voiceseg2 = multi_vad(fs, frame_s2)

    voiceseg = {}

    if(len(voiceseg1) > len(voiceseg2)):
        voiceseg = voiceseg2
    else:
        voiceseg = voiceseg1

    for i in range(len(voiceseg)):
        x11 = voiceseg[i]['start']
        x12 = voiceseg[i]['end']
        i_frame_s1 = frame_s1[x11:x12]
        i_frame_s2 = frame_s2[x11:x12]
        tau_result = calculate_frame_tdoa_between_two_signal(
            i_frame_s1, i_frame_s2, MIC_DISTANCE, fs)
        print(tau_result)
    print(calculate_theta(tau_result[0])*180/np.pi)


if __name__ == '__main__':
    time2_start = time.time()
    # simulate()
    simulate()
    time2_end = time.time()
    print('program time cost', time2_end-time2_start, 's')
