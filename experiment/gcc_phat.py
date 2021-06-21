import numpy as np
from scipy.fftpack import rfft, irfft
from scipy.signal import find_peaks


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=64):
    n = sig.shape[0] + refsig.shape[0]
    # 应该是2的n次方

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    # cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(interp * n / 2)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    peaks, _ = find_peaks(cc, max(cc)/2)

    if len(peaks) > 0:

        sort = sorted(cc[peaks], reverse=True)
        zzz = []
        for i in range(len(peaks)):
            zzz.append(np.array(np.where(cc == sort[i])))
        zzz = np.array(zzz).reshape(-1)
        K = zzz[0]
    else:
        print("not enough")
        K = 0

    shift = K - max_shift
    tau = shift / float(interp * fs)

    return tau, cc, len(peaks)
