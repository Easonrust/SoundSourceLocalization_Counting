from ctypes import cdll
import socket
import struct
import pickle
from multiprocessing import Process, Queue

from doa_2mics import gcc_in_range, preprocess, postprocess, mp_preprocess, mp_gcc, mp_postprocess
import numpy as np
import time


def c_record():
    print("c_record")
    ll = cdll.LoadLibrary
    lib = ll("./record.so")
    lib.getData()


def record_server(rawDataQueue, host, port):

    print("mp_record")

    s = socket.socket()
    s.bind((host, port))
    s.listen(5)
    c, addr = s.accept()
    print("address: ", addr)

    buffer = bytes('', 'utf-8')

    current_channel = 0

    list0 = 0
    list1 = 0

    count = 0

    while True:
        while len(buffer) < 5120:
            data = c.recv(10240)
            buffer = buffer + data

        while len(buffer) >= 5120:
            audio = buffer[0:5120]
            buffer = buffer[5120:]
            tmp = struct.unpack("h" * 2560, audio)
            if current_channel == 0:
                list0 = list(tmp)
            elif current_channel == 1:
                list1 = list(tmp)
                rawDataQueue.put(np.array([list0, list1]))
                count = count + 1
                # print("rawDataQueue: ", rawDataQueue.qsize())
                # print("rec: ", count)
            current_channel = (current_channel + 1) % 4


if __name__ == '__main__':
    rawDataQueue = Queue()
    preprocessedDataQueue = Queue()
    gccedDataQueue = Queue()

    host = "127.0.0.1"
    port = 8877
    recordProcess = Process(target=record_server,
                            args=(rawDataQueue, host, port, ))
    recordProcess.start()

    preprocessProcess = Process(target=mp_preprocess, args=(
        16000, rawDataQueue, preprocessedDataQueue, ))
    preprocessProcess.start()

    gccProcess = Process(target=mp_gcc, args=(
        0.1, 16000, preprocessedDataQueue, gccedDataQueue, ))
    gccProcess.start()

    postprocessProcess = Process(
        target=mp_postprocess, args=(gccedDataQueue, 0.1, 16000, ))
    postprocessProcess.start()

    time.sleep(1)
    c_record()
