from ctypes import cdll
import socket
import struct
from multiprocessing import Process, Queue

from doa_4mics import gcc_in_range, preprocess, postprocess, mp_preprocess, mp_gcc, mp_postprocess, calculate_center
import numpy as np
import time
import json
from aiohttp import web


centerQueue = Queue()
httpRawDataQueue = Queue()


def c_record():
    print("c_record")
    ll = cdll.LoadLibrary
    lib = ll("./record.so")
    lib.getData()


def record_server(rawDataQueue, httpRawDataQueue, host, port):

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
    list2 = 0
    list3 = 0

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
            elif current_channel == 2:
                list2 = list(tmp)
            elif current_channel == 3:
                list3 = list(tmp)
                rawDataQueue.put(np.array([list0, list1, list2, list3]))
                httpRawDataQueue.put(json.dumps(
                    [{"id": 1, "data": [list0[::2], list1[::2]]}]))
                if httpRawDataQueue.qsize() > 1:
                    httpRawDataQueue.get()
                # print(httpRawDataQueue.get())
                count = count + 1
                # print("qsize: ", rawDataQueue.qsize())
                # print("rec: ", count)
            current_channel = (current_channel + 1) % 4


async def result_handle(request: web.Request) -> web.StreamResponse:
    if centerQueue.qsize() == 0:
        return web.Response(text=json.dumps([{"id": 0, "center": "there is not center current!"}]))
    while(centerQueue.qsize() > 1):
        centerQueue.get()
    res = centerQueue.get()
    text = str(res)
    return web.Response(text=text)


async def raw_data_handle(request: web.Request) -> web.StreamResponse:
    res = httpRawDataQueue.get()
    text = str(res)
    return web.Response(text=text)

if __name__ == '__main__':
    fs = 16000
    mic_distance = 0.1
    MAX_TDOA = mic_distance / float(340)
    rawDataQueue = Queue()
    preprocessedDataQueue = Queue()
    gccedDataQueue1 = Queue()
    gccedDataQueue2 = Queue()
    frameNumQueue = Queue()
    sQueue = Queue()
    frameQueue = Queue()
    nQueue = Queue()
    tauResultQueue1 = Queue()
    tauResultQueue2 = Queue()

    host = "127.0.0.1"
    port = 8877
    recordProcess = Process(target=record_server,
                            args=(rawDataQueue, httpRawDataQueue, host, port, ))
    recordProcess.start()

    preprocessProcess = Process(target=mp_preprocess, args=(
        fs, rawDataQueue, preprocessedDataQueue, frameNumQueue, sQueue, frameQueue, nQueue, ))
    preprocessProcess.start()

    gccProcess = Process(target=mp_gcc, args=(
        mic_distance, fs, preprocessedDataQueue, gccedDataQueue1, gccedDataQueue2, ))
    gccProcess.start()

    postprocessProcess = Process(
        target=mp_postprocess, args=(
            gccedDataQueue1, gccedDataQueue2, tauResultQueue1, tauResultQueue2, ))
    postprocessProcess.start()

    calCenterProcess = Process(
        target=calculate_center, args=(
            fs, MAX_TDOA, tauResultQueue1, tauResultQueue2, frameNumQueue, frameQueue, nQueue, centerQueue, ))
    calCenterProcess.start()

    time.sleep(1)

    recordProcess = Process(target=c_record, args=())
    recordProcess.start()
    # c_record()

    app = web.Application()
    app.add_routes(
        [web.get("/getCenter", result_handle),
         web.get("/getRawData", raw_data_handle)]
    )

    web.run_app(app, host='192.168.1.13', port=8080)
