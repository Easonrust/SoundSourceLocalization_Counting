## 多声源麦克风阵列定位算法

### 1. 算法整体流程

![算法原理图](https://tva1.sinaimg.cn/large/008i3skNly1grq73zqsehj313e0kdwhg.jpg)

### 2. 关键步骤

- 加窗分帧

  由于语音信号是时变信号，在10～40ms内短时平稳，需要进行分帧处理，对每一帧32ms进行处理，为保证平稳过渡，采取帧移为16ms。采用汉明窗进行分帧操作，并进行预加重处理。

  > 相关代码见utils.py/enframe

- 端点检测

  为了减少运算量，需要将语音信号段提取出来，这里采用基于短时过零率和短时能量的双门限法进行检测（短时能量为主，短时过零率为辅）。

  > 相关代码见vad.py

- 直方图统计后进行核密度估计

  主要利用了信号在时频域上的稀疏特性，假设每一个时频点或时频域内只有一个声源的信号成分占主导地位（这里我们采取的时频点或时频域为一帧）。这样计算每一帧主导声源的时间延迟估计，将多帧结果统一起来绘制直方图，之后使用高斯核密度估计的方法进行拟合，再搜索峰值。

  > 相关代码见doa.py/calculate_frame_tdoa_between_two_signal

- 基于入射信号功率的交叉定位

  采用计算入射信号功率的方式进行角度的配对。

### 3. 文件组织

- doa.py

  算法仿真测试的主体文件

- vad.py

  语音检测的代码，仅可进行单段的语音检测，目前没有被使用

- multiple_vad.py

  语音检测的代码，可进行多段语音检测，目前被使用

- gcc_phat.py 

  利用基本的GCC_PHAT算法计算两路信号之间的时间延迟

- utils.py

  一些通用函数

- data

  - audio_-58.21与audio_58.21文件夹下为160ms长度的麦克风实测数据，由于两阵列没有按照规定摆放，只可用于测向。
  - audio0-3文件夹下为长度为7秒的麦克风实测数据，由于两阵列没有按照规定摆放，只可用于测向。
  - 其余数据为使用电脑麦克风录制的音频数据，可作为模拟声源的源文件使用

### 4. 运行方式

1. 模拟环境下测试定位算法，在终端运行doa.py文件即可
2. 模拟环境下测试端点检测，在终端运行vad.py或者multiple_vad.py

### 5. 一些有用的参考资料

- 采用的仿真测试环境Pyroomacoustics 的说明文档

  https://pyroomacoustics.readthedocs.io/en/pypi-release/index.html

- 端点检测的方法(MATLAB)

  https://blog.csdn.net/ziyuzhao123/article/details/8932336

  https://blog.csdn.net/qcyfred/article/details/53007018?utm_medium=distribute.pc_relevant_bbs_down.none-task--2~all~first_rank_v2~rank_v29-8.nonecase&depth_1-utm_source=distribute.pc_relevant_bbs_down.none-task--2~all~first_rank_v2~rank_v29-8.nonecase

- 对于GCC-PHAT方法测向的理解

  https://www.funcwj.cn/2018/05/10/gcc-phat-for-tdoa-estimate/

- 波束形成技术的理解

  https://blog.csdn.net/weixin_40679412/article/details/80230163

- 可用于测试文件的数据库

  CMU_ARCTIC数据库：http://festvox.org/cmu_arctic/

- 对理解更有帮助的论文

  王松. 基于TDOA的声源定位算法研究与实现[D].山东大学,2020.

  张晓丹. 麦克风阵列下子带分析的多声源定位算法研究[D].太原理工大学,2016.

  孙俊岱. 基于信号稀疏特性的多声源定位及分离技术研究[D].北京工业大学,2018.

  马亚南. 基于小型阵列MIC的声源定位系统设计与实现[D].东南大学,2018.

  肖骏. 基于麦克风阵列的实时声源定位技术研究[D].电子科技大学,2015.

  徐佳新. 基于声传感网的多声源定位方法研究[D].南京理工大学,2017.

  Salvati D, A Rodà, Canazza S, et al. A real-time system for multiple acoustic sources localization based on isp comparison[C]// International Conference on Digital Audio Effects. 2010.



