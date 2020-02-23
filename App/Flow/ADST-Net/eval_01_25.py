# -*- coding:utf-8 -*-
import pandas as pd
import datetime
import numpy as np
from .Multitask_ADST import stresnet
import warnings

warnings.filterwarnings('ignore')

N_hours = 24
N_time_slice = 6
N_station = 81
N_flow = 2
len_seq1 = 2  # week时间序列长度为2
len_seq2 = 3  # day时间序列长度为3
len_seq3 = 5  # hour时间序列长度为5
nb_flow = 2  # 输入特征

# 加载预测日pred_day所需要依赖的天数据
pred_day = datetime.datetime.now().day
last_day = pred_day - 1
last_week = pred_day - 7
data = np.load('raw_node_data.npy')
data_0124 = data[144*(last_day-1):144*last_day, :, :]
data_0118 = data[144*(last_week-1):144*last_week, :, :]
edge_data = np.load('raw_edge_data.npy')
edge_data_0124 = edge_data[144*(last_day-1):144*last_day, :, :]
edge_data_0118 = edge_data[144*(last_week-1):144*last_week, :, :]

out_maximum = 30  # 估摸出站的最大阈值  因为在getMetroFlow函数中已经除了100近似归一化了
data_0118 /= out_maximum
data_0124 /= out_maximum
edge_out_maximum = 233.0
edge_data_0118 /= edge_out_maximum
edge_data_0124 /= edge_out_maximum

# ——————————————————————加载预训练权重模型———————————————————————————
model = stresnet(c_conf=(5, 2, 81), p_conf=(3, 2, 81), t_conf=(2, 2, 81),
                 c1_conf=(5, 81, 81), p1_conf=(3, 81, 81), t1_conf=(2, 81, 81),
                 external_dim1=1, external_dim2=1, external_dim3=None,
                 external_dim4=1, external_dim5=81, external_dim6=None,
                 external_dim7=None, external_dim8=None, external_dim9=1,
                 nb_residual_unit=4, nb_edge_residual_unit=4)     # 对应修改这里,和训练阶段保持一致
model.load_weights('/Users/wanghao/Downloads/Multitask-learning2/'
                   'log/edge_conv1d/units_12_3channel_618/85-0.00437290.hdf5')  # 替换这里哦，记住记住！

sum_of_predictions = 24*6 - 5  # 前5个没法用
# 注意别写反
matrix_01_18 = data_0118
matrix_01_24 = data_0124
matrix_01_25 = np.zeros([N_hours * N_time_slice, N_station, N_flow])

matrix_01_24_edge = edge_data_0118
matrix_01_18_edge = edge_data_0124
matrix_01_25_edge = np.zeros([N_hours * N_time_slice, N_station, N_station])

# 都初始化为零矩阵
xr_test = np.zeros([1, N_station, len_seq3 * N_flow])
xp_test = np.zeros([1, N_station, len_seq2 * N_flow])
xt_test = np.zeros([1, N_station, len_seq1 * N_flow])
xredge_test = np.zeros([1, N_station, len_seq3 * N_station])
xpedge_test = np.zeros([1, N_station, len_seq2 * N_station])
xtedge_test = np.zeros([1, N_station, len_seq1 * N_station])

# 特征1: 01/25是周五(weekday信息)
x_val_external_information1 = np.zeros([24*6, 1])
x_val_external_information1[:, 0] = 4
# 特征2: 时间片信息
x_val_external_information2 = np.zeros([24*6, 1])
HOUR = 0
for i in range(0, 24*6):
    x_val_external_information2[i, 0] = HOUR
    HOUR = HOUR + 1
# 特征3: 天气信息(25号为多云)
x_val_external_information4 = np.zeros([24*6, 1])
x_val_external_information4[:, 0] = 1
# 特征4: 闸机信息
x_val_external_information5 = np.zeros([24*6, 81])
t = np.load('sluice_machines.npy')
t = t[:, 0]
for i in range(144):
    x_val_external_information5[i, :] = t
# 特征5: 峰时段信息
x_val_external_information9 = np.zeros([N_hours * N_time_slice, 1])
# #——————————————————早晚高峰—————————————————————
x_val_external_information9[39:54, 0] = 2  # 7：30 - 9：00
x_val_external_information9[102:114, 0] = 2  # 17：00 - 19：00
# #——————————————————高峰—————————————————————————
x_val_external_information9[33:39, 0] = 1  # 6:30-7:30
x_val_external_information9[63:70, 0] = 1  # 10:30-11:30
x_val_external_information9[99:102, 0] = 1  # 16:30-17:30
x_val_external_information9[114:132, 0] = 1  # 19:00-22:00


def pred_ADST():
    for i in range(sum_of_predictions):
        t = matrix_01_18[i + 4:i + 6, :, :]  # trend = 2
        p = matrix_01_24[i + 3:i + 6, :, :]  # period = 3
        r = matrix_01_25[i:i + 5, :, :]  # recent = 5
        er = matrix_01_24_edge[i:i + 5, :, :]
        ep = matrix_01_24_edge[i + 3:i + 6, :, :]
        et = matrix_01_24_edge[i + 4:i + 6, :, :]
        mask = np.ones([1, 81, 81])
        for j in range(len_seq3):
            for k in range(2):
                xr_test[0, :, j * 2 + k] = r[j, :, k]
        for j in range(len_seq2):
            for k in range(2):
                xp_test[0, :, j * 2 + k] = p[j, :, k]
        for j in range(len_seq1):
            for k in range(2):
                xt_test[0, :, j * 2 + k] = t[j, :, k]

        for j in range(len_seq3):
            for k in range(81):
                xredge_test[0, :, j * 81 + k] = er[j, :, k]
        for j in range(len_seq2):
            for k in range(81):
                xpedge_test[0, :, j * 81 + k] = ep[j, :, k]
        for j in range(len_seq1):
            for k in range(81):
                xtedge_test[0, :, j * 81 + k] = et[j, :, k]
        # 对应修改了这里
        ans, ans1 = model.predict([xr_test, xp_test, xt_test, xredge_test, xpedge_test, xtedge_test, mask,
                                   x_val_external_information1, x_val_external_information2,
                                   x_val_external_information4,
                                   x_val_external_information5, x_val_external_information9])
        matrix_01_25[i + 5, :, :] = ans
    return matrix_01_25

# np.save('mae_compare/predict_day_25_1-0.2_0219.npy', matrix_01_25)


print('Testing Done...')
