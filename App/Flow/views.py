# App/Flow/views.py
# import from framework
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
# import from project
from .models import get_flow_data
import numpy as np
import datetime

@api_view(['POST'])
def show_flow(request):
    cur_day = datetime.datetime.now().day
    pred = np.load('/Users/wanghao/Downloads/Multitask-learning2/mae_compare/predict_day_25_1-0.2_0219.npy') * 3000
    pred = np.around(pred, 0)
    pred[pred < 2] = 0  # 玄学
    def trans(inlist):
        return dict((k_1, dict((k_2, dict((k_3, v_3) for k_3, v_3 in zip(['in', 'out'], v_2))) for k_2, v_2 in
                               zip(range(1, len(v_1) + 1), v_1))) for k_1, v_1 in
                    zip(range(1, len(inlist) + 1), inlist))
    def retrive_one_station(i, Dic):
        return {k_1: v_1[i] for k_1, v_1 in Dic.items()}
    def regroup(flatDic):
        return {"date_%d" % (cur_day + i): {j: flatDic[i * 144 + j] for j in range(1, 145)} for i in range(1)}
    intlist = [[[int(i) for i in j] for j in k] for k in pred.tolist()]
    Dic = trans(intlist)
    # 以下是将Dic中的(144*1天, 81, 2)字典重排为(81, 144, 2)字典, 并存到中转变量dbCache中
    dbCache = {}
    for i in range(1, 82):
        dbCache["station_%d" % i] = regroup(retrive_one_station(i, Dic))
    print("=== final Dic ===")
    print("dbCache: " + str(type(dbCache)))
    new_keys = ['station_%d' % request.data["stations"][0]]  # 获取请求的站点id
    new_data = {key: dbCache[key] for key in new_keys}

    print(request.data)
    data = get_flow_data(year=request.data["year"],
                         month=request.data["month"],
                         dates=request.data["dates"], 
                         stations=request.data["stations"])
    if data == "":
        return JsonResponse(data={"errors": "StationFlow Not Exist"})
    return JsonResponse(data=data, safe=False)


@api_view(['GET'])
def show_board(request):
    return JsonResponse(data={"errors": "Board data Not Support"})
