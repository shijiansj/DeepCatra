# -*- coding:utf-8 -*-

import os
import datetime
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.bytecodes import dvm
from androguard.core.analysis.analysis import Analysis
import sys
import multiprocessing

# 分析的apk目录
# apks = r"F:\apks\benign"
# apks = r"F:\apks\benign_error"

# 存放结果的目录
# result = "static_benign_result"
# result = "static_new"

# 配置运行的线程数（默认为cpu核心数-3）
run_thread_num = multiprocessing.cpu_count() - 3


# 获取每个方法内的所有api
def get_method_apis(bytecode_buff):

    method_apis = []

    line_s = bytecode_buff.split("\n")



    for line in line_s:
        if line!="" and line.startswith("\t") and (not line.startswith("	(")) and len(line.strip()) > 20:
            line = line.strip()

            if line.find("invoke") != -1:

                invoke_info = line[line.find("L"):]
                if invoke_info.find('->') == -1:
                    continue
                invoke_method_full_name = invoke_info.replace("->",".").replace("(",":(")
                # if invoke_method_full_name.startswith("Ljava") or invoke_method_full_name.startswith("Landroid"):

                method_apis.append(invoke_method_full_name.lower())

    return method_apis


# 将apk中的方法名转换成指定格式
def format_method(method_full_name):
    method_full_name = method_full_name.lower()
    class_name = method_full_name[:method_full_name.find(" ")]
    other = method_full_name[method_full_name.find(" ") + 1:]
    method_name = other[:other.find(" ")]
    params = other[other.find(" ") + 1:]
    return class_name + "." + method_name + ":" + params


#对apk反编译，并获取方法字典，包含每个节点以及invoke信息
def decompile(apk_path):
    api_list = []

    time_start = datetime.datetime.now()
    try:
        apk_obj = APK(apk_path)
    except Exception:
        print("apk反编译出错")
        return 0,0
    time_end = datetime.datetime.now()
    decompile_time = time_end - time_start
    print("反编译完成，总用时：", decompile_time)
    print("---------------------------------")

    apks_methods = []   # 用户自定义的方法集合

    for dex in apk_obj.get_all_dex():
        dex_obj = DalvikVMFormat(dex)
        analysis = Analysis(dex_obj)
        for cls in dex_obj.get_classes():
            for method in cls.get_methods():
                apks_methods.append(format_method(method.full_name))
                bytecode_buff = dvm.get_bytecodes_method(dex, analysis, method)

                if method.get_access_flags_string().find("native") == -1:
                    api_list.extend(get_method_apis(bytecode_buff))

    standard_apis = [api for api in api_list if api not in apks_methods]   # api不是用户自定义的方法



    print("提取apk信息完成...")
    print("---------------------------------")
    return standard_apis


# 将一个列表均分为n个
def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1

    for i in range(0, n):
        yield origin_list[i * cnt:(i + 1) * cnt]


def extract(apk_folder, apks_list, result_folder):
    cnt = 0
    for apk in apks_list:
        apk_name=apk[:apk.rfind(".")]
        cnt += 1
        if os.path.exists(result_folder + "/" + apk_name + ".txt"):
            print("提取过的apk：",apk)
            continue

        start_time=datetime.datetime.now()
        print("apk开始时间：",start_time)

        print("开始处理 {}：{}".format(cnt, apk_name))
        try:
            api_list = decompile(apk_folder+"/"+apk)
            with open(result_folder + "/" + apk_name + ".txt", "w", encoding="utf-8") as file:
                for line in api_list:
                    line = line.replace(' ', '')
                    file.write(line + "\n")

            end_time=datetime.datetime.now()
            print("apk结束时间：", end_time)
            print("---------------------------------")
        except Exception as e:
            print("出错的apk:",apk_name)
            print(e)
            continue


def main(apk_folder, result_folder):
    all_apks = os.listdir(apk_folder)

    processes = run_thread_num

    thread_list = []
    for apks_list in split_list_n_list(all_apks, processes):
        thread = multiprocessing.Process(target=extract,
                                         args=(apk_folder, apks_list, result_folder))
        thread_list.append(thread)

    # 开启所有子进程
    for thread in thread_list:
        thread.start()

    # 等待所有子进程运行结束
    for thread in thread_list:
        thread.join()


if __name__=="__main__":
    apks = sys.argv[1]
    result = sys.argv[2]
    main(apks, result)

