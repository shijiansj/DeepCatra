import sys
from androguard.core.bytecodes import dvm
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.apk import APK
import os
from strace import getPid, strace
from monkey_test import monkey_test
import datetime
import time
import sys


def syscall_deal(apk_path, des_path):      # 将中间结果转化为系统调用序列
    with open(apk_path,'r',encoding='utf_8') as read_file, open(des_path, 'w', encoding='utf_8') as write_file:
        for line in read_file.readlines():
            line = line.strip('\n')
            if 'resume' in line:
                continue
            if '(' in line :
                tmp = line.split()[1]
                syscall = tmp[:tmp.find('(')]
                write_file.write(syscall+'\n')


def extract_syscall(apks_path, tmp_path, out_path):
    start_time = datetime.datetime.now()
    print('开始时间：{}'.format(start_time))
    with open('install_error.txt', 'r', encoding='utf-8') as f1:
        error_log = f1.read()

    cnt = 0
    for apk in os.listdir(apks_path):
        print("处理apk：",apk)
        apk_name, ext = os.path.splitext(apk)
        if(os.path.exists(os.path.join(out_path, apk_name+".txt"))):
            print("该apk结果已存在: ",apk)
            continue
        if apk in error_log:
            print('该apk无法成功安装')
            continue

        cnt += 1

        apk_path = os.path.join(apks_path, apk)
        apk_obj = APK(apk_path)         #反编译apk
        package_name = apk_obj.get_package()
        result = os.popen("adb install " + apk_path)  # 安装apk

        if "Success" in result.read():  # 输出安装结果
            print("apk安装成功")
        else:
            print("apk安装失败")
            with open('install_error.txt','a+',encoding='utf-8') as f:
                f.write(apk+'\n')

            pids = os.popen('tasklist /svc > pid.txt').read()    # 得到player进程的pid
            player_pid = 0
            with open('pid.txt', 'r') as f:
                for line in f.readlines():
                    if (line.startswith('player.exe')):
                        player_pid = line.split()[1]
                        print('player.exe  pid:', player_pid)
            os.remove('pid.txt')
            # if(player_pid != 0):
            #     os.popen('taskkill /im ' + str(player_pid) + ' /f')     # 关闭模拟器

            emu_result = os.popen(r'D:\Downloads\VirtualBox\VBoxManage list vms').read()
            emu_id = emu_result[:emu_result.rfind('"') + 1]
            os.popen(r'D:\Downloads\Genymobile\Genymotion\player --vm-name ' + emu_id)  # 启动模拟器
            time.sleep(10)
            continue

        main_activity = apk_obj.get_main_activity()
        if main_activity == None:
            os.popen('aapt dump badging ' + apk_path + ' > main.txt')
            time.sleep(1)
            with open('main.txt', 'r', encoding='utf-8') as f:
                aapt_result = f.read().split('\n')
                for i in aapt_result:
                    if i.startswith('launchable-activity'):
                        name = i.split()[1]
                        main_activity = name[name.find("'") + 1:name.rfind("'")]
                        # print(name)
                # print(activity)
        if(main_activity!=None):
            activity = package_name+'/'+main_activity
            os.popen("adb shell am start -n "+activity)     #启动应用
            time.sleep(2)
            pid = getPid(package_name, apk)                       #获取应用程序pid
            print("该应用程序的pid：",pid)
            if(pid==-1):
                uninstall = os.popen("adb uninstall " + package_name)  # 卸载该app
                if ("Success" in uninstall.read()):
                    print("该apk已被卸载： " + apk)
                else:
                    print("卸载该apk时出现异常，apk未卸载成功")
                continue
            strace(pid)                           #跟踪当前的应用程序

            monkey_test(package_name)  # 进行monkey测试
                # time.sleep(20)
            # os.popen("adb shell kill " + pid)  # 关闭strace
            os.popen("adb pull /sdcard/temp.txt " + tmp_path)  # 提取输出文件到pc端
            time.sleep(5)
            name,_ = os.path.splitext(apk)

            os.rename(os.path.join(tmp_path,"temp.txt"),os.path.join(tmp_path,name+".txt"))    #对temp.txt改名

            syscall_deal(os.path.join(tmp_path,name+".txt"), os.path.join(out_path,name+".txt"))
            if(os.path.exists(os.path.join(tmp_path,name+".txt"))):
                os.remove(os.path.join(tmp_path,name+".txt"))
                print('中间结果已被删除')

            uninstall_result = os.popen("adb uninstall "+package_name)     #卸载该app
            if ("Success" in uninstall_result.read()):
                print("该apk已被卸载： " + apk)
            else:
                print("卸载该apk时出现异常，apk未卸载成功")
            os.popen("adb shell rm /sdcard/temp.txt")   #删除android系统中残留的文件

        else:
            print("未找到main_activity,启动应用程序失败")
            uninstall_result2 = os.popen("adb uninstall "+package_name)     #卸载该app
            if ("Success" in uninstall_result2.read()):
                print("该apk已被卸载： " + apk)
            else:
                print("卸载该apk时出现异常，apk未卸载成功")

    print('用时：{}'.format(datetime.datetime.now()-start_time))


if __name__ == '__main__':
    apks_path = sys.argv[1]
    tmp_path = sys.argv[2]
    out_path = sys.argv[3]
    extract_syscall(apks_path, tmp_path, out_path)