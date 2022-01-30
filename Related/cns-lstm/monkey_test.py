#用于进行monkey测试

import re
import os

def monkey_test(packageName):     #对指定apk进行monkey测试
    # adb shell monkey --ignore-crashes --ignore-timeouts --throttle 500 --pct-motion 40 --pct-touch 45 --pct-syskeys 5 --pct-appswitch 5 --pct-anyevent 5 --kill-process-after-error -s 100 -p
    test_result = os.popen("adb shell monkey  --throttle 500 "       # orign 500 500
                           "--pct-motion 30 --pct-touch 55 --pct-syskeys 3 --pct-appswitch 3 --pct-anyevent 9 --kill-process-after-error -s 100 -p "+packageName+" 500")
                                                                 #暂定执行次数为500，时间间隔为500，设置各个事件的占比,若触发程序崩溃，则立即停止
    print(test_result.buffer.read().decode(encoding='utf-8'))
    print("monkey test is ok !")






