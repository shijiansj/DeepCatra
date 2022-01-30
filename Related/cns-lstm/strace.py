import os

def strace(z_pid):
    # output = os.popen("adb shell pm list package -3")
    os.popen("adb shell strace -o /sdcard/temp.txt -e trace=all -f -p "+z_pid)
    # result = os.popen('ps')

    # print(z_pid)


def getPid(process, apk):
    process_id = -1

    process_info = os.popen("adb shell ps | adb shell grep "+process)  # root      1672  1     1342592 71204 poll_sched ae2a0424 S zygote
    try:
        process_id = process_info.read().split()[1]  # split无参数，则默认将所有空格当分割符，进行分割, 获取到指定进程的pid

    except IndexError:
        with open('install_error.txt', 'a+', encoding='utf-8') as f:
            f.write(apk + '\n')
        print("获取pid失败")

        # print(zygote_info)
    # except IndexError:

    return process_id



# print(getPid("zygote"))
# strace()