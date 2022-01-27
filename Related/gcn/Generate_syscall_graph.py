import os

nodes_list = ['recvfrom','write','ioctl','read','sendto','dup','writev','pread','close','socket','bind','connect','mkdir','access','chmod','open','fchown','rename','unlink','pwrit','unmask','fcntl64','recvmsg','sendmsg','getdents64','epoll_wait']
nodes_dict = {}
for i in range(len(nodes_list)):
    nodes_dict[nodes_list[i]] = i+1

#根据apk的系统调用序列生成系统调用图的边
def generate_edge(source_path,des_path):

    apks_list = os.listdir(source_path)
    for apk_txt in apks_list:
        syscall_seq_txt_path = source_path+'\\'+apk_txt
        edge_txt = des_path+'\\'+apk_txt

        syscall_seq_list = []
        edge = []
        if not os.path.exists(edge_txt):
            with open(syscall_seq_txt_path, 'r',encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    syscall_seq_list.append(line)
                    file.close()
            for i in range(len(syscall_seq_list)):
                if syscall_seq_list[i] in nodes_dict.keys():
                    if i == len(syscall_seq_list) - 1:
                        break
                    if syscall_seq_list[i + 1] in nodes_dict.keys():
                        temp = [nodes_dict[syscall_seq_list[i]], nodes_dict[syscall_seq_list[i + 1]]]
                        if temp not in edge:
                            edge.append(temp)
            if not os.path.exists(edge_txt):
                with open(edge_txt, 'w', encoding='utf-8') as w_file:
                    for i in range(len(edge)):
                        w_file.write(str(edge[i][0]))
                        w_file.write(' ')
                        w_file.write(str(edge[i][1]))
                        w_file.write('\n')
                    w_file.close()

generate_edge(malware_sysycall_path,malware_edge_path)
generate_edge(benign_sysycall_path,benign_edge_path)
