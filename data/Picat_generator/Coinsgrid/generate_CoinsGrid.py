import random
import numpy as np
import subprocess
import time
import shutil
import os

# Coins Grid (minimization question with linear constrains)
# Rules: N*N grid ; C coins , must satisfy
# 1. In each row exactly C coins must be placed.
# 2. In each column exactly C coins must be placed.
# 3. The sum of the quadratic horizontal distance from the main diagonal of all cells containing
#    a coin must be as small as possible.
# 4. In each cell at most one coin can be placed.

# 似乎不必看了，类似 N 皇后，规模大1750, 500的内存太大，规模小的31*31秒解，懒得看了 (如何加sum约束) ...
# N=300 时，EasySAT 也解不出来 , 好消息。

def copy_file_to_directory(source_file_path, dest_file_path):
    """
    将一个文件从源路径复制到目标目录。
    :param source_file_path: 源文件的完整路径。
    :param destination_directory_path: 目标目录的路径。
    """
    # 确保目标目录存在
    destination_directory_path = os.path.dirname(dest_file_path)
    if not os.path.exists(destination_directory_path):
        os.makedirs(destination_directory_path, exist_ok=True)

    # 复制文件
    shutil.copy(source_file_path, dest_file_path)
    print(f"文件已从 {source_file_path} 复制到 {dest_file_path}")
    return

# 示例使用
# source_file = '/path/to/source/file.txt'
# destination_dir = '/path/to/destination/directory'
# copy_file_to_directory(source_file, destination_dir)

def run_script_and_monitor_file(bash_path, file_path, check_interval=1, stable_duration=5,save_path='',):
    save_folder, file_name = os.path.dirname(save_path) , os.path.basename(save_path)
    if file_name in os.listdir(save_folder):
        return

    # 在后台运行脚本
    exec_code = os.system("bash {}".format(bash_path))
    # 等待文件出现
    while not os.path.exists(file_path):
        time.sleep(check_interval)
    print('already have cnf , but size = 0...')
    # 等待文件大小大于0
    while os.path.getsize(file_path) == 0:
        time.sleep(check_interval)
    print('prepare to save...')
    # 检测文件大小是否稳定
    last_size = os.path.getsize(file_path)
    stable_time_start = None
    while True:
        time.sleep(check_interval)
        current_size = os.path.getsize(file_path)
        if current_size == last_size:
            if stable_time_start is None:
                stable_time_start = time.time()
            elif time.time() - stable_time_start >= stable_duration:
                # 文件大小稳定，杀死进程
                try:
                    result = subprocess.run(['pkill', '-f', 'run_picat'], check=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True)
                    print('kill...s')
                except Exception as e:
                    print(f"wrong when killing procession: {e}")
                break
        else:
            stable_time_start = None
        last_size = current_size

    # copy 并删除
    save_cnf_path = save_path
    copy_file_to_directory(file_path, save_cnf_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 {file_path} 已被删除。")
        print('新一轮准备开始...')
    return

def rewrite_picat(script_path, template_path, replace_codes_dict):
    # Fill in the actual parameters into the template file.
    with open(template_path, 'r') as file:
        template_content = file.read()

    new_content = template_content
    for k , v in replace_codes_dict.items():
        new_content = new_content.replace(k, str(v))

    with open(script_path, 'w') as file:
        file.write(new_content)
    return


if __name__ == '__main__':
    target_str_N = '{*-N-*}'
    target_str_C = '{*-C-*}'

    save_folder = './CoinsGrid_cnf'
    cnf_file_name = 'CoinsGrid_N{}_C{}.cnf'
    template_path = 'CoinsGrid_template.pi'
    method_name = template_path.replace('./','').replace('_template.pi','')
    bash_path = 'run_picat.sh'
    tmp_pi = 'CoinsGrid_.pi' # correspond with file in bash path

    os.makedirs(save_folder, exist_ok=True)

    # 4.17 more data
    # N = 31
    # Coins = 14

    # N = 300
    # Coins = 125
    #
    # cnf_save_path = os.path.join( save_folder, cnf_file_name.format(N, Coins))
    # rewrite_picat(script_path=tmp_pi, template_path=template_path,
    #               replace_codes_dict={target_str_N: N, target_str_C: Coins})
    # run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
    # print('over ... next one')
    # assert False
    train = True
    # train set
    if train:
        save_folder = './CoinsGrid_train'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        for N in (32,35,40,43,46): # (50,60,75,100,120,135,150,175,200): # 60的已经解不出来了... (450,600,850,1150,1350,1550,1750):
            for _ in range(3):
                Coins = int( (0.32 + np.random.rand() / 8 ) * N )
                try:
                    cnf_save_path = os.path.join(save_folder, cnf_file_name.format(N, Coins ) )
                    rewrite_picat(script_path=tmp_pi, template_path=template_path, replace_codes_dict={target_str_N:N,target_str_C:Coins })
                    run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
                    print('over ... next one')
                except:
                    pass
                time.sleep(10)

        save_folder = './CoinsGrid_cnf'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        for N in (45,48,56,64): # (66,80,90,125,145,160,180,200,240,260,280,300,320,360,400): # (450,600,850,1150,1350,1550,1750):
            for _ in range(3):
                Coins = int( (0.32 + np.random.rand() / 8 ) * N )
                try:
                    cnf_save_path = os.path.join(save_folder, cnf_file_name.format(N, Coins ) )
                    rewrite_picat(script_path=tmp_pi, template_path=template_path, replace_codes_dict={target_str_N:N,target_str_C:Coins })
                    run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
                    print('over ... next one')
                except:
                    pass
                time.sleep(10)

    else:
        for M in (1250,1500,1600,2000,2500): # 200, 300 ,500,750,800, 900 ,1000, 1200
            for ratio in (0.5,0.8,1,1.2,1.5,2):
                N = int(M * ratio)
                K_ratio = 0.3 + np.random.rand() / 10
                K = int(K_ratio * M * N)
                p = random.choice([0.32, 0.35, 0.38])
                print(cnf_file_name.format(M, N, K, int(round(p, 2) * 100)))
                try:
                    ret_str = make_MineSweeper_instance(grid_size=(M, N), K_Hints=K, p=p)
                    cnf_save_path = os.path.join(save_folder, cnf_file_name.format(M, N, K, int(round(p, 2) * 100)))

                    rewrite_picat(script_path=tmp_pi, template_path=template_path, replace_codes_dict={target_str: ret_str})
                    run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
                    print('over ... next one')
                except:
                    pass
                time.sleep(10)

    # ret_str = make_MineSweeper_instance(grid_size=(M, N), K_Hints=K, p=p)
    # cnf_save_path = os.path.join(save_folder, cnf_file_name.format(M, N, K, int(round(p, 2) * 100)))
    #
    # rewrite_picat(script_path=tmp_pi , template_path=template_path, replace_codes_dict={target_str:ret_str})
    # run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
    # print('over ... next one')

    # mappingD2U = {45: [30, 40, 45, 50], 49: [35, 40, 45, 50], 55: [15, 30, 40, 50, 55, 60, 70],
    #               65: [30, 40, 50, 55, 60, 70], 70: [35, 50, 65, 70, 80], 99: [45, 55, 65, 75, 80, 100, 120],
    #               125: [60, 90, 100, 10], 150: [300, 200, 125]}
    # for D in mappingD2U.keys():
    #     for MU in mappingD2U[D]:
    #         for u in range(-2, 3):
    #             U = MU + u
    #             # 从模板中读，写入 script 里
    #             rewrite_picat(script_path='./PRP.pi', template_path='./PRP_template.pi', params=params)
    #             run_script_and_monitor_file('run_PRP.sh', file_path='__tmp.cnf', params=params)
    #             print('over ... next one')

