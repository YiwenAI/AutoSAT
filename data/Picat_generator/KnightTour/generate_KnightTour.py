# KnightTour
import random
import numpy as np
import subprocess
import time
import shutil
import os



# 马走日，走满 N * N , 是否有解
# 无解时，需要遍历，可能会很难找

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
    target_str = '{*-K-*}'
    save_folder = './KnightTour_cnf'
    cnf_file_name = 'KnightTour_K{}.cnf'
    template_path = 'KnightTour_template.pi'
    method_name = template_path.replace('./','').replace('_template.pi','')
    bash_path = 'run_picat.sh'
    tmp_pi = 'KnightTour_.pi' # correspond with file in bash path

    os.makedirs(save_folder, exist_ok=True)

    # # 4.17 more data
    # for K in (20,8,32,40,75,126):
    #     print('here')
    #     cnf_save_path = os.path.join(save_folder, cnf_file_name.format(K))
    #     rewrite_picat(script_path=tmp_pi, template_path=template_path,
    #                   replace_codes_dict={target_str: K})
    #     run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
    #     print('over ... next one')

    train = True
    train_num = 24
    test_num = 36
    # all_ = random.sample(range(6, 70), train_num + test_num)
    all_ = list(range(6,54,2))
    # train set
    if train:
        save_folder = './KnightTour_train'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        for _ in range(train_num):
            K = all_[_]
            try:
                cnf_save_path = os.path.join(save_folder, cnf_file_name.format(K))
                rewrite_picat(script_path=tmp_pi, template_path=template_path,
                              replace_codes_dict={target_str: K})
                run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
                time.sleep(10)
            except:
                print('sth wrong')
        save_folder = './KnightTour_cnf'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        for _ in range(test_num):
            try:
                K = all_[train_num + _]
                cnf_save_path = os.path.join(save_folder, cnf_file_name.format(K))
                rewrite_picat(script_path=tmp_pi, template_path=template_path,
                              replace_codes_dict={target_str: K})
                run_script_and_monitor_file(bash_path=bash_path, file_path='__tmp.cnf', save_path=cnf_save_path)
                time.sleep(10)
            except:
                print('wrong when generate test...')



