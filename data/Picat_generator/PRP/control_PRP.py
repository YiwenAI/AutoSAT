import subprocess
import time
import shutil
import os


def copy_file_to_directory(source_file_path, dest_file_path):
    """
    将一个文件从源路径复制到目标目录。

    :param source_file_path: 源文件的完整路径。
    :param destination_directory_path: 目标目录的路径。
    """
    # 确保目标目录存在
    destination_directory_path = os.path.dirname(dest_file_path)
    if not os.path.exists(destination_directory_path):
        os.makedirs(destination_directory_path,exist_ok=True)

    # 复制文件
    shutil.copy(source_file_path, dest_file_path)
    print(f"文件已从 {source_file_path} 复制到 {dest_file_path}")
    return

def run_script_and_monitor_file(script_path, file_path, check_interval=1, stable_duration=5,params={'save_folder':'./','D':10,'U':10} ):
    if 'PRP_D{}_U{}.cnf'.format( params['D'] , params['U'])  in os.listdir( params['save_folder']):
        return
    exec_code = os.system("bash {}".format(script_path) )
    while not os.path.exists(file_path):
        time.sleep(check_interval)
    print('already have cnf , but size = 0...')
    while os.path.getsize(file_path) == 0:
        time.sleep(check_interval)
    print('prepare to save...')
    last_size = os.path.getsize(file_path)
    stable_time_start = None
    while True:
        time.sleep(check_interval)
        current_size = os.path.getsize(file_path)
        if current_size == last_size:
            if stable_time_start is None:
                stable_time_start = time.time()
            elif time.time() - stable_time_start >= stable_duration:
                try:
                    result = subprocess.run(['pkill', '-f', 'run_PRP'], check=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True)
                    print('kill...s')
                except Exception as e:
                    print(f"wrong when killing procession: {e}")
                break
        else:
            stable_time_start = None
        last_size = current_size

    save_cnf_path = os.path.join( params['save_folder'] , 'PRP_D{}_U{}.cnf'.format( params['D'] , params['U']) )
    copy_file_to_directory(file_path, save_cnf_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 {file_path} 已被删除。")
        print('新一轮准备开始...')


def rewrite_picat(script_path, template_path, params):
    # Fill in the actual parameters into the template file.
    with open(template_path , 'r') as file:
        template_content = file.read()

    new_content = template_content.replace("{*-DAY-*}", str(params['D']))
    new_content = new_content.replace("{*-U-*}", str(params['U']))
    with open(script_path, 'w') as file:
        file.write(new_content)
    return


if __name__ == '__main__':
    params = {'save_folder':'./', 'D': 10, 'U': 10}
    os.makedirs(params['save_folder'],exist_ok=True)

    mappingD2U = {45:[30,40,45,50],49:[35,40,45,50],55:[15,30,40,50,55,60,70],65:[30,40,50,55,60,70],70:[35,50,65,70,80],99:[45,55,65,75,80,100,120],125:[60,90,100,10],150:[300,200,125]}
    for D in mappingD2U.keys():
        for MU in mappingD2U[D]:
            for u in range(-2,3):
                U = MU + u
                # 从模板中读，从模板中写
                params['D'] = D
                params['U'] = U
                rewrite_picat(script_path='./PRP.pi', template_path='./PRP_template.pi', params=params)
                run_script_and_monitor_file('run_PRP.sh', file_path='__tmp.cnf', params=params)
                print('over ... next one')