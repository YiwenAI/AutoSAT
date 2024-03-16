from jinja2 import FileSystemLoader, Environment
import os
import re
import glob
import shutil
import platform
import subprocess


def get_code(answer, seperator):
    start = answer.find(seperator[0]) + len(seperator[0])
    end = answer.find(seperator[1], start) # - len(seperator[1])
    return answer[start:end]


def get_batch_id(count, batch_size):
    return (count-1) % batch_size

def revise_file(file_name, save_dir, *args, **kwargs):
    env = Environment(loader=FileSystemLoader('.'))
    try:
        template = env.get_template(file_name)
    except:
        raise FileNotFoundError(f'''file path '{file_name}' is not correct. 
                                ATTENTION, please ENSURE :                          
                                (1) `SAT_solver_file_path` is located within the current working directory, `./` 
                                (2) `SAT_solver_file_path` should be a relative path. ''')
    output = template.render(*args, **kwargs)
    with open(save_dir, 'w') as f:
        f.write(output)


def clean_files(folder_path, mode, *args, **kwargs):
    if mode == "all":
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    elif mode == "exe":
        for file_path in glob.glob(os.path.join(folder_path, "*.exe")):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    elif mode == "folder":
        pass
    else:
        raise NotImplemented


def process_raw_results(folder_path, timeout, answers=None):
    result = {
        "time": {},
        "prompt": {},
        "PAR-2": {},
        "satisfiable": {},
        "unsatisfiable": {},
        "timeout": {},
    }
    record_all_data = [] if answers is None else None
    for filename in os.listdir(folder_path):
        match = re.match(r'(\d+)_(\d+).txt', filename)
        if match:
            id, num = match.groups()
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                tmp_total_time = 0
                tmp_situation = {"satisfiable": 0,
                                 "unsatisfiable": 0,
                                 "timeout": 0}
                tmp_par2 = 0
                with open(file_path, 'r') as file:
                    for line in file.readlines():
                        line = line.strip().strip('\n').strip()
                        if line.startswith('File name'):
                            continue
                        parts = line.split('\t')
                        duration = int(parts[1])
                        situation_single = parts[2].lower()
                        tmp_situation[situation_single] += 1
                        tmp_total_time += duration
                        tmp_par2 += duration if duration < timeout else 2*timeout
                        if record_all_data is not None:
                            cnf_file_name = parts[0]
                            record_all_data.append((cnf_file_name, duration, situation_single))
                # finish reading the file, load temp results
                if id in result["time"]:
                    result["time"][id] += tmp_total_time
                    result["PAR-2"][id] += tmp_par2
                    for situation_key in tmp_situation:
                        result[situation_key][id] += tmp_situation[situation_key]
                else:
                    result["time"][id] = tmp_total_time
                    result["PAR-2"][id] = tmp_par2
                    result["prompt"][id] = answers[int(id)] if answers else 'Evaluation Stage.'
                    for situation_key in tmp_situation:
                        result[situation_key][id] = tmp_situation[situation_key]
    if answers is not None: # train
        return result
    else: # eval
        result['total time'] = result.pop('time')
        result.pop('prompt')
        result_dict = {k: v['1'] for k, v in result.items()}
        result_dict['#question'] = result_dict['satisfiable'] + result_dict['unsatisfiable'] + result_dict['timeout']
        result_dict['PAR-2'] = round(result_dict['PAR-2'] / result_dict['#question'] , 2)
        return result_dict, record_all_data


def collect_results(answers, repetition_dict, results, args):
    repetition_result = {
        "time": {},
        "prompt": {},
        "PAR-2": {},
        "satisfiable": {},
        "unsatisfiable": {},
        "timeout": {},
    }
    folder_path = './temp/results/'
    result = process_raw_results(folder_path=folder_path, timeout=args.timeout, answers=answers)
    if args.devoid_duplication:
        for value in list(repetition_dict.values()):
            key = find_key_for_value(results["prompt"], value)
            if key == None:
                break
            repetition_result["time"][key] = results["time"][key]
            repetition_result["prompt"][key] = results["prompt"][key]
        # repetition_result = {key: results["time"][key] for key in repetition_list if key in results["time"]}

        result["time"].update(repetition_result["time"])
        result["prompt"].update(repetition_result["prompt"])

    best_key = min(result["time"], key=result["time"].get)
    return result, {best_key: [result["time"][best_key], result["prompt"][best_key], result["PAR-2"][best_key]]}


def collect_results_eval(raw_path, final_path, args):
    folder_path = raw_path
    result_dict, record_all_data = process_raw_results(folder_path=folder_path, timeout=args.eval_timeout, answers=None)  # eval mode
    with open(final_path, 'a+', encoding='utf-8') as f:
        f.write("cnf File \t Duration \t Situation \n")
        for cnf_name, duration, situation in record_all_data:
            f.write(f"{cnf_name}\t{duration}\t{situation}\n")
        f.write(str(result_dict) + '\n')
    return result_dict


def fill_core_codes(origin_file, target_file, answer_code,**kwargs):
    revise_file(file_name=origin_file,
                save_dir=target_file,
                timeout='{{ timeout }}',
                data_dir='{{ data_dir }}',
                replace_code=answer_code,
                **kwargs)
    return


def delete_InfiniteLoopInst(candidates, result_dict, results_folder='./temp/results/'):
    failed_id_list = []
    for file_name in candidates:
        if not os.path.isfile(os.path.join(results_folder, file_name)):  # failed
            id_str = file_name.replace('finished', '').split('_')[0]
            for key in result_dict:
                if id_str in result_dict[key]:
                    result_dict[key].pop(id_str)
                    failed_id_list.append(id_str)
    # kill the procession. Maybe dangerous.
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(['taskkill', '/F', '/IM', 'EasySAT'], check=True, text=True)  # TODO check
        except:
            pass
        pass
    elif platform.system() == 'Linux':
        try:
            result = subprocess.run(['pkill', '-f', 'EasySAT'], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        except Exception as e:
            print(f"wrong when killing procession: {e}")
    else:
        raise NotImplementedError('sorry, we only support Wins Or Linux.')

    return


def copy_folder(src_folder, num, mode='train', target_folder = None):
    if mode == 'train':
        for i in range(num):
            new_folder_path = src_folder[:-1] + "_{}/".format(i)
            if os.path.exists(new_folder_path):
                shutil.rmtree(new_folder_path)
            shutil.copytree(src_folder, new_folder_path)
    elif mode == 'eval':
        if target_folder is None:
            raise ValueError('please set target folder to save source files.')
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        shutil.copytree(src_folder, target_folder)
    else:
        raise NotImplementedError('please choose `mode` between `train` or `eval`.')


def find_key_for_value(results, value_to_find):
    for key, value in results.items():
        if value == value_to_find:
            return key
    return None


def train_init(args):
    if os.path.exists("./temp/results/"):
        clean_files(folder_path="./temp/results/", mode="all")
    else:
        os.makedirs("./temp/results/")
    os.makedirs("./temp/prompts/",exist_ok=True)
    os.makedirs("./temp/prompts/", exist_ok=True)
    copy_folder('./examples/EasySAT/original_EasySAT', args.batch_size, mode='eval', target_folder="./temp/EasySAT/")
    clean_files(folder_path="./temp/EasySAT/", mode="exe")
    copy_folder("./temp/EasySAT/", args.batch_size)
    return

def check_reIteration(round, best_result_dict, baseline):
    # True: restart the prompt to avoid terrible functions; False: no need to restart
    if round != 1: return False
    best_results = next(iter(best_result_dict.values()))
    if best_results[0] < baseline['time'] or best_results[2] < baseline['PAR-2']:
        return False
    return True



if __name__ == "__main__":
    a = {'1': ['950\n', 'else if (conflicts % 1000 == 0 && fast_lbd_sum / lbd_queue_size > 5) restart();']}
    value = list(a.values())[0][1]
    print(value)
