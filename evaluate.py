# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import subprocess
from datetime import datetime
import time
import re
import yaml
import math
import warnings

from autosat.utils import revise_file, clean_files, collect_results_eval, copy_folder
from autosat.execution.execution_worker import ExecutionWorker


def evaluate(args, SAT_solver_file_path, method_name=None):
    formatted_date_time = datetime.now().strftime("d%m_%d_h%Hm%M")  # time string: month , day , hour , minute

    # # STEP 1. run SAT solver , get raw results in parallel.
    results_save_path_intermediate = os.path.join(args.results_save_path, 'tmp_{}'.format(
        formatted_date_time))  # path for saving the intermediate executable files and results.
    os.makedirs(results_save_path_intermediate, exist_ok=True)

    SAT_files_dir = os.path.dirname(SAT_solver_file_path)
    copy_folder(src_folder=SAT_files_dir, num=-1, mode='eval', target_folder=results_save_path_intermediate)

    tmp_cpp_source_path = os.path.join(results_save_path_intermediate, 'SAT_Solver_tmp.cpp')
    tmp_executable_file_path = os.path.join(results_save_path_intermediate, 'SAT_Solver_tmp')
    # change the file.
    revise_file(
        file_name=SAT_solver_file_path,
        save_dir=tmp_cpp_source_path,
        timeout=args.eval_timeout,
        data_dir="\"" + args.eval_data_dir + "\"",
    )

    cnf_duration_situation_fpath = './temp/results/'  # TODO intermediate files are locked in EasySAT.cpp ...
    if os.path.exists(cnf_duration_situation_fpath):
        clean_files(folder_path="./temp/results/", mode="all")
    else:
        os.makedirs(cnf_duration_situation_fpath)  # save results
    method_name = method_name if method_name else os.path.basename(SAT_solver_file_path).replace('.cpp', '')
    execution_worker = ExecutionWorker()
    success = execution_worker.execute_eval(source_cpp_path=tmp_cpp_source_path,
                                            executable_file_path=tmp_executable_file_path,
                                            data_parallel_size=args.eval_parallel_size)
    if not success:
        raise RuntimeError("cannot correctly execute... plz check again")

    eval_data_dir = args.eval_data_dir  # TODO change xxx
    filenames = [str(1) + "_" + str(num) + ".txt" for num in
                 range(args.eval_parallel_size)]  # set `id` = 1 during evaluation
    data_num = len([f for f in os.listdir(eval_data_dir) if os.path.isfile(os.path.join(eval_data_dir, f))])
    print("data_num:", data_num, "eval_parallel_sizes: ", args.eval_parallel_size)
    if args.eval_parallel_size > data_num:
        warnings.warn(f"The parallel num for training is too large: {args.eval_parallel_size} > {data_num}. "
                      f"It will be replaced with the train set total num: {data_num}",
                      category=UserWarning, stacklevel=2)
        setattr(args, 'eval_parallel_size', data_num)
    start_time = time.time()
    while True:
        end_time = time.time()
        if end_time - start_time > args.eval_timeout * 1.5 * math.ceil(data_num / args.eval_parallel_size) + 10:
            raise RuntimeError("Infinite loop error!!!")
        all_exist = all(
            os.path.exists(os.path.join('./temp/results/', 'finished' + filename)) for filename in filenames)
        if all_exist:
            break
    if not all_exist:
        raise ValueError("sth. wrong during evaluation")

    print('SAT Solver finished...')
    # STEP 2. collect results.
    result_dict = collect_results_eval(raw_path=cnf_duration_situation_fpath,
                                       final_path=os.path.join(args.results_save_path,
                                                               'results_{}_{}.txt'.format(method_name,
                                                                                          formatted_date_time)),
                                       args=args)
    print(f'results are saved in {args.results_save_path} ...')

    # STEP 3. remove ...
    if not args.keep_intermediate_results:
        try:
            shutil.rmtree(cnf_duration_situation_fpath)
            shutil.rmtree(results_save_path_intermediate)
        except:
            warnings.warn("Wrong when remove the temporary files...You can manually delete the folders.{},{}".format(
                cnf_duration_situation_fpath, results_save_path_intermediate),
                          category=UserWarning, stacklevel=2)
    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='./examples/EasySAT/eval_config.yaml',
                        help='Path to the config file')
    parser.add_argument('--SAT_solver_file_path', type=str, default='./template/EasySAT_eval/EasySAT_template.cpp',
                        help='SAT solver file path (NOTICE: auxiliary functions should be in the same directory).')
    parser.add_argument('--eval_data_dir', default='./evaluation/', type=str,
                        help='the directory where cnf files are stored.')
    parser.add_argument('--results_save_path', type=str, default='./temp/eval_results/',
                        help='where the final result are saved.')
    parser.add_argument('--eval_parallel_size', type=int, default=16, help='parallel in K processions.')
    parser.add_argument('--eval_timeout', type=int, default=1500, help='time-out for SAT Solver')
    parser.add_argument('--rand_seed', type=int, default=42, help='random seed')
    parser.add_argument('--keep_intermediate_results', type=bool, default=True,
                        help='whether to keep intermediate results.')
    parser.add_argument('--method_name', type=str, default=None, help='character or name for the SAT Solver')

    args = parser.parse_args()

    if os.path.exists(args.config):
        print('eval config .. get here...')
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            for key, value in config.items():
                setattr(args, key, value)
                print(key, value)

    print('eval tiemout: {} , eval SAT_solver path: {}'.format(args.eval_timeout, args.SAT_solver_file_path) )
    evaluate(args, method_name=args.method_name, SAT_solver_file_path=args.SAT_solver_file_path)
