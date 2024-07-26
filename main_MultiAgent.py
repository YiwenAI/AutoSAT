import os
import argparse
import json
import time
import yaml
import ray
import random
from jinja2 import Template, FileSystemLoader, Environment

from autosat.utils import revise_file, clean_files, collect_results, get_code, copy_folder, fill_core_codes, \
    delete_InfiniteLoopInst, get_batch_id, train_init, check_reIteration, \
    decodeRawJsonAnswer, sanitize_filename
from autosat.llm_api.base_api import GPTCallAPI, LocalCallAPI, fastllm
from autosat.execution.execution_worker import ExecutionWorker
from autosat.evaluation.evaluate import evaluate
import warnings


@ray.remote
def synchronized_asked(coder_args, evaluator_args, global_id, args):
    # Coder first , then Evaluator
    if args.llm_model[0:3] == "gpt":
        llm_api = GPTCallAPI(api_base=args.api_base,
                             api_key=args.api_key,
                             model_name=args.llm_model,
                             stream=False)
    elif args.llm_model == 'Qwen':
        # * you need
        llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                               api_key="sk-",
                               model_name="modelscope/qwen/Qwen-72B-Chat")
    elif args.llm_model == 'llama':
        llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                               api_key="sk-",
                               model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    elif args.llm_model == 'deepseek':
        llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                               api_key="sk-",
                               model_name="modelscope/deepseek-ai/deepseek-coder-33b-instruct")
    else:
        # write your own API call HERE.
        raise NotImplementedError

    # *------------- Coder START --------------------*
    if len(coder_args['_directions']) > 0:
        random.shuffle(coder_args['modification_direction'])
        coder_args['modification_direction'] = "Here are some potential improvement directions: \n    " + "\n    ".join(
            coder_args['_directions'][0:2])
    else:
        coder_args['modification_direction'] = ''

    coder_prompt = get_promptFromArgs(coder_args)
    coder_result = llm_api.call_api_prompt(prompt=coder_prompt, temperature=args.temperature)

    llm_generation = get_code(coder_result, seperator=['// start\n', '\n// end'])
    extra_params = {}  # record extra_var's require modifying at the same time
    for extra_var, default_value in coder_args.get('extra_var_dict', {}).items():
        # TODO We implement this by passing params through `coder_args['extra_var_dict']`. There might be better way.
        extra_var_value = get_code(coder_result, seperator=[f'// start {extra_var}\n', f'\n// end {extra_var}'])
        extra_params[extra_var] = extra_var_value if extra_var_value else str(default_value)

    if len(llm_generation) == 0:  # nothing generated
        return global_id, '', {}

    if 'evaluator' not in args.agent_type:  # evaluator unnecessary.
        return global_id, llm_generation, extra_params
    else:
        # *------------- Evaluator START --------------------*
        evaluator_args['llm_generation'] = llm_generation
        evaluator_prompt = get_promptFromArgs(evaluator_args)
        temperature_now = random.uniform(0.2, max(args.temperature, 0.2))
        evaluator_result = llm_api.call_api_prompt(prompt=evaluator_prompt, temperature=temperature_now)

        evaluator_feedback = decodeRawJsonAnswer(evaluator_result)
        print(evaluator_feedback)
        cls_type = evaluator_feedback.get('type', '')
        if cls_type == 'No modification':
            # give coder another Change, re-write.
            temperature_now = random.uniform(0.2, max(args.temperature, 0.2))
            coder_result = llm_api.call_api_prompt(prompt=coder_prompt, temperature=temperature_now)
            llm_generation = get_code(coder_result, seperator=['// start\n', '\n// end'])
            for extra_var, default_value in coder_args.get('extra_var_dict', {}).items():
                extra_var_value = get_code(coder_result, seperator=[f'// start {extra_var}\n', f'\n// end {extra_var}'])
                extra_params[extra_var] = extra_var_value if extra_var_value else str(default_value)
            evaluator_args['llm_generation'] = llm_generation
            evaluator_prompt = get_promptFromArgs(evaluator_args)
            temperature_now = random.uniform(0.2, max(args.temperature, 0.2))
            evaluator_result = llm_api.call_api_prompt(prompt=evaluator_prompt, temperature=temperature_now)
            evaluator_feedback = decodeRawJsonAnswer(evaluator_result)

        extra_params['-extra_analysis'] = evaluator_feedback.get('extra_analysis', '')
        answer_code = llm_generation
        return global_id, answer_code, extra_params


@ray.remote
def synchronized_executed(global_id, results, arguments, answer_code, **kwargs):
    project_dir = os.path.join(arguments.project, arguments.task)
    execution_worker = ExecutionWorker()

    if arguments.devoid_duplication and (answer_code in results["prompt"].values()):
        return global_id, 0, answer_code
    else:
        revise_file(file_name=os.path.join("./examples/", project_dir, "EasySAT.cpp"),
                    save_dir='./temp/EasySAT_{}/EasySAT.cpp'.format(format((global_id - 1) % arguments.batch_size)),
                    replace_code=answer_code,
                    timeout=arguments.timeout,
                    data_dir="\"{}\"".format(arguments.data_dir),
                    **kwargs,
                    )
        success = execution_worker.execute(global_id, arguments.batch_size, arguments.data_parallel_size)
        return global_id, success, answer_code


def agents_init(init_files_folder):
    # read `json files` and get agent_args to initialize
    with open(os.path.join(init_files_folder, 'advisor_args.json'), 'r') as f:
        advisor_args = json.load(f)
    with open(os.path.join(init_files_folder, 'coder_args.json'), 'r') as f:
        coder_args = json.load(f)
    with open(os.path.join(init_files_folder, 'evaluator_args.json'), 'r') as f:
        evaluator_args = json.load(f)
    assert coder_args['origin_target_code'] == advisor_args['origin_target_code']
    # set PROMPT template `advisor.txt' , 'evaluator.txt'
    advisor_args['prompt_fpath'] = os.path.join(init_files_folder, 'advisor.txt')
    evaluator_args['prompt_fpath'] = os.path.join(init_files_folder, 'evaluator.txt')
    return advisor_args, coder_args, evaluator_args


def resultsUpdatePerIteration(results, result, extra_params_all, advisor_record):
    # result
    results["time"].update(result["time"])
    results["PAR-2"].update(result["PAR-2"])
    results["prompt"].update(result["prompt"])
    results["timeout"].update(result["timeout"])
    results["extra_params"].update({str(k): v for k, v in extra_params_all.items()})
    for current_global_id in result["time"].keys():
        for key, value in advisor_record.items():
            if key not in results:
                results[key] = {current_global_id: value}
            else:
                results[key][current_global_id] = value
    return


def get_promptFromArgs(role_args):
    with open(role_args['prompt_fpath'], 'r', encoding='utf-8') as f:
        template_str_advisor = f.read()
    template = Template(template_str_advisor)
    prompt_str = template.render(**role_args)
    return prompt_str


def ask_advisor(advisor_args, args):
    advisor_prompt = get_promptFromArgs(advisor_args)
    llm_answer = fastllm(advisor_prompt, args)  # str
    advisor_feedback = decodeRawJsonAnswer(llm_answer)  # dict
    return advisor_feedback


def main(args):
    from datetime import datetime
    formatted_date_time = datetime.now().strftime("d%m_%d_h%Hm%M")  # time string: month,day,hour,minute

    data_dir = args.data_dir
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    data_num = len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
    if args.data_parallel_size > data_num:
        warnings.warn(f"The parallel num for training is too large: {args.data_parallel_size} > {data_num}. "
                      f"It will be replaced with the train set total num: {data_num}",
                      category=UserWarning, stacklevel=2)
        setattr(args, 'data_parallel_size', data_num)
    execution_worker = ExecutionWorker()

    answers = {}  # answers from llm. (generated codes)
    extra_params_all = {}  # other params outside the target heuristic parts.  e.g. restart condition `lbd_queue_size`
    count = 0  # global_id
    results = {
        "time": {},
        "prompt": {},
        "PAR-2": {},
        "timeout": {},
        "extra_params": {},
    }

    project_dir = os.path.join(args.project, args.task)
    print("project_dir: {}".format(project_dir))

    # initialize workspace
    train_init(args)
    advisor_args, coder_args, evaluator_args = agents_init(init_files_folder=args.agent_args_folder)

    if args.original:  # eval backbone SAT Solver
        env = Environment(loader=FileSystemLoader(os.path.join("./examples/", project_dir)))
        template = env.get_template("EasySAT_original.cpp")
        output = template.render(timeout=args.timeout, data_dir="\"{}\"".format(args.data_dir))
        with open("./temp/EasySAT_{}/EasySAT.cpp".format(count), 'w') as f:
            f.write(output)

        success = execution_worker.execute_original(count, args.data_parallel_size)
        assert (count == 0)
        filenames = [str(count) + "_" + str(num) + ".txt" for num in range(args.data_parallel_size)]
        start_time = time.time()
        while True:
            end_time = time.time()
            if end_time - start_time > args.timeout * (2 * data_num / args.data_parallel_size + 30):
                raise ValueError("Infinite loop error!!!")
            all_exist = all(
                os.path.exists(os.path.join('./temp/results/', 'finished' + filename)) for filename in filenames)
            if all_exist:
                result, best_result = collect_results(answers={0: advisor_args['origin_target_code']},
                                                       repetition_dict={},
                                                       results={},
                                                       args=args)
                resultsUpdatePerIteration(results, result, {}, {})
                break
        print(results["time"]["0"])

    else:
        results["time"]["0"] = args.original_result['time']
        results["prompt"]["0"] = advisor_args['origin_target_code']
        results["PAR-2"]["0"] = args.original_result['PAR-2']

    with open(f'original_results_{dataset_name}.txt', 'a+', encoding='utf-8') as f:
        f.write("EasySAT(baseline) result-- time: {} seconds ; PAR-2: {}".format(results["time"]["0"],
                                                                                 results["PAR-2"]["0"]))
        f.write('\n')
    print(
        "Backbone(original) result -- time: {} seconds ; PAR-2: {}".format(results["time"]["0"], results["PAR-2"]["0"]))

    for i in range(args.iteration_num):
        # clean temp results
        clean_files(folder_path="./temp/results/", mode="all")
        id_list = []  # save potential successful `id`
        answer_code_cur_round = {}  # save llm generation code for `current iteration`
        extraInfo_cur_round = {}  # save extra params for `current iteration`

        # main train pipeline -- update advisor/coder/evaluator args
        if i == 0:
            # first Iter, update advisor
            coder_args["prompt_fpath"] = os.path.join(args.agent_args_folder, "coder_firstIter.txt")  # first_iteration
            if 'advisor' not in args.agent_type:
                advisor_result = {'description': coder_args.get('origin_description', "")}
            else:
                advisor_result = ask_advisor(advisor_args, args)  # dict: {'description':xx, modification_direction:xx}
            print(246, advisor_result)
            description = advisor_result.get('description', '')
            if len(description): coder_args['description'] = description
            if 'modification_direction' in advisor_result \
                    and isinstance(advisor_result['modification_direction'], list) \
                    and len(advisor_result['modification_direction']) > 0:
                coder_args['_directions'] = advisor_result['modification_direction']
            else:
                coder_args['_directions'] = []
            coder_args[
                "origin_result"] = f"execution time is {results['time']['0']} seconds. PAR-2 is {results['PAR-2']['0']}."

        elif check_reIteration(round=i, best_result_dict=best_result,
                               baseline={'time': results["time"]["0"], 'PAR-2': results["PAR-2"]["0"]}):
            # restart llm-searching if necessary , you can set your own rule by `check_reIteration`  in ./autosat/utils.py
            print('restart ... ')
            coder_args["prompt_fpath"] = os.path.join(args.agent_args_folder,
                                                      "coder_firstIter.txt")  # restart_iteration
            args.temperature = min(max(0.5, random.random() + 0.1), 0.8)

            if 'advisor' not in args.agent_type:
                advisor_result = {'description': coder_args.get('origin_description', "")}
            else:
                advisor_result = ask_advisor(advisor_args, args)
            description = advisor_result.get('description', '')
            if len(description): coder_args['description'] = description
            if 'modification_direction' in advisor_result \
                    and isinstance(advisor_result['modification_direction'], list) \
                    and len(advisor_result['modification_direction']) > 0:
                coder_args['_directions'] = advisor_result['modification_direction']
            else:
                coder_args['_directions'] = []
        else:
            # we ignore directions here to let llm guided by hints from Evaluators
            coder_args['_directions'] = []
            coder_args["prompt_fpath"] = os.path.join(args.agent_args_folder, "coder_WithFeedback.txt")
            experiment_results = [
                f"Experiment {cur_idx}, Your provided {args.project}--{args.task}: \n'''{result['prompt'][value[0]]}''', \n execution time is {result['time'][value[0]]} seconds. PAR-2 is {result['PAR-2'][value[0]]}. "
                for cur_idx, value in enumerate(result["time"].items())]
            print(experiment_results)
            best_code_id = next(iter(best_result.keys()))
            best_code_lastIter = results['prompt'][best_code_id]
            best_code_description = results['extra_params'][best_code_id].get('-extra_analysis',
                                                                              '')  # where is analysis ???

            coder_args["experiment_results"] = "\n ".join(experiment_results)
            coder_args["best_code"] = best_code_lastIter
            coder_args["best_code_description"] = best_code_description
        print('*------------------------------*')
        print(f'iteration {i}')
        print(advisor_result)
        print('*------------------------------*')
        start_time = time.time()
        query_llm_tasks = [
            synchronized_asked.remote(coder_args, evaluator_args, global_id=i * args.batch_size + batch_id + 1,
                                      args=args)
            for batch_id in range(args.batch_size)]
        for future in ray.get(query_llm_tasks):
            if isinstance(future, tuple):
                global_id, answer_code, extra_params = future
                batch_id = get_batch_id(global_id, args.batch_size)
                answer_code_cur_round[batch_id] = answer_code
                extraInfo_cur_round[batch_id] = extra_params

        end_time = time.time()
        print("querying consuming: {} seconds".format(end_time - start_time))

        tasks = [synchronized_executed.remote(
            global_id=i * args.batch_size + batch_id + 1,
            results=results, arguments=args,
            answer_code=answer_code_cur_round[batch_id],
            **extraInfo_cur_round[batch_id]) for batch_id in range(args.batch_size)]

        repetition_dict = {}
        for future in ray.get(tasks):
            global_id, success, answer_code = future
            batch_id = get_batch_id(global_id, args.batch_size)
            answers[global_id] = answer_code_cur_round[batch_id]  # global_id: int
            extra_params_all[str(global_id)] = extraInfo_cur_round[batch_id]  # To support more params
            if success:
                id_list.append(global_id)
            elif args.devoid_duplication and success == 0:
                repetition_dict[global_id] = answer_code

        print("start to run generated codes ...")
        start_time = time.time()
        filenames = [str(global_id) + "_" + str(num) + ".txt" for global_id in id_list for num in
                     range(args.data_parallel_size)]
        print("filenames: ", filenames)
        while True:
            end_time = time.time()
            if end_time - start_time > args.timeout * (2 * data_num / args.data_parallel_size + 30):
                warnings.warn(f": Infinite loop for some Solver Programs... please check later",
                              category=UserWarning, stacklevel=2)
                result, _ = collect_results(answers=answers,
                                            repetition_dict=repetition_dict,
                                            results=results,
                                            args=args)
                delete_InfiniteLoopInst(candidates=['finished' + fname for fname in filenames], result_dict=result)
                # re-construct best_result according to `valid result`
                best_key = min(result["PAR-2"], key=result["PAR-2"].get)
                best_result = {
                    best_key: [result["time"][best_key], result["prompt"][best_key], result["PAR-2"][best_key]]}
                break

            all_exist = all(
                os.path.exists(os.path.join('./temp/results/', 'finished' + filename)) for filename in filenames)
            if all_exist:
                result, best_result = collect_results(answers=answers,
                                                      repetition_dict=repetition_dict,
                                                      results=results,
                                                      args=args)
                break

        print("collecting execution time consuming: ", end_time - start_time)
        advisor_record = {"task_description": description,
                          "modification_direction": str(advisor_result.get('modification_direction', ''))}  # TODO
        resultsUpdatePerIteration(results, result, extra_params_all, advisor_record)  # update global Dict `results`
        json_save_path = f'./temp/prompts/iter_{i}_result_{dataset_name}_{formatted_date_time}.json'
        with open(json_save_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    ray.shutdown()
    final = {}
    for global_id_str in results["time"]:
        final[global_id_str] = {
            "time": results["time"][global_id_str],
            "PAR-2": results["PAR-2"][global_id_str],
            "prompt": results["prompt"][global_id_str],
            "extra_params": results["extra_params"].get(global_id_str, ''),  # * include '-analysis':code_analysis
            "task_description": results["task_description"].get(global_id_str, ''),
            "modification_direction": results["modification_direction"].get(global_id_str, ''),
        }

    valid_model_name = sanitize_filename(args.llm_model)  # Model name : use '_' to replace special char
    json_save_path = f'./temp/prompts/final_result_{dataset_name}_{valid_model_name}_{formatted_date_time}.json'
    with open(json_save_path, 'w') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    print('saved final prompt...')

    # *---------------  EVALUATION AFTER TRAINING --------------------------
    if not args.NeedEval: return
    print('start evaluation ...')
    if os.path.exists("./temp/results/"):
        clean_files(folder_path="./temp/results/", mode="all")
    baseline = results["PAR-2"]["0"]
    record_info = []
    print(extra_params_all)
    print('*------------------------------------------*')
    print(final)
    print('backbone as Baseline : {}'.format(baseline))
    # only eval those outperform baseline during searching stage.
    for global_id_str in final.keys():
        global_id = int(global_id_str)
        if global_id_str != "0":
            if final[global_id_str]["PAR-2"] < baseline:
                record_info.append((global_id, final[global_id_str]["PAR-2"], final[global_id_str]["prompt"],
                                    extra_params_all[global_id_str]))
    record_info.sort(key=lambda x: x[1])

    print("{} New SAT Solver to be evaluated...".format(len(record_info)))
    if len(record_info) == 0: return

    for global_id, par_2, answer_code, params_dict in record_info:  # global_id is just `count`
        method_name = f"{args.task}_{args.agent_type}_{args.llm_model}_{global_id}".replace('/', '')
        SAT_folder = f'./temp/EasySAT_{method_name}/'
        copy_folder(src_folder="./temp/EasySAT/", num=1, mode='eval', target_folder=SAT_folder)
        SAT_solver_file_path = os.path.join(SAT_folder, 'EasySAT_modified.cpp')
        # replace the answer code for the specific function and other auxiliary params such as `lbd size`
        fill_core_codes(origin_file=os.path.join("./examples/", project_dir, "EasySAT.cpp"),
                        target_file=SAT_solver_file_path,
                        answer_code=answer_code,
                        **params_dict)
        evaluate(args, method_name=method_name, SAT_solver_file_path=SAT_solver_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./examples/EasySAT/config.yaml', help='Path to the config file')
    parser.add_argument('--iteration_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_parallel_size', type=int, default=3)
    parser.add_argument('--devoid_duplication', type=bool, default=False)
    parser.add_argument('--llm_model',
                        type=str,
                        default="gpt-4-turbo",
                        choices=["gpt-4-turbo", "gpt-3.5-turbo", "Qwen", "llama", "deepseek"])
    parser.add_argument('--timeout', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--project', type=str, default="EasySAT")
    parser.add_argument('--task',
                        type=str,
                        default="bump_var_function",
                        choices=["restart_condition", "bump_var_function", "rephase_function"])
    parser.add_argument('--agent_args_folder', type=str, default="")
    parser.add_argument('--original', type=bool, default=False)
    parser.add_argument('--agent_type',
                        type=str,
                        default="advisor_evaluator_coder",
                        choices=['advisor-coder-evaluator', 'coder-evaluator', 'advisor-coder', 'coder-only'])
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--NeedEval', type=bool, default=True)
    parser.add_argument('--api_base', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')

    args = parser.parse_args()
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            for key, value in config.items():
                setattr(args, key, value)

    main(args)