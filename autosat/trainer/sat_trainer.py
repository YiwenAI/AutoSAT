import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from jinja2 import FileSystemLoader, Environment

from autosat.trainer.base_trainer import BaseTrainer
from autosat.utils import get_code, revise_file, clean_files, collect_results, copy_folder
from autosat.llm_api.base_api import GPTCallAPI, LocalCallAPI
from autosat.execution.execution_worker import ExecutionWorker


class SATTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.execution_worker = ExecutionWorker()
        data_dir = os.path.join("./temp", args.data_dir)
        self.data_num = len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])

        self.answers = {}
        self.count = 0
        self.results = {
            "time": {},
            "prompt": {},
            "PAR-2": {}
        }

        self.project_dir = os.path.join(args.project, args.task)

        clean_files(folder_path="./temp/results/", mode="all")
        clean_files(folder_path="./temp/EasySAT/", mode="exe")

        copy_folder("./temp/EasySAT/", args.batch_size)

    def original_solver_execute(self):
        args = self.args
        env = Environment(loader=FileSystemLoader(os.path.join("./examples/", self.project_dir)))

        template = env.get_template("EasySAT_original.cpp")
        output = template.render(timeout=args.timeout, data_dir="\"{}\"".format(args.data_dir))
        with open("./temp/EasySAT/EasySAT.cpp", 'w') as f:
            f.write(output)

        if args.original:
            success = self.execution_worker.execute_original(self.count, args.data_parallel_size)
            filenames = [str(self.count) + "_" + str(num) + ".txt" for num in range(args.data_parallel_size)]

            while True:
                all_exist = all(os.path.exists(os.path.join('./temp/results/', filename)) for filename in filenames)
                if all_exist:
                    file_list = [os.path.join('./temp/results/', filename) for filename in filenames]
                    for file_dir in file_list:
                        with open(file_dir, 'r') as file:
                            original_result = file.read()[:-1]
                            if self.results["time"] == {}:
                                self.results["time"]["0"] = int(original_result)
                            else:
                                self.results["time"]["0"] += int(original_result)

                    self.results["prompt"]["0"] = " "
                    break
        else:
            self.results["time"]["0"] = args.original_result  # 28707 16424
            self.results["prompt"]["0"] = " "
            self.results["PAR-2"]["0"] = 100
        print("original result: ", self.results["time"]["0"], " seconds")

    def synchronized_asked(self, prompt_file_dir, count):
        args = self.args

        if args.llm_model == "gpt-4-1106-preview":
            llm_api = GPTCallAPI(api_base=args.api_base,
                                 api_key=args.api_key,
                                 model_name=args.llm_model)
        elif args.llm_model == "gpt-3.5-turbo":
            llm_api = GPTCallAPI(api_base=args.api_base,
                                 api_key=args.api_key,
                                 model_name=args.llm_model)
        elif args.llm_model == 'Qwen':
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

        answer = llm_api.call_api(prompt_file=prompt_file_dir)

        answer_code = get_code(answer, seperator=['// start\n', '\n// end'])
        lbd_queue_size = get_code(answer, seperator=['// start lbd_queue_size\n', '\n// end lbd_queue_size'])

        return count, answer_code, lbd_queue_size

    def synchronized_executed(self, count, results, arguments, answer_code, *args, **kwargs):
        project_dir = os.path.join(arguments.project, arguments.task)
        execution_worker = ExecutionWorker()

        if arguments.devoid_duplication and (answer_code in results["prompt"].values()):
            return count, 0, answer_code
        else:
            revise_file(file_name=os.path.join("./examples/", project_dir, "EasySAT.cpp"),
                        save_dir='./temp/EasySAT_{}/EasySAT.cpp'.format(format((count - 1) % arguments.batch_size)),
                        replace_code=answer_code,
                        timeout=arguments.timeout,
                        data_dir="\"{}\"".format(arguments.data_dir),
                        *args, **kwargs
                        )
            success = execution_worker.execute(count, arguments.batch_size, arguments.data_parallel_size)
            return count, success, answer_code

    def train(self):
        args = self.args
        ex = ProcessPoolExecutor(max_workers=50)
        if args.original:
            self.original_solver_execute()
        else:
            self.results["time"]["0"] = args.original_result  # 28707 16424
            self.results["prompt"]["0"] = " "
            self.results["PAR-2"]["0"] = 100

        for i in range(args.iteration_num):
            # clean temp results
            clean_files(folder_path="./temp/results/", mode="all")
            id_list = []

            if i == 0:
                prompt_file_dir = os.path.join("./examples/", self.project_dir, "original_prompt.txt")
            else:
                result_prompt = [
                    f"Experiment {num}, Your provided judgement condition of restart function: \n'''{result['prompt'][value[0]]}''', \n execution time is {result['time'][value[0]]} seconds."
                    for num, value in enumerate(result["time"].items())]
                result_prompt = '\n '.join(result_prompt)
                print("iteration: ", i, "\n results_prompt: \n", result_prompt)

                # generate new prompt based on results in the previous iteration
                revise_file(file_name=os.path.join("./examples/", self.project_dir, "feedback_prompt.txt"),
                            save_dir='./temp/prompts/feedback_prompt.txt',
                            replace_code=result_prompt,
                            original_time=int(self.results["time"]["0"]),
                            best_code=list(best_result.values())[0][1]
                            )

                prompt_file_dir = './temp/prompts/feedback_prompt.txt'

            start_time = time.time()
            answer_code_list = []
            lbd_queue_size_list = []
            tasks = [ex.submit(self.synchronized_asked, prompt_file_dir, i * args.batch_size + count + 1) for count in
                     range(args.batch_size)]
            for future in as_completed(tasks):
                count, answer_code, lbd_queue_size = future.result()
                answer_code_list.append(answer_code)
                lbd_queue_size_list.append(lbd_queue_size)
            end_time = time.time()
            print("querying time consuming: ", end_time - start_time)

            start_time = time.time()

            if args.task == "restart_condition":
                tasks = [ex.submit(self.synchronized_executed,
                                   count=i * args.batch_size + count + 1,
                                   results=self.results, arguments=args,
                                   answer_code=answer_code_list[count],
                                   lbd_queue_size=lbd_queue_size[count]) for count in range(args.batch_size)]

            else:
                tasks = [ex.submit(self.synchronized_executed,
                                   count=i * args.batch_size + count + 1,
                                   results=self.results, arguments=args,
                                   answer_code=answer_code_list[count]) for count in range(args.batch_size)]

            repetition_dict = {}
            for future in as_completed(tasks):
                count, success, answer_code = future.result()
                self.answers[count] = answer_code
                if success:
                    id_list.append(count)
                elif args.devoid_duplication and success == 0:
                    repetition_dict[count] = answer_code
            end_time = time.time()
            print("sending execution time consuming: ", end_time - start_time)

            start_time = time.time()
            # whether task have been completed
            filenames = [str(id) + "_" + str(num) + ".txt" for id in id_list for num in range(args.data_parallel_size)]
            print("filenames", filenames)
            while True:
                end_time = time.time()
                if end_time - start_time > args.timeout * (2 * self.data_num / args.data_parallel_size):
                    raise ValueError("Infinite loop error!!!")
                all_exist = all(
                    os.path.exists(os.path.join('./temp/results/', 'finished' + filename)) for filename in filenames)
                if all_exist:
                    # collect results
                    result, best_result = collect_results(answers=self.answers,
                                                          repetition_dict=repetition_dict,
                                                          results=self.results,
                                                          args=args)
                    break
            print("collecting execution time consuming: ", end_time - start_time)
            # save intermediate results
            self.results["time"].update(result["time"])
            self.results["PAR-2"].update(result["PAR-2"])
            self.results["prompt"].update(result["prompt"])
            with open('./results/iter_{}_result.json'.format(i), 'w') as f:
                json.dump(result, f)

        final = {}
        for key in self.results["time"]:
            final[key] = {
                "time": self.results["time"][key],
                "PAR-2": self.results["PAR-2"][key],
                "prompt": self.results["prompt"][key],
            }

        # save final results
        with open('./results/final_result.json', 'w') as f:
            json.dump(final, f)
