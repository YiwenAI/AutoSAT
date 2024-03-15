# AutoSAT

Welcome to the official repository for the paper ["AutoSAT: Automatically Optimize SAT Solvers via Large Language Models"](https://arxiv.org/abs/2402.10705).
This repository is dedicated to automatically optimize heuristics in SAT solvers through Large Language Models (LLMs). 
We hope our method can be universally applied to a variety of solvers and welcome researchers from all backgrounds to join us in this endeavor!


## Clone this repo

`git clone https://github.com/YiwenAI/AutoSAT`



## Installation
We support both **Linux** and **Windows**  
1. Python 3.10 
2. G++ 17 or higher to support `filesystem`

Install requirements:
`pip install -e .`

Install this package
`python setup.py develop`


## Use Docker

`docker build -t autosat .`


## Train&Test

### Train

```
python3 main_parallel.py \
        --iteration_num 4 \
        --batch_size 4 \
        --data_parallel_size 6 \
        --devoid_duplication False \
        --timeout 500 \
        --data_dir your_train_set_path \
        --project EasySAT \
        --task bump_var_function \
        --original True \
        --api_base your_api_base \
        --api_key your_api_key
```

> :bulb: **Tips**
>
> We recomend to use `python3 main.py --config your_config_file_path ` and pass the params by **.yaml** such as [config.yaml](./examples/EasySAT/config.yaml)
> 
> When `original` is set to `False`, you have to set **dict** `original_result` in your config.yaml
>
> Refer to [configs explanation](./examples/EasySAT/config_explanation.txt) for more details.
>
> The heuristic functions generated and the some metrics are save in the folder: './temp/prompts/' (final_result.json or iter_xx_result.json)
 

### Test
```
python3 evaluate.py \
        --SAT_solver_file_path SAT_solver_file_path \
        --results_save_path your_final_eval_results_savePath \
        --batch_size 4 \
        --eval_parallel_size 6 \
        --eval_timeout 500 \
        --eval_data_dir your_test_set_path \
        --rand_seed 42
        --keep_intermediate_results False \
        --method_name your_solver_name
```
> :bulb: **Tips**
>
> We recomend to use `python3 evaluate.py --config your_eval_config_file_path` and pass the params by **.yaml** such as [config_eval.yaml](./examples/EasySAT/config_eval.yaml)
>
> Refer to [configs explanation](./examples/EasySAT/config_explanation.txt) for more details.
>
> The final evaluation results are saved in the folder you set previously -- **results_save_path**

### Dataset
* AutoSAT adapts the following training set, a total of 48 SAT instances (37.5% from SAT Competition 2018, 62.5% from SAT Competition 2022). We also collect some specific SAT questions such as CNP , SCPC and PAR. All data are in [cnf_data](https://drive.google.com/drive/folders/1-au8hBbx4YAdJDlct9glCODpL0TQcYnA?usp=drive_link)

* We also provide the codes we used to generate the specific questions in [`./data/`](./data/), and the default data directory is in `./temp/data_train`
  
* Access more SAT Competition questions by visiting [SAT Competition](https://satcompetition.github.io/)   


## Metrics
We use the following metrics to evaluate the performance of a Solver.

  * PAR-2:  The Penalized Average Runtime with factor 2

  * #solved:  Number of questions the Solver solved within timeout

  * total time: Total running time for a Solver.
  
  * #satisfied: Number of getting feasible solution.
    
  * #unsatisfied: Number of solving unfeasible questions

  * #timeout: Number of timeout cases.


## Acknowledgement
Our baseline is [EasySAT](https://github.com/shaowei-cai-group/EasySAT) and we only add data parallel and file saving modules. Thanks for the wonderful work.

## Citing us

If our work has been helpful to you, please feel free to cite us:

```latex
@article{sun2024autosat,
  title={AutoSAT: Automatically Optimize SAT Solvers via Large Language Models},
  author={Sun, Yiwen and Zhang, Xianyin and Huang, Shiyu and Cai, Shaowei and Zhang, Bing-Zhen and Wei, Ke},
  journal={arXiv preprint arXiv:2402.10705},
  year={2024}
}
```
