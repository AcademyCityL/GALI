# copy from longbench pred.py and modified
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from models.DPI.dpi import get_model_and_tokenizer
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from models.DPI.help_funcs import wrap_with_model_special_input
import toml
from datetime import datetime
# from longbench_eval import *
from leval.Baselines import LEval_config 
import datasets
from typing import List, Optional, Tuple, Union
from eval import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def get_params(config_file_path):
    config = json.loads(json.dumps(toml.load(config_file_path)))
    return config 

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    return parser.parse_args(args)

def ensure_unique_folder(base_path, folder_name):
    new_folder_name = folder_name
    count = 1

    # 检查文件夹是否已存在，如果存在则添加后缀
    while os.path.exists(os.path.join(base_path, new_folder_name)):
        new_folder_name = f"{folder_name}_{count}"
        count += 1

    # os.makedirs(os.path.join(base_path, new_folder_name))
    return new_folder_name

# This is the customized building prompt for chat models
def build_chat_leval(metric, inst, document, out_path, model_name):
    sys_prompt = LEval_config.get_sys_prompt(metric, out_path)
    if "Llama-2" in model_name:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        if "gsm" in out_path or "codeU" in out_path:
            context = document + "\n\n" + inst
            message = B_INST + B_SYS + sys_prompt + E_SYS + context
        elif "topic" in out_path:
            context = document + "\n\n" + inst
            message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
        elif metric == "exam_eval":
            context = "Document is as follows. {document} \nQuestion: {inst}.  Please directly give the answer without any additional output or explanation "
            message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
            message += "\nAnswer:"
        else:
            pass
        try:
            context_inputs = message.format(document=document, inst=inst)
        except:
            context_inputs = message
    elif "Llama-3" in model_name:
        if "gsm" in out_path or "codeU" in out_path:
            context = document + "\n\n" + inst
            message = context
        elif "topic" in out_path:
            context = document + "\n\n" + inst
            message = context
        elif metric == "exam_eval":
            context = "Document is as follows. {document} \nQuestion: {inst}.  Please directly give the answer without any additional output or explanation "
            message = context
            message += "\nAnswer:"
        else:
            pass
        try:
            context_inputs = message.format(document=document, inst=inst)
        except:
            context_inputs = message
    return sys_prompt, context_inputs

def build_chat_lbench(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "Llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "Llama-3" in model_name:
        pass
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def generate_prompt_landmark_needle(n_garbage, seed, percent):
    """Generates a text file and inserts an passkey at a random position."""
    # rnd_state = random.get_state()
    # random.seed(seed)
    # n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_prefix = int(percent * n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 50000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(50000, 500000)

    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    # random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def needle_worker(rank, world_size, max_pe, out_path, times, min_k, max_k, gap, config):
    # code comes from chunkllama
    model_name = config['model']
    method = config['method']
    params = config['params']
    ori_max_position_embeddings = config.get('ori_max_position_embeddings', None)
    model, tokenizer = get_model_and_tokenizer(model_name, config, method, params, max_position_embeddings=ori_max_position_embeddings)
    seed = 42
    seed_everything(seed)
    device = torch.device(f'cuda:{rank}')
    model = model.eval().to(device)
    # hyper params
    k = 1000
    max_length = max_k * k # default 256k
    min_length = min_k * k # default 1k
    gap = gap * k # default 8k
    num_per = 16 # default 16
    depth_percent = 1 / num_per
    
    # length_list = [k] + [i for i in range(4*k, max_length + 1, gap)]
    length_list = [i for i in range(min_length, max_length + 1, gap)]
    test = config['test']
    if test == True:
        length_list = [length_list[0]]

    results = []
    if 'dpi' == method:
        output_attentions = True
    else:
        output_attentions = False
    for length in length_list:
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * length // 1024 * 1024)
 
        depths = [depth_percent * i for i in range(1, num_per + 1)]
        for depth in depths:
            passed_tests = 0
            all_accuries = {}
            for j in range(times): # default 10
                torch.cuda.empty_cache()

                prompt, answer = generate_prompt_landmark_needle(n_garbage, seed, depth)

                 # add by dpi
                prompt = build_chat_lbench(tokenizer, prompt, model_name)
                inputs = wrap_with_model_special_input(model_name, prompt, tokenizer, device, "passkey")

                len_token = inputs.input_ids.shape[-1]
            
                print("len tokens", len_token)

                answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
                generation_output = model.generate(
                    input_ids=inputs.input_ids, max_new_tokens=answer_ids.shape[-1]+ 100, num_beams=1, use_cache=True,  do_sample=False, temperature=1.0,
                    output_attentions = output_attentions,
                    return_dict_in_generate = True,
                )['sequences'][0]
                torch.cuda.empty_cache()
                model_answer = generation_output[len_token:].cpu()
                # model_answer = generation_output[-answer_ids.shape[-1]:].cpu()
                pred = tokenizer.decode(model_answer, skip_special_tokens=True)
                # model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
                # is_correct = (model_answer == answer_ids[0]).all().item()
                print(f"pred: {pred}, answer {answer}")
                if answer in pred:
                    passed_tests += 1
                    
                print("--------")

                # model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
                # is_correct = (model_answer == answer_ids[0]).all().item()
                # return is_correct, len_token

                # is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, depth, n_garbage=n_garbage, seed=j)
            accuracy = float(passed_tests) / times
            res = {"context_length": f"{length // k}k", "pred":pred, "answer":answer, "depth_percent": depth * 100, "score": accuracy}
            results.append(res)
            print(res)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)
                f.write('\n')
            # with open(out_path, "a") as f:
            #     print(json.dumps(res), file=f)
    print("results ",results)

def needle(config, world_size):
    # code comes from chunkllama
    exp_name = config['exp_name']
    max_pe = config['max_pe']
    times = config['times']
    min_k = config['min_k']
    max_k = config['max_k']
    gap = config['gap']
    results_root = "results/needle/"
    exp_base_path = os.path.join(results_root, 'pred')
    if not os.path.exists(exp_base_path):
        os.makedirs(exp_base_path)

    base_folder_name = f"{exp_name}"
    exp_folder_name = ensure_unique_folder(exp_base_path, base_folder_name)
    exp_dir = os.path.join(exp_base_path, exp_folder_name)
    os.makedirs(exp_dir)

    print(f"Your prediction file will be saved to: {exp_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}.json"

    file_path = os.path.join(exp_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(config, json_file, ensure_ascii=False, indent=4)

    processes = []
    out_path = os.path.join(exp_dir, "needle_answer.jsonl")
    for rank in range(world_size):
        p = mp.Process(target=needle_worker, args=(rank, world_size, max_pe, out_path, times, min_k, max_k, gap, config))
        p.start()
        processes.append(p)
        break # only support single card now
    for p in processes:
        p.join()

def generate_prompt_landmark(n_garbage, seed):
    # code comes from chunkllama
    """Generates a text file and inserts an passkey at a random position."""
    # rnd_state = random.getstate()
    # random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 500000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    # random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def passkey_retrieval_worker(rank, world_size, max_pe, out_path, n_garbage, stride, times, config):
    # code comes from chunkllama
    model_name = config['model']
    method = config['method']
    params = config['params']
    ori_max_position_embeddings = config.get('ori_max_position_embeddings', None)
    model, tokenizer = get_model_and_tokenizer(model_name, config, method, params, max_position_embeddings=ori_max_position_embeddings)
    seed = 42
    seed_everything(seed)
    device = torch.device(f'cuda:{rank}')
    model = model.eval().to(device)
    
    if 'dpi' == method:
        output_attentions = True
    else:
        output_attentions = False

    correct = 0
    n_garbage = int(3.75 * n_garbage // 1024 * 1024) # approimate the real n_garbage
    for i in range(times):
        prompt, answer = generate_prompt_landmark(n_garbage, seed)

        # add by dpi
        prompt = build_chat_lbench(tokenizer, prompt, model_name)
        inputs = wrap_with_model_special_input(model_name, prompt, tokenizer, device, "passkey")

        len_token = inputs.input_ids.shape[-1]
        print("len tokens", len_token // 1000, "k")
        answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
        torch.cuda.empty_cache()
        generation_output = model.generate(
            input_ids=inputs.input_ids, max_new_tokens=answer_ids.shape[-1] + 100, num_beams=1, use_cache=True,  do_sample=False, temperature=1.0,
            output_attentions = output_attentions,
            return_dict_in_generate = True,
        )['sequences'][0]

        prompt_length = inputs.input_ids.size()[-1]
        model_answer = generation_output[prompt_length:].cpu()
        # model_answer = generation_output[-answer_ids.shape[-1]:].cpu()
        pred = tokenizer.decode(model_answer, skip_special_tokens=True)
        torch.cuda.empty_cache()
        # model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
        # is_correct = (model_answer == answer_ids[0]).all().item()
        print(f"pred: {pred}, answer {answer}")
        if answer in pred:
            correct += 1
        print("--------")
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"answer": answer, "pred":pred}, f, ensure_ascii=False)
            f.write('\n')
    acc = correct/times
    print("accuracy: ", acc)
    with open(out_path, "a", encoding="utf-8") as f:
        json.dump({"accuracy": acc}, f, ensure_ascii=False)


def passkey_retrieval(config, world_size):
    exp_name = config['exp_name']
    max_pe = config['max_pe']
    n_garbage = config['n_garbage']
    times = config['times']
    stride = 0

    results_root = "results/passkey/"
    exp_base_path = os.path.join(results_root, 'pred')
    if not os.path.exists(exp_base_path):
        os.makedirs(exp_base_path)

    base_folder_name = f"{exp_name}"
    exp_folder_name = ensure_unique_folder(exp_base_path, base_folder_name)
    exp_dir = os.path.join(exp_base_path, exp_folder_name)
    os.makedirs(exp_dir)

    print(f"Your prediction file will be saved to: {exp_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}.json"

    file_path = os.path.join(exp_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(config, json_file, ensure_ascii=False, indent=4)
    
    processes = []
    out_path = os.path.join(exp_dir, "passkey_answer.jsonl")
    for rank in range(world_size):
        p = mp.Process(target=passkey_retrieval_worker, args=(rank, world_size, max_pe, out_path, n_garbage, stride, times, config))
        p.start()
        processes.append(p)
        break # only support single card now
    for p in processes:
        p.join()


def pg19_pred_worker(rank, world_size, data, max_pe, split, out_path, stride, test, config):
    model_name = config['model']
    method = config['method']
    params = config['params']
    ori_max_position_embeddings = config.get('ori_max_position_embeddings', None)
    model, tokenizer = get_model_and_tokenizer(model_name, config, method, params, max_position_embeddings=ori_max_position_embeddings)
    device = torch.device(f'cuda:{rank}')
    model = model.eval().to(device)
    if method == 'repro_chunkllama':
        setattr(model, 'cache_all_logits', True)
    max_prefill_length = stride

    save_data = []
    log_prob_list = []
    if 'dpi' == method:
        output_attentions = True
    else:
        output_attentions = False
    with torch.no_grad():
        for sample in tqdm(data):
            file_name, text = sample
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_pe-1).to(device)
            input_ids = encodings.input_ids
            # position_id = encodings.position_id
            n_tokens = input_ids.size(1)
            log_prob_list.append([])
            # Sliding window approach
            # stride_log_prob = 0
            past_key_values = None
            if method in ('dpi','repro_chunkllama'):
                if method == 'repro_chunkllama':
                    setattr(model, 'logits_cache', ())

                all_labels = input_ids.clone()
                all_logits = model.generate(**encodings,
                    max_new_tokens=1,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=True,
                    output_attentions = output_attentions,
                    return_dict_in_generate = True,
                    output_logits = True,
                    top_p=None)
                # print("all_logits ",all_logits.keys())
                if method == 'repro_chunkllama':
                    all_logits = getattr(model, 'logits_cache', ())
                else:
                    all_logits = all_logits['logits']
                    
                all_logits = torch.cat(all_logits,dim=1).to(all_labels.device)
                # print("all_logits ",all_logits.shape, input_ids.shape)
                # Calculate log probabilities
                shift_logits = all_logits[:, :-1, :].contiguous()
                shift_labels = all_labels[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # stride_log_prob += loss.sum().item()
                log_prob_list[-1].extend(loss.tolist())
            else:
                if method== 'repro_se':
                    stride = n_tokens
                for i in range(0, n_tokens, stride):
                    end = i + stride
                    input_ids_chunk = input_ids[:, i:end]
                    labels_chunk = input_ids_chunk.clone()

                    # Get logits from model
                    # if method == 'repro_chunkllama':
                    #     outputs = model(input_ids_chunk, labels=labels_chunk, use_cache=True)
                    # else:
                    # print("len(input_ids_chunk) ",input_ids_chunk.shape)
                    outputs = model(input_ids_chunk, labels=labels_chunk, use_cache=True, past_key_values=past_key_values, output_attentions=output_attentions)
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits

                    # Calculate log probabilities
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels_chunk[:, 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    # stride_log_prob += loss.sum().item()
                    log_prob_list[-1].extend(loss.tolist())

            save_data.append({"log_prob_list": log_prob_list[-1], "file_name":file_name})
            # with open(out_path, "a", encoding="utf-8") as f:
            #     json.dump({"log_prob_list": log_prob_list[-1], "file_name":file_name}, f, ensure_ascii=False)
            # total_log_prob += stride_log_prob
            # total_tokens += n_tokens
    torch.save(save_data, out_path)
    # Calculate perplexity
    # avg_log_prob = total_log_prob / total_tokens
    # perplexity = torch.exp(torch.tensor(avg_log_prob))
    dist.destroy_process_group()

def read_txt_files_to_dict(directory):
    """
    读取指定目录下的所有 .txt 文件，并将它们存储在一个字典中。

    参数:
        directory (str): 要读取的目录路径。

    返回:
        dict: 一个字典，键是文件名，值是文件内容。
    """
    txt_files_dict = {}

    # 检查目录是否存在
    if not os.path.exists(directory):
        raise ValueError(f"目录 '{directory}' 不存在。")
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为 .txt
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            # 读取文件内容并存储到字典
            with open(file_path, 'r', encoding='utf-8') as f:
                txt_files_dict[filename] = f.read()

    return txt_files_dict

def pg19_pred(config, world_size):
    exp_name = config['exp_name']
    test = config['test']
    max_pe = config['max_pe']
    stride = config['stride']

    results_root = "results/pg19/"
    exp_base_path = os.path.join(results_root, 'pred')
    if not os.path.exists(exp_base_path):
        os.makedirs(exp_base_path)

    base_folder_name = f"{exp_name}"
    exp_folder_name = ensure_unique_folder(exp_base_path, base_folder_name)
    exp_dir = os.path.join(exp_base_path, exp_folder_name)
    os.makedirs(exp_dir)

    print(f"Your prediction file will be saved to: {exp_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}.json"

    file_path = os.path.join(exp_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(config, json_file, ensure_ascii=False, indent=4)

    split = 'test'
    pg19 = read_txt_files_to_dict(f"pg19/{split}")
    # pg19 = datasets.load_dataset("deepmind/pg19",split=split)

    
    data_all = [(file_name, text) for file_name,text in pg19.items()]
    if test == True:
        idx = 0
        # print("data_all[idx]['length'] ",data_all[idx]['length'], data_all[idx]['answers'])
        data_subsets = [[data_all[idx]][i::world_size] for i in range(world_size)]
    else:
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
    processes = []
    out_path = os.path.join(exp_dir, "log_prob_list.pt")
    for rank in range(world_size):
        p = mp.Process(target=pg19_pred_worker, args=(rank, world_size, data_subsets[rank], max_pe, \
                    split, out_path, stride, test, config))
        p.start()
        processes.append(p)
        break # only support single card now
    for p in processes:
        p.join()


def leval_pred_worker(rank, world_size, data, max_pe, max_gen, dataset, metric, out_path, test, config):
    # openai.api_base = "https://api.openai-sb.com/v1"
    model_name = config['model']
    method = config['method']
    params = config['params']
    ori_max_position_embeddings = config.get('ori_max_position_embeddings', None)
    model, tokenizer = get_model_and_tokenizer(model_name, config, method, params, max_position_embeddings=ori_max_position_embeddings)
    device = torch.device(f'cuda:{rank}')
    start_idx = 0
    fw = open(out_path, "w")
    model = model.eval().to(device)
    max_prefill_length = max_pe - LEval_config.max_new_tokens

    if 'dpi' == method:
        output_attentions = True
    else:
        output_attentions = False
    with torch.no_grad():
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while LEval_config.num_tokens_from_string(document, tokenizer) > max_prefill_length - 50:
                if "code" not in out_path:
                    document = " ".join(document.split(" ")[:max_prefill_length - cnt]) # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_prefill_length:]) # chunk the input len from left
                cnt += 250                

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                sys_prompt, message = build_chat_leval(metric, inst, document, out_path, model_name)
                save_d['prompt'] = message.replace(document, "<long document>")
                text_inputs = message

                inputs = wrap_with_model_special_input(model_name, message, tokenizer, device, dataset, sys_prompt = sys_prompt)
                
                if  "Llama-3" in model_name:
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                else:
                    terminators = None

                sample = model.generate(**inputs,
                        eos_token_id=terminators,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        output_attentions = output_attentions,
                        return_dict_in_generate = True,
                        top_p=None)['sequences'][0]
                prompt_length = inputs.input_ids.size()[-1]
                output = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)

                # save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
                save_d['pred'] = output
                save_d['evaluation'] = d['evaluation']
                save_d['prefill_len'] = prompt_length
                save_d['final_len'] = len(sample)

                # test the factuality in scientific fiction
                if "sci_fi" in out_path:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "Please directly answer without any additional output or explanation. \nAnswer:"

                    inputs = wrap_with_model_special_input(model_name, text_inputs, tokenizer, device, dataset, sys_prompt = None)

                    sample = model.generate(**inputs,
                        eos_token_id=terminators,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        output_attentions = True,
                        return_dict_in_generate = True,
                        top_p=None)['sequences'][0]
                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)
                    save_d['pred'] += f" [fact: {output}]"
                    save_d['final_len_fact'] = len(sample)

                if start_idx < 10000000:
                    print('document len', LEval_config.num_tokens_from_string(document, tokenizer))
                    print("[document]:",text_inputs[:100] + "...")
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d['pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
                fw.flush()
                # break
    fw.close()
    dist.destroy_process_group()

def leval_pred(config, world_size):

    max_pe = config['max_pe']
    test = config['test']
    exp_name = config['exp_name']
    metric = 'exam_eval'

    results_root = "results/le/"
    exp_base_path = os.path.join(results_root, 'pred')
    if not os.path.exists(exp_base_path):
        os.makedirs(exp_base_path)

    base_folder_name = f"{exp_name}"
    exp_folder_name = ensure_unique_folder(exp_base_path, base_folder_name)
    exp_dir = os.path.join(exp_base_path, exp_folder_name)
    os.makedirs(exp_dir)

    print(f"Your prediction file will be saved to: {exp_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}.json"

    file_path = os.path.join(exp_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(config, json_file, ensure_ascii=False, indent=4)

    key_data_pairs = LEval_config.build_key_data_pairs_dpiversion(exp_dir)

    for out_path, data in key_data_pairs.items():
        data_all = [data_sample for data_sample in data]
        if test == True:
            idx = 0
            # print("data_all[idx]['length'] ",data_all[idx]['length'], data_all[idx]['answers'])
            data_subsets = [[data_all[idx]][i::world_size] for i in range(world_size)]
        else:
            data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        path, tail = os.path.split(out_path)
        dataset = tail.split('.')[0]
        for rank in range(world_size):
            p = mp.Process(target=leval_pred_worker, args=(rank, world_size, data_subsets[rank], max_pe, \
                        LEval_config.max_new_tokens, dataset, metric, out_path, test, config))
            p.start()
            processes.append(p)
            break # only support single card now
        for p in processes:
            p.join()

def longbench_pred_worker(rank, world_size, data, max_pe, max_gen, prompt_format, dataset, out_path, test, config):
    model_name = config['model']
    method = config['method']
    params = config['params']
    ori_max_position_embeddings = config.get('ori_max_position_embeddings', None)
    model, tokenizer = get_model_and_tokenizer(model_name, config, method, params, max_position_embeddings=ori_max_position_embeddings)
    device = torch.device(f'cuda:{rank}')
    print("start get_pred, device: ",device, dataset)
    model = model.eval().to(device)
    # max_gen = 2
    predictions = []
    answers = []
    all_classes = None
    lengths = []
    max_prefill_length = max_pe - max_gen

    if 'dpi' == method:
        output_attentions = True
    else:
        output_attentions = False

    with torch.no_grad():
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            if test == True:
                print("prompt: {} ... {}".format(prompt[:2000], prompt[-2000:]))
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_prefill_length - 50:
                half = int((max_prefill_length-50)/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat_lbench(tokenizer, prompt, model_name)
            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = wrap_with_model_special_input(model_name, prompt, tokenizer, device)
                else:
                    input = prompt.to(device)
            else:
                input = wrap_with_model_special_input(model_name, prompt, tokenizer, device, dataset)
            context_length = input.input_ids.shape[-1]
            if test == True:
                print("context_length ",context_length)

            if  "Llama-3" in model_name:
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            else:
                terminators = None
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                if terminators is None:
                    terminators = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
                else:
                    terminators.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    output_attentions = output_attentions,
                    return_dict_in_generate = True,
                    eos_token_id=terminators,
                    top_p = None
                )['sequences'][0]
            else:
                output = model.generate(
                    **input,
                    eos_token_id=terminators,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    output_attentions = output_attentions,
                    return_dict_in_generate = True,
                    top_p=None
                )['sequences'][0]
                
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)

            predictions.append(pred)
            answers.append(json_obj["answers"])
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "prefill_len":context_length, "final_len": len(output)}, f, ensure_ascii=False)
                f.write('\n')
    if test == True:
        score = scorer(dataset, predictions, answers, json_obj["all_classes"])
        print("score: ",score)
    dist.destroy_process_group()


def longbench_pred(config, world_size):

    max_pe = config['max_pe']
    test = config['test']
    exp_name = config['exp_name']
    longbench_e = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if longbench_e == True:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        # datasets = ["2wikimqa"]
    if test == True:
        datasets = ["narrativeqa"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    results_root = "results/lb/"
    if not os.path.exists(f"{results_root}pred"):
        os.makedirs(f"{results_root}pred")
    if not os.path.exists(f"{results_root}pred_e"):
        os.makedirs(f"{results_root}pred_e")
    if longbench_e == True:
        data_name_tmp = "{}_e"
        exp_base_path = f"{results_root}pred_e"
    else:
        data_name_tmp = "{}"
        exp_base_path = f"{results_root}pred"
    base_folder_name = f"{exp_name}"
    exp_folder_name = ensure_unique_folder(exp_base_path, base_folder_name)
    exp_dir = os.path.join(exp_base_path, exp_folder_name)
    os.makedirs(exp_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}.json"

    file_path = os.path.join(exp_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(config, json_file, ensure_ascii=False, indent=4)

    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', data_name_tmp.format(dataset), split='test')
        out_path = os.path.join(exp_dir, f"{dataset}.jsonl")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if test == True:
            idx = 0
            # print("data_all[idx]['length'] ",data_all[idx]['length'], data_all[idx]['answers'])
            data_subsets = [[data_all[idx]][i::world_size] for i in range(world_size)]
        else:
            data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=longbench_pred_worker, args=(rank, world_size, data_subsets[rank], max_pe, \
                        max_gen, prompt_format, dataset, out_path, test, config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    cfg_name = args.cfg
    # keep = args.keep
    # if keep != True:
    #     import sys
    #     sys.exit(0)
    config = get_params(cfg_name)
    # model_name = config['model']
    # method = config['method']
    task = config['task']
    # max_pe = config['max_pe']
    # test = config['test']
    # params = config['params']
    # exp_name = config['exp_name']


    # model, tokenizer = get_model_and_tokenizer(model_name, config, method, params)
    if task == "longbench":
        longbench_pred(config, world_size)
    elif task == "leval":
        leval_pred(config, 1)
    elif task == "pg19":
        # stride = config['stride']
        pg19_pred(config, 1)
    elif task == "passkey_retreival":
        # n_garbage = config['n_garbage']
        # times = config['times']
        stride = 0 # unsupport now
        passkey_retrieval(config, 1)
    elif task == "needle":
        # times = config['times']
        # min_k = config['min_k']
        # max_k = config['max_k']
        # gap = config['gap']
        needle(config, 1)
