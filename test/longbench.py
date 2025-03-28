import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from datasets import load_dataset
import re
from tqdm import tqdm
from data.dataset import get_dataset
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
from models.modelling_phi3 import Phi3ForCausalLM
from models.cache import FlashSimpleCache, StreamingLLMEvictionCache, RetrievalCache
from utils.decoding import Autoregressive, TriForce
from utils.misc import print_config
from utils.graph_infer import GraphInferenceEngine
import pandas as pd

from transformers import set_seed

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68M', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--dataset', type=str, default='longbench_v2', help='dataset')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    parser.add_argument('--infer_mode', type=str, default='zero-shot', help='inference mode')
    
    args = parser.parse_args()
    
    return args

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None
        
def create_input_ids(tokenizer, query, max_len=120000, device="cuda:0", use_chat_template=True):
    if use_chat_template:
        messages = [
            {"role": "user", "content": query},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = query
    
    input_ids = tokenizer(
        [text], 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids.to(device)
    
    if input_ids.shape[1] > max_len:
        input_ids = torch.cat([input_ids[:, :max_len//2], input_ids[:, -max_len//2:]], dim=1)
        
    return input_ids

if __name__ == "__main__":
    args = parse_arguments()

    ######## model initialization ########
    if args.target == "NousResearch/Yarn-Llama-2-7b-128k":
        target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="cuda:0")
        use_chat_template = False
    elif args.target == 'microsoft/Phi-3-mini-128k-instruct':
        target = Phi3ForCausalLM.from_pretrained(
            args.target, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="flash_attention_2")
        use_chat_template = True
    elif args.target == 'LargeWorldModel/LWM-Text-Chat-128K':
        target = LlamaForCausalLM.from_pretrained(args.target, torch_dtype=torch.float16, device_map="cuda:0")
        use_chat_template = False
    else:    
        raise NotImplementedError
        
    target = target.eval()

    draft = LlamaForCausalLM_68M.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="cuda:0")
    draft = draft.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True, legacy=False)

    ######## sampling parameters ########

    top_k = -1
    top_p = args.top_p
    temperature = args.temp

    prefill = args.prefill
    gen_len = args.gen_len
    gamma = args.gamma
    verbose = args.verbose
    chunk_size = args.chunk_size
    max_budget = args.budget

    print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=None, method="TriForce", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset="longbenchv2")
    
    ####### cache init #######

    draft_cache_budget = args.draft_cache_budget
    recent_size = draft_cache_budget - 16 - gamma
    cache = FlashSimpleCache(target, prefill+gen_len+16)
    graph_cache = RetrievalCache(target, max_budget=max_budget, prefill=prefill, gamma=gamma, chunk_size=chunk_size)
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

    graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
    graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=temperature, top_p=top_p)

    cache.print_status()
    graph_cache.print_status()
    draft_cache.print_status()
    
    ######## inference ########
    set_seed(args.seed)
    
    if args.infer_mode == "zero-shot":
        query_template = open('prompt_templates/query.txt', encoding='utf-8').read()
    else:
        query_template = open('prompt_templates/query_cot.txt', encoding='utf-8').read()
        cot_template = open('prompt_templates/query_cot_ans.txt', encoding='utf-8').read()
    
    ds = load_dataset('THUDM/LongBench-v2', split='train')
    acc = 0
    latency_dict = {}
    for i, item in tqdm(enumerate(ds), total=len(ds), desc="Evaluating"):
        long_context = item["context"]
        query = query_template.replace('$DOC$', long_context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        if args.infer_mode == "zero-shot":
            max_new_tokens = 128
            input_ids = create_input_ids(tokenizer, query, max_len=args.prefill, use_chat_template=use_chat_template)
            generate_ids, latency = TriForce(
                tokenizer, graph_engine, input_ids, gamma=gamma, max_len=max_new_tokens, top_k=top_k, top_p=top_p, 
                temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, 
                spec_args=None, return_tokens=True)
            output = tokenizer.decode(generate_ids, skip_special_tokens=True)
        else:
            max_new_tokens = 1024
            input_ids = create_input_ids(tokenizer, query, max_len=args.prefill, use_chat_template=use_chat_template)
            cot_context, latency = TriForce(
                tokenizer, graph_engine, input_ids, gamma=gamma, max_len=max_new_tokens, top_k=top_k, top_p=top_p, 
                temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, 
                spec_args=None, return_tokens=True)
            output = tokenizer.decode(generate_ids, skip_special_tokens=True)
            query = cot_template.replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', cot_context)
            input_ids = create_input_ids(tokenizer, query, max_len=args.prefill, use_chat_template=use_chat_template)
            generate_ids, acceptance_rate, speed = TriForce(
                tokenizer, graph_engine, input_ids, gamma=gamma, max_len=128, top_k=top_k, top_p=top_p, 
                temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, 
                spec_args=None, return_tokens=True)
            output = tokenizer.decode(generate_ids, skip_special_tokens=True)
        
        print(output)
        answer = extract_answer(output)
        if answer == item['answer']:
            acc += 1
        
        if i == 0:
            for key in latency:
                latency_dict[key] = [latency[key]]
            
            for key in ["_id", "domain", "sub_domain", "difficulty", "length"]:
                latency_dict[key] = [item[key]]
        else:
            for key in latency:
                latency_dict[key].append(latency[key])
                
            for key in ["_id", "domain", "sub_domain", "difficulty", "length"]:
                latency_dict[key].append(item[key])
            
        
    if args.dataset == "longbench_v2":
        df = pd.DataFrame(latency_dict)
        with open(f'{args.target}-triforce-{args.infer_mode}-accuracy.txt', 'w') as f:
            f.write(f'Accuracy: {acc/len(ds)}')
            f.write('\n')
            f.write(f'Average Prefill time: {df["prefill_time"].mean()}')
            f.write('\n')
            f.write(f'Average Tokens Per Second: {df["tokens_per_second"].mean()}')
        
        df.to_csv(f'{args.target}-triforce-{args.infer_mode}-latency.csv', index=False)
        