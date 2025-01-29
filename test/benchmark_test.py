import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored
from datasets import load_dataset
from tqdm import tqdm
from data.dataset import get_dataset
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
from models.cache import FlashSimpleCache, StreamingLLMEvictionCache, RetrievalCache
from utils.decoding import Autoregressive, TriForce
from utils.misc import print_config
from utils.graph_infer import GraphInferenceEngine
import re


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68M', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='gs', help='dataset')
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


if __name__ == "__main__":

    args = parse_arguments()

    ######## model initialization ########
    target = LlamaForCausalLM.from_pretrained(args.target, torch_dtype=torch.float16, device_map="cuda:0")
    target = target.eval()
    
    draft = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="cuda:0")
    draft = draft.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True, legacy=False)
    ds = load_dataset('THUDM/LongBench-v2', split='train')

    ######## sampling parameters ########

    top_k = -1
    top_p = args.top_p
    temperature = args.temp
    prefill = args.prefill
    gamma = args.gamma
    verbose = args.verbose
    chunk_size = args.chunk_size
    max_budget = args.budget

    print_config(draft, target, prefill, gamma, top_k, top_p, temperature, file_path=None, method="TriForce", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset=args.dataset)

    ######## Warm up for our method ########
    if args.infer_mode == "zero-shot":
        query_template = open('prompt_templates/query.txt', encoding='utf-8').read()
    else:
        query_template = open('prompt_templates/query_cot.txt', encoding='utf-8').read()
        cot_template = open('prompt_templates/query_cot_ans.txt', encoding='utf-8').read()

    all_acceptance_rate = []
    all_speed = []
    
    acc = 0
    
    for i, item in tqdm(enumerate(ds), total=len(ds), desc="Evaluating"):
        long_context = item["context"]
        query = query_template.replace('$DOC$', long_context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        if args.infer_mode == "zero-shot":
            gen_len = 128
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
            
            messages = [
                {"role": "system", "content": "You are a knowledgeable person."},
                {"role": "user", "content": query},
            ]
                
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = tokenizer(
                [text], 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids.to(target.device)
            
            if input_ids.shape[1] > 120000:
                    input_ids = torch.cat([input_ids[:, :60000], input_ids[:, -60000:]], dim=1) 
            
            acceptance_rate, speed, outputs = TriForce(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, spec_args=None)
            
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            print(output)
            answer = extract_answer(output)
            if answer == item['answer']:
                acc += 1
        
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)

    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[TriForce] average latency: {method_latency} ms", "red"))