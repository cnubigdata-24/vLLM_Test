# 실습을 위해 Hugging Face 가입 필요
# 구글 코랩 실행시 Hugging Face Access Token 환경변수 설정 필요: HF_TOKEN

import sys
import os

# GPU 확인
get_ipython().system('nvidia-smi')

# vLLM 설치 여부 확인
try:
    import vllm
    VLLM_INSTALLED = True
except ImportError:
    VLLM_INSTALLED = False

if not VLLM_INSTALLED:
    print("="*80)
    print("vLLM 설치 시작...")
    print("="*80)
    
    print("pyairports 설치 중...")
    get_ipython().system('pip install pyairports -q')
    print("vLLM 설치 중...")
    get_ipython().system('pip install vllm -q')
    
    print("\n"+"="*80)
    print("설치 완료! 런타임 재시작 중...")
    print("="*80)
    
    import IPython
    IPython.Application.instance().kernel.do_shutdown(True)

# 여기부터는 vLLM이 설치되어 있을 때만 실행됨
print("="*80)
print("테스트 시작")
print("="*80)

os.environ['VLLM_USE_V1'] = '1'

import torch
import gc
torch.cuda.empty_cache()
gc.collect()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "facebook/opt-125m"
print(f"\n[모델 로딩 시작] {model_name}")
print("-" * 80)

print("\n1. vLLM 모델 로드... ")
print("-" * 80)

start_vllm_load = time.time()
llm = LLM(model=model_name)
vllm_load_time = time.time() - start_vllm_load

print(f"   vLLM 로드 시간: {vllm_load_time:.2f}초")
print("-" * 80)

print("\n2. HuggingFace 모델 로드 중...")
print("-" * 80)

start_hf_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
hf_load_time = time.time() - start_hf_load

print(f"   HuggingFace 로드 시간: {hf_load_time:.2f}초")
print("-" * 80)
print(f">> vLLM의 초기 로드 시간이 HuggingFace보다 긴 것이 일반적:  KV 캐시 블록 풀 메모리 사전 할당, CUDA 커널들을 사전 컴파일하고 최적화, 배치 처리 준비")
print(f">> HuggingFace는 모델만 로드하고 추론 시점에 필요한 만큼만 메모리를 할당하여 초기 로딩 속도가 더 빠름")

print("\n1. 단일 추론 속도 비교")
print("=" * 80)

prompt = "The future of AI is"
params = SamplingParams(temperature=0.8, max_tokens=50)

# vLLM
print("\n[vLLM 추론]")
print("-" * 80)

start_vllm = time.time()
vllm_outputs = llm.generate([prompt], params)
vllm_time = time.time() - start_vllm

print(f">> 입력: {prompt}")
print(f">> 출력: {vllm_outputs[0].outputs[0].text[:100]}...")
print(f">> 처리 시간: {vllm_time:.3f}초")

# HuggingFace
print("\n[HuggingFace 추론]")
print("-" * 80)

start_hf = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    hf_outputs = hf_model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )
hf_time = time.time() - start_hf
hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)

print(f">> 입력: {prompt}")
print(f">> 출력: {hf_text[len(prompt):len(prompt)+100]}...")
print(f">> 처리 시간: {hf_time:.3f}초")

print(f"\n>> 속도 비교: vLLM이 HuggingFace보다 {hf_time/vllm_time:.1f}배 빠름")

print("\n2. 배치 처리 속도 비교 (Continuous Batching)")
print("=" * 80)

prompts = [
    "Artificial intelligence can",
    "Machine learning is",
    "Deep learning helps",
    "Neural networks are",
    "Python programming"
]

# vLLM
print(f"\n[vLLM 배치 추론] {len(prompts)}개 동시 처리")
print("-" * 80)

start_vllm_batch = time.time()
vllm_batch_outputs = llm.generate(prompts, params)
vllm_batch_time = time.time() - start_vllm_batch

for i, output in enumerate(vllm_batch_outputs[:5]):
    print(f"  [{i+1}] {prompts[i]}")
    print(f"      -> {output.outputs[0].text[:60]}...")

print(f"  ... (총 {len(prompts)}개)")
print(f">> 총 처리 시간: {vllm_batch_time:.3f}초")
print(f">> 처리량: {len(prompts)/vllm_batch_time:.2f} 요청/초")
print("-" * 80)

# HuggingFace
print(f"\n[HuggingFace 배치 추론] {len(prompts)}개 순차 처리")
print("-" * 80)

start_hf_batch = time.time()
hf_batch_outputs = []
for prompt_item in prompts:
    inputs = tokenizer(prompt_item, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True
        )
    hf_batch_outputs.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
hf_batch_time = time.time() - start_hf_batch

for i in range(5):
    print(f"  [{i+1}] {prompts[i]}")
    print(f"      -> {hf_batch_outputs[i][len(prompts[i]):len(prompts[i])+60]}...")

print(f"  ... (총 {len(prompts)}개)")
print(f">> 총 처리 시간: {hf_batch_time:.3f}초")
print(f">> 처리량: {len(prompts)/hf_batch_time:.2f} 요청/초")

print(f"\n>> 배치 처리 속도 비교: vLLM이 HuggingFace보다 {hf_batch_time/vllm_batch_time:.1f}배 빠름")

print("\n3. 메모리 효율성 비교 (PagedAttention): 긴 시퀀스 생성 시 메모리 사용량")
print("-" * 80)

test_prompt = "Write a detailed story about artificial intelligence:"
max_tokens = 2000

print(f"\n최대 생성 토큰: {max_tokens}개")
print("─" * 60)

# vLLM 메모리 측정
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

memory_before_vllm = torch.cuda.memory_allocated() / (1024**2)

params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
start_vllm = time.time()
vllm_output = llm.generate([test_prompt], params)
vllm_time = time.time() - start_vllm

memory_after_vllm = torch.cuda.memory_allocated() / (1024**2)
memory_peak_vllm = torch.cuda.max_memory_allocated() / (1024**2)
memory_used_vllm = memory_peak_vllm - memory_before_vllm

print(f"\nvLLM PagedAttention:")
print("-" * 80)
print(f"  처리 시간: {vllm_time:.3f}초")
print(f"  추론 전 메모리: {memory_before_vllm:.2f} MB")
print(f"  최대 메모리: {memory_peak_vllm:.2f} MB")
print(f"  추가 사용: {memory_used_vllm:.2f} MB")

# HuggingFace 메모리 측정
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

memory_before_hf = torch.cuda.memory_allocated() / (1024**2)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
start_hf = time.time()
with torch.no_grad():
    hf_output = hf_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True
    )
hf_time = time.time() - start_hf

memory_after_hf = torch.cuda.memory_allocated() / (1024**2)
memory_peak_hf = torch.cuda.max_memory_allocated() / (1024**2)
memory_used_hf = memory_peak_hf - memory_before_hf

print(f"\nHuggingFace:")
print("-" * 80)
print(f"  처리 시간: {hf_time:.3f}초")
print(f"  추론 전 메모리: {memory_before_hf:.2f} MB")
print(f"  최대 메모리: {memory_peak_hf:.2f} MB")
print(f"  추가 사용: {memory_used_hf:.2f} MB")

# 비교
memory_savings = memory_used_hf - memory_used_vllm
memory_savings_pct = (memory_savings / memory_used_hf) * 100
speed_ratio = hf_time / vllm_time

print("\n" + "─" * 60)
print("비교 결과:")
print(f"  메모리 절약: {memory_savings:.2f} MB ({memory_savings_pct:.1f}%)")
print(f"  속도: vLLM이 {speed_ratio:.1f}배 빠름")
print("─" * 60)

print("\n vLLM PagedAttention:")
print("  - KV 캐시를 고정 크기 블록(페이지)으로 분할")
print("  - 필요할 때마다 블록을 동적으로 할당, 메모리 단편화 최소화 및 효율적 재사용")

print("\n HuggingFace:")
print("  - KV 캐시를 연속된 메모리 공간에 사전 할당, 최대 시퀀스 길이만큼 메모리 예약")

print("\n4. 동시 사용자 처리 비교")
print("=" * 80)

user_queries = [f"User {i}: Hello, my name is" for i in range(1, 11)]

# vLLM 동시 처리
print(f"\n[vLLM] 10명의 동시 사용자 요청 처리 (Continuous Batching)")
print("-" * 80)
start_vllm_users = time.time()
vllm_user_outputs = llm.generate(user_queries, SamplingParams(temperature=0.7, max_tokens=20))
vllm_users_time = time.time() - start_vllm_users

for i in range(3):
    print(f"  {user_queries[i]}")
    print(f"    -> {vllm_user_outputs[i].outputs[0].text}")
print(f"  ... (총 10개 요청)")
print(f">> 처리 시간: {vllm_users_time:.3f}초")
print(f">> 처리 능력: {10/vllm_users_time:.1f} 사용자/초")
print("-" * 80)

# HuggingFace 순차 처리
print(f"\n[HuggingFace] 10명의 사용자 요청 순차 처리")
print("-" * 80)
start_hf_users = time.time()
hf_user_outputs = []
for query in user_queries:
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
    hf_user_outputs.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
hf_users_time = time.time() - start_hf_users

for i in range(3):
    print(f"  {user_queries[i]}")
    print(f"    -> {hf_user_outputs[i][len(user_queries[i]):]}")
print(f"  ... (총 10개 요청)")
print(f">> 처리 시간: {hf_users_time:.3f}초")
print(f">> 처리 능력: {10/hf_users_time:.1f} 사용자/초")

print(f"\n동시 사용자 처리: vLLM이 HuggingFace보다 {hf_users_time/vllm_users_time:.1f}배 빠름")
print("\nvLLM의 Continuous Batching:")
print("  - 여러 사용자 요청을 효율적으로 동시 처리, 실시간 서비스에 최적화, 높은 Throughput")
print("\nHuggingFace:")
print("  - 기본적으로 순차 처리, 배치 처리도 가능하지만 vLLM만큼 최적화되지 않음")
