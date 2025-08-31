#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LoRA pipeline (model-wise, NOT benchmark-wise):
1) Preprocess mixed train/eval JSONL into a single SFT format
2) Train LoRA per model on the full train set (no per-benchmark split)
3) Merge adapters into base weights
4) Evaluate per model on the full eval set

Refs:
- TRL SFTTrainer quickstart: https://huggingface.co/docs/trl/main/en/sft_trainer
- TRL SFTTrainer docs (versioned): https://huggingface.co/docs/trl/v0.11.1/en/sft_trainer
- PEFT LoRA merge_and_unload: https://huggingface.co/docs/peft/main/en/developer_guides/lora
"""
import os, re, json, argparse
from typing import List, Dict

import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GenerationConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_gsm8k_final(ans: str) -> str:
    # GSM8K format ends with: "#### <final_answer>"
    m = re.search(r"####\s*([^\n]+)$", ans.strip())
    if m:
        return m.group(1).strip()
    # fallback: last number
    nums = re.findall(r"[-]?\d+(?:\.\d+)?", ans.replace(",", ""))
    return nums[-1] if nums else ans.strip()

# Prompt builders (kept minimal and instruction-y across listed models)
def build_prompt_mmlu(q, choices):
    letters = ["A","B","C","D"]
    lines = "\n".join([f"{L}. {c}" for L, c in zip(letters, choices)])
    return (
        "You are a helpful assistant. Choose the correct option (A, B, C, or D).\n"
        f"Question: {q}\nChoices:\n{lines}\nAnswer:"
    )

def build_prompt_arc(q, choices_text):
    letters = ["A","B","C","D"]
    lines = "\n".join([f"{L}. {c}" for L, c in zip(letters, choices_text)])
    return (
        "You are a helpful math QA assistant. Choose the correct option (A, B, C, or D).\n"
        f"Question: {q}\nChoices:\n{lines}\nAnswer:"
    )

def build_prompt_gsm8k(q):
    return (
        "Solve the problem. Provide only the final numeric answer after 'Answer:'.\n"
        f"Problem: {q}\nAnswer:"
    )

def normalize_row(row: dict) -> dict:
    """
    Expect rows shaped like:
    {
      "benchmark": "mmlu" | "ai2_arc" | "gsm8k",
      "raw_example": { ...original fields... }
    }
    We convert to a unified SFT record: {benchmark, prompt, completion, messages}
    """
    b = row.get("benchmark")
    ex = row.get("raw_example", {})
    if b == "ai2_arc":
        q = ex["question"]
        choices = ex["choices"]["text"]
        gold = str(ex["answerKey"]).strip()
        prompt = build_prompt_arc(q, choices)
        target = gold
    elif b == "mmlu":
        q = ex["question"]
        choices = ex["choices"]
        gold = str(ex["answer"]).strip()
        prompt = build_prompt_mmlu(q, choices)
        target = gold
    elif b == "gsm8k":
        q = ex["question"]
        gold = extract_gsm8k_final(ex["answer"])
        prompt = build_prompt_gsm8k(q)
        target = gold
    else:
        raise ValueError(f"Unknown row schema: keys={list(row.keys())}")
    
    # Fix: Use messages format for apply_chat_template
    # This ensures the model's native chat format is used
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target}
    ]
    
    return {
        "benchmark": b,
        "prompt": prompt,
        "completion": target,
        "messages": messages  # Store messages for apply_chat_template
    }

def preprocess_mixed(path: str) -> Dataset:
    rows = read_jsonl(path)
    norm = [normalize_row(r) for r in rows]
    return Dataset.from_list(norm)

def apply_chat_template_to_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """
    Apply the model's chat template to convert messages to text format.
    This ensures consistent tokenization behavior.
    """
    def apply_template(row):
        try:
            # Use the model's native chat template
            text = tokenizer.apply_chat_template(
                row["messages"], 
                tokenize=False, 
                add_generation_prompt=False
            )
            return {**row, "text": text}
        except Exception as e:
            # Fallback: simple concatenation if chat template fails
            print(f"Warning: Chat template failed, using fallback: {e}")
            text = row["prompt"] + "\n" + row["completion"]
            return {**row, "text": text}
    
    return dataset.map(apply_template)

def validate_tokenization_consistency(dataset: Dataset, tokenizer) -> bool:
    """
    Validate that prompt tokenization matches the start of full text tokenization.
    This helps catch tokenization mismatch issues early.
    """
    print("Validating tokenization consistency...")
    
    for i, row in enumerate(dataset):
        if i >= 5:  # Check first 5 examples
            break
            
        prompt = row["prompt"]
        full_text = row["text"]
        
        # Tokenize prompt and full text separately
        prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        
        prompt_len = prompt_tokens["input_ids"].shape[1]
        full_start = full_tokens["input_ids"][0, :prompt_len]
        
        # Check if prompt tokens match the start of full text tokens
        if not torch.equal(prompt_tokens["input_ids"][0], full_start):
            print(f"❌ Tokenization mismatch in example {i}:")
            print(f"   Prompt: {prompt[:100]}...")
            print(f"   Full text: {full_text[:100]}...")
            print(f"   Prompt tokens: {prompt_tokens['input_ids'][0]}")
            print(f"   Full start tokens: {full_start}")
            print(f"   Chat template used: {tokenizer.chat_template is not None}")
            return False
    
    print("✅ Tokenization consistency validated successfully!")
    return True

# -------------------------
# Model / Tokenizer / LoRA
# -------------------------
def load_model_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right", trust_remote_code=True)
    
    # Fix: Ensure consistent tokenization behavior
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Fix: Set add_eos_token to False to prevent automatic EOS token addition
    # This helps maintain consistency between prompt and full text tokenization
    tok.add_eos_token = False
    
    # Fix: Ensure proper padding token handling
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tok

def infer_lora_targets_from_model(model) -> List[str]:
    # Lightweight heuristic to work across LLaMA/Mistral/Gemma/Qwen/Phi families
    candidates = [
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        "query_key_value",
        "Wqkv","wo","wi","w1","w2","w3"
    ]
    present = set()
    for n, _ in model.named_modules():
        for c in candidates:
            if n.endswith(c):
                present.add(c)
    if not present:
        present = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    return sorted(present)

def build_lora_config(model):
    targets = infer_lora_targets_from_model(model)
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"
    )

# -------------------------
# Trainer / Evaluation
# -------------------------
def make_sft_trainer(model, tok, train_ds: Dataset, eval_ds: Dataset, outdir: str, epochs: int):
    # Per TRL docs, use TrainingArguments + SFTConfig together
    targs = SFTConfig(
        output_dir=outdir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=4092,
        dataset_text_field="text",  # Explicitly set the text field
        packing=False,  # Disable packing to avoid tokenization issues
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=targs,
        peft_config=None,  # we pass in a PEFT-wrapped model
        processing_class=tok
    )
    return trainer

@torch.no_grad()
def evaluate_unified(model, tok, eval_ds: Dataset, max_new_tokens=16) -> Dict[str, float]:
    """
    One unified evaluation over the entire eval set (mixed benchmarks).
    Acc = mean of per-item correctness using task-appropriate post-processing.
    """
    model.eval()
    correct, total = 0, 0
    detailed_results = []

    def postproc(pred_text: str, gold: str, bench: str) -> bool:
        text = pred_text
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()
        if bench in ("mmlu", "ai2_arc"):
            letters = re.findall(r"\b([ABCD])\b", text.upper())
            pred = letters[-1] if letters else text.strip().upper()
            return (pred == gold.upper())
        else:  # gsm8k
            gp = re.findall(r"[-]?\d+(?:\.\d+)?", text.replace(",", ""))
            gg = re.findall(r"[-]?\d+(?:\.\d+)?", gold.replace(",", ""))
            return (len(gp) > 0 and len(gg) > 0 and gp[-1] == gg[-1])

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=1.0,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )

    for r in eval_ds:
        prompt = r["prompt"]; gold = str(r["completion"]).strip(); bench = r["benchmark"]
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, generation_config=gen_cfg)
        text = tok.decode(out[0], skip_special_tokens=True)
        ok = postproc(text, gold, bench)
        correct += int(ok); total += 1
        
        # Store detailed result for each example
        detailed_results.append({
            "benchmark": bench,
            "prompt": prompt,
            "gold": gold,
            "prediction": text,
            "correct": ok
        })

    return {
        "accuracy": correct / max(1, total), 
        "n": total,
        "detailed_results": detailed_results
    }

# -------------------------
# Orchestrator
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", type=str, required=True)
    ap.add_argument("--evalset", type=str, required=True)
    ap.add_argument("--output_root", type=str, default="./runs_unified")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--models", nargs="+", required=True,
                    help="e.g. Qwen/Qwen3-4B meta-llama/Llama-3.1-8B google/gemma-3-4b-pt mistralai/Mistral-7B-v0.3 microsoft/phi-4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    # 1) Preprocess full (no per-benchmark split)
    ds_train = preprocess_mixed(args.trainset)
    ds_eval  = preprocess_mixed(args.evalset)

    results = {}
    for model_name in args.models:
        print(f"\n=== MODEL: {model_name} ===")
        base_model, tok = load_model_tokenizer(model_name)
        
        # Fix: Apply chat template to ensure consistent formatting
        print(f"Applying chat template for {model_name}...")
        ds_train_with_text = apply_chat_template_to_dataset(ds_train, tok)
        ds_eval_with_text = apply_chat_template_to_dataset(ds_eval, tok)
        
        lora_cfg = build_lora_config(base_model)

        # fresh PEFT wrapping, train on FULL train set
        model = get_peft_model(base_model, lora_cfg)
        outdir = os.path.join(args.output_root, model_name.replace("/","__"))
        os.makedirs(outdir, exist_ok=True)

        trainer = make_sft_trainer(model, tok, ds_train_with_text, ds_eval_with_text, outdir, args.epochs)
        trainer.train()

        # 3) Merge LoRA → base weights, save merged
        print("[Merge] merge_and_unload...")
        merged = model.merge_and_unload()   # per PEFT LoRA docs
        merged.save_pretrained(os.path.join(outdir, "merged"))
        tok.save_pretrained(os.path.join(outdir, "merged"))

        # 4) Unified evaluation on FULL eval set
        print("[Eval] unified eval on full evaluation set...")
        metrics = evaluate_unified(merged, tok, ds_eval_with_text)
        results[model_name] = metrics
        print(f"[Eval][{model_name}] Accuracy={metrics['accuracy']:.4f} (n={metrics['n']})")
        
        # Save detailed evaluation results to model-specific directory
        eval_results_path = os.path.join(outdir, "evaluation_results.json")
        with open(eval_results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[Eval][{model_name}] Detailed results saved to {eval_results_path}")
        
        # Save per-benchmark breakdown
        benchmark_results = {}
        for result in metrics["detailed_results"]:
            bench = result["benchmark"]
            if bench not in benchmark_results:
                benchmark_results[bench] = {"correct": 0, "total": 0, "examples": []}
            benchmark_results[bench]["total"] += 1
            if result["correct"]:
                benchmark_results[bench]["correct"] += 1
            benchmark_results[bench]["examples"].append(result)
        
        # Calculate per-benchmark accuracy
        for bench in benchmark_results:
            benchmark_results[bench]["accuracy"] = benchmark_results[bench]["correct"] / benchmark_results[bench]["total"]
        
        benchmark_path = os.path.join(outdir, "benchmark_breakdown.json")
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        print(f"[Eval][{model_name}] Benchmark breakdown saved to {benchmark_path}")

        # free memory
        del model, merged, base_model
        torch.cuda.empty_cache()

    # Write summary
    summary_path = os.path.join(args.output_root, "summary_unified.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY (Unified) ===")
    print(json.dumps(results, indent=2))
    print(f"[OK] Wrote {summary_path}")

if __name__ == "__main__":
    main()
