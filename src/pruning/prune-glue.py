#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
This file is adapted from the popular `run_glue.py` script from the `transformers` library. 
It prunes the model by "training" 0-1 masks on the attention heads and MLP layers of the model.
"""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import warnings
import math
import torch
import pickle
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.optim import AdamW

from modeling_pert import PertForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.36.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    glue_path: Optional[str] = field(
        default="glue",
        metadata={"help": "The path to GLUE."},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    start_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial sparsity of the model."}
    )
    target_sparsity: Optional[float] = field(
        default=0.5,
        metadata={"help": "The target sparsity of the model."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "The number of steps to ramp up the target sparsity to the specified level."}
    )
    reg_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term lambdas."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of warmup to use for the regularization term."}
    )

@dataclass
class ModelArguments:
    initialize_from: str = field(
        metadata={"help": "Path to model underlying the PERT model."}
    )
    ref_initialize_from: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model underlying the reference model."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    avg_activation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the average activations of the model."}
    )

class DataCollatorGLUE:
    def __init__(
        self, 
        tokenizer,
        max_length,
        key1,
        key2,
        is_regression
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key1 = key1
        self.key2 = key2
        self.is_regression = is_regression

    def __call__(self, examples):
        if self.key2 is None:
            batch = self.tokenizer(
                [e[self.key1] for e in examples], 
                padding="max_length", 
                max_length=self.max_length, 
                truncation=True, 
                return_tensors="pt"
            )
        else:
            batch = self.tokenizer(
                [e[self.key1] for e in examples], 
                [e[self.key2] for e in examples], 
                padding="max_length", 
                max_length=self.max_length, 
                truncation=True, 
                return_tensors="pt"
            )
        
        if self.is_regression:
            batch["labels"] = torch.tensor([e["label"] for e in examples], dtype=torch.float)
        else:
            batch["labels"] = torch.tensor([e["label"] for e in examples], dtype=torch.long)

        return batch    

class PertTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.target_sparsity = kwargs.pop('target_sparsity', 0.0)
        self.start_sparsity = kwargs.pop('start_sparsity', 0.0)
        self.num_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.ref_model = kwargs.pop('ref_model', None)
        super().__init__(*args, **kwargs)

    def get_current_target_sparsity(self, global_step):
        if global_step < self.num_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return self.start_sparsity + (self.target_sparsity - self.start_sparsity) * global_step / self.num_sparsity_warmup_steps
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_sparsity) + (math.log(1 - self.target_sparsity) - math.log(1 - self.start_sparsity)) * global_step / self.num_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_sparsity

    def compute_loss(self, model, inputs, return_outputs=False):   
        outputs = model(**inputs, target_sparsity=self.get_current_target_sparsity(self.state.global_step))
        
        zs_loss = outputs.zs_loss
        logits = outputs.logits
        
        with torch.no_grad():
            ref_logits = self.ref_model(**inputs).logits
        
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        ref_logits = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        
        # Use a KL loss, since we want faithulness above all
        kl_loss = torch.nn.functional.kl_div(logits, ref_logits, reduction='batchmean', log_target=True)
        
        loss = zs_loss + kl_loss

        return (loss, outputs) if return_outputs else loss

def get_optimizers(model, lr, reg_lr, num_training_steps, warmup_steps=0):
    optimizer_1_group = []
    optimizer_2_group = []

    for n, p in model.named_parameters():
        if 'log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda' in n:
            optimizer_2_group.append(p)
    
    optimizer = AdamW(
        [
            {
                'params': optimizer_1_group,
            },
            {
                'params': optimizer_2_group,
                'maximize': True,   # The regularization lambdas try to maximize the penalty
                'lr': reg_lr
            } 
        ],
        lr=lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler

def freeze_all_expecting_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def load_avg_activations(model, avg_activation_path, device):
    avg_activations = pickle.load(open(avg_activation_path, "rb"))
    for n, m in model.named_modules():
        if n in avg_activations:
            m.set_avg_activation(torch.from_numpy(avg_activations[n]).to(device))

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if os.path.exists(data_args.glue_path):
        raw_datasets = load_from_disk(os.path.join(data_args.glue_path, data_args.task_name))
    else:
        raw_datasets = load_dataset(data_args.glue_path, data_args.task_name)

    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Don't pass num_labels -- the checkpoint is expected to have taken care of it
    model = PertForSequenceClassification.from_pretrained(model_args.initialize_from)
    if model_args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.initialize_from)
    if model_args.avg_activation_path is not None:
        load_avg_activations(model, model_args.avg_activation_path, training_args.device)
    freeze_all_expecting_pruning_params(model)
    
    ref_path = model_args.ref_initialize_from if model_args.ref_initialize_from is not None else model_args.initialize_from
    ref_model = PertForSequenceClassification.from_pretrained(ref_path).to("cuda")

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    collator = DataCollatorGLUE(
        tokenizer=tokenizer,
        max_length=max_seq_length,
        key1=sentence1_key,
        key2=sentence2_key,
        is_regression=is_regression
    )

    optimizers = get_optimizers(
        model=model,
        lr=training_args.learning_rate,
        reg_lr=data_args.reg_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps
    )

    metric = evaluate.load("glue", data_args.task_name)

    def eval_fn(p: EvalPrediction):
        preds, _, zs_loss, target_sparsity, model_sparsity, sc_loss = p.predictions

        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        result = {
            "eval_"+k : v 
            for k, v in result.items()
        }
        
        result["zs_loss"] = zs_loss.mean().item()
        result["sc_loss"] = sc_loss.mean().item()
        result["target_sparsity"] = target_sparsity.mean().item()
        result["model_sparsity"] = model_sparsity.mean().item()
        return result

    # Initialize our Trainer
    trainer = PertTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=optimizers,
        start_sparsity=data_args.start_sparsity,
        target_sparsity=data_args.target_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()