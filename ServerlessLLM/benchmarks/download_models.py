# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

############## SLLM-CONDENSE #########
from sllm_store.transformers import save_model, save_model_condense


def get_args():
    parser = argparse.ArgumentParser(
        description="Save a model with ServerlessLLM"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to save",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        required=True,
        choices=["sllm", "safetensors", "sllm-condense"],
        help="Format to save the model in",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save models",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of replicas to save",
    )
    ############## SLLM-CONDENSE #########
    parser.add_argument(
        "--fptc-dir",
        type=str,
        default=os.getenv(
            "SLLM_CONDENSE_FPTC_DIR",
            "/home/ben046/projects/TensorProcessing/generate_compressed/comp",
        ),
        help="Directory containing per-tensor .fptc packages for sllm-condense",
    )
    return parser.parse_args()


def main():
    args = get_args()

    save_dir = args.save_dir
    save_format = args.save_format
    model_name = args.model_name
    replicas = args.num_replicas

    # Check if save_dir exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory {save_dir} does not exist")

    # Load model into memory
    print(f"Loading model {model_name} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # # Save model
    ############## SLLM-CONDENSE #########
    if save_format in ("sllm", "sllm-condense"):
        print(f"Saving {replicas} {save_format} models to {save_dir}")
        for i in tqdm(range(replicas)):
            model_suffix = (
                f"{model_name}_{i}"
                if save_format == "sllm"
                else f"{model_name}_sllm-condense_{i}"
            )
            model_dir = os.path.join(save_dir, model_suffix)
            if save_format == "sllm":
                save_model(model, model_dir)
            else:
                save_model_condense(model, model_dir, args.fptc_dir)
            tokenizer.save_pretrained(model_dir)
    elif save_format == "safetensors":
        print(f"Saving {replicas} safetensors models to {save_dir}")
        for i in tqdm(range(replicas)):
            model_dir = os.path.join(save_dir, f"{model_name}_safetensors_{i}")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
    else:
        raise ValueError(f"Invalid save format {save_format}")

    print("Done!")


if __name__ == "__main__":
    main()
