# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import paddle
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from paddle.distributed import fleet
from paddle import LazyGuard

from paddlenlp.transformers import GLMConfig, PretrainedModel, GLMForConditionalGeneration
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer
from paddlenlp.utils.log import logger
from utils import update_word_embedding_weights

FUSE_MT = os.getenv("FUSE_MT") == "1"

def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="THUDM/glm-large-chinese",
        type=str,
        # required=True,
        help="Model type selected in the list",
    )
    parser.add_argument(
        "--data_type",
        default="float32",
        type=str,
        # required=True,
        help="datatype of the program: now support[float32 / float16]",
    )
    parser.add_argument(
        "--model_path",
        default="/root/.paddlenlp/models/THUDM/glm-large-chinese",
        type=str,
        required=False,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/glm",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def load_model(model_name_or_path: str, model_class: Type[PretrainedModel], dtype="float32"):
    config = GLMConfig.from_pretrained(model_name_or_path)
    paddle.set_default_dtype(dtype)

    # Detecting last checkpoint.
    config["enable_fuse_transformer"] = False
    config["use_cache"] = True
    config["use_pure_fp16"] = False

    # TODO(wj-Mcat): only support `mp_degree`, so world_size is equal to `world_size`
    world_size = paddle.distributed.get_world_size()

    if world_size == 1:
        with LazyGuard():
            model = model_class.from_pretrained(model_name_or_path, config=config)
        weight_file = os.path.join(model_name_or_path, f"model_state.pdparams")
        state_dict = paddle.load(weight_file, return_numpy=True)
    
        if FUSE_MT:
            update_word_embedding_weights(config, state_dict, model)
        return model
        

    # start to init distributed env
    strategy = fleet.DistributedStrategy()

    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": world_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }

    seed = 1002
    # Set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

    fleet.init(is_collective=True, strategy=strategy)

    # Obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()

    config["tensor_parallel_rank"] = mp_rank
    with LazyGuard():
        # init the model without initialized parameters
        model = model_class(config=config)

    weight_file = os.path.join(model_name_or_path, f"model_state.tp{mp_rank:0>2d}.pdparams")
    logger.info(f"start to loading sharding model weight file<{weight_file}>")

    # support shard state_dict
    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"sharding model weight file<auto_dist{mp_rank}.pdparams> not found under <{model_name_or_path}>"
        )
    state_dict = paddle.load(weight_file, return_numpy=True)

    model.set_state_dict(state_dict)
    return model

def main():
    args = parse_args()
    token_path = "/root/.paddlenlp/models/THUDM/glm-large-chinese"
    use_fp16 = args.data_type == "float16"
    if use_fp16:
        paddle.set_default_dtype("float16")
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    # model = AutoModelForConditionalGeneration.from_pretrained(args.model_path)
    model = load_model(args.model_path, GLMForConditionalGeneration, dtype=args.data_type)
    

    model.eval()
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64"),
        paddle.static.InputSpec(shape=[None, None, None], dtype="int64"),  # pos_ids
        # max_length
        20,
        # min_length
        0,
        # decode_strategy
        "sampling",
        # temperature
        1.0,
        # top_k
        1,
        # top_p
        1.0,
        1.0,
        # repetition_penalty
        1,
        # num_beam_groups
        1,
        0.0,
        # early_stopping
        False,
        # bos_token_id
        tokenizer.sop_token_id,
        # eos_token_id
        tokenizer.eop_token_id,
        # pad_token_id
        tokenizer.pad_token_id,
    ]
    model = paddle.jit.to_static(model.generate, input_spec=input_spec)

    # # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
