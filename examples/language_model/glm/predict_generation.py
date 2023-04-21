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

import paddle
import os

from paddle.distributed import fleet
from paddle import LazyGuard
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer
from paddlenlp.transformers import GLMConfig, PretrainedModel, GLMForConditionalGeneration

FUSE_MT = os.getenv("FUSE_MT") == "1"

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", required=True, help="The directory of model.")
    parser.add_argument("--model_path", default="/root/.paddlenlp/models/THUDM/glm-large-chinese/", help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=20, help="The batch size of data.")
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts

def get_state_dict(config, state_dict, model):
    embed_dim = config.hidden_size
    num_attention_heads = config.num_attention_heads // config.tensor_parallel_degree
    head_dim = embed_dim // num_attention_heads
    new_state_dict = {}
    dtype = "float16" if model.glm.word_embeddings.weight.dtype.name == "FP16" else "float32"
    for k, v in state_dict.items():
        if k.startswith("glm.word_embeddings"):
            model.glm.word_embeddings.weight.set_value(v.astype(dtype))
            continue
        elif k.startswith("glm.transformer.position_embeddings"):
            model.glm.transformer.position_embeddings.weight.set_value(v.astype(dtype))
            continue
        elif k.startswith("glm.transformer.final_layernorm.weight"):
            model.glm.transformer.final_layernorm.weight.set_value(v.astype(dtype))
            continue
        elif k.startswith("glm.transformer.final_layernorm.bias"):
            model.glm.transformer.final_layernorm.bias.set_value(v.astype(dtype))
            continue
        elif k.endswith("glm.transformer.block_position_embeddings.weight"):
            model.glm.transformer.block_position_embeddings.weight.set_value(v.astype(dtype))
            continue
        idx = int(k.split(".")[3])
        if k.endswith("input_layernorm.weight"):
            new_state_dict["fusemt.{}.ln_scale".format(idx)] = v
            model.glm.transformer.transformer_block.ln_scales[idx].set_value(v.astype("float32"))
        elif k.endswith("input_layernorm.bias"):
            new_state_dict["fusemt.{}.ln_bias".format(idx)] = v
            model.glm.transformer.transformer_block.ln_biases[idx].set_value(v.astype("float32"))
        elif k.endswith("attention.query_key_value.weight"):
            new_state_dict["fusemt.{}.qkv_weight".format(idx)] = v.reshape([embed_dim, 3, num_attention_heads,  head_dim])
            model.glm.transformer.transformer_block.qkv_weights[idx].set_value(v.reshape([embed_dim, 3, num_attention_heads,  head_dim]))
        elif k.endswith("attention.query_key_value.bias"):
            new_state_dict["fusemt.{}.qkv_bias".format(idx)] = v.reshape([3, num_attention_heads, head_dim])
            model.glm.transformer.transformer_block.qkv_biases[idx].set_value(v.reshape([3, num_attention_heads, head_dim]))
        elif k.endswith("attention.dense.weight"):
            new_state_dict["fusemt.{}.linear_weight".format(idx)] = v
            model.glm.transformer.transformer_block.linear_weights[idx].set_value(v)
        elif k.endswith("attention.dense.bias"):
            new_state_dict["fusemt.{}.linear_bias".format(idx)] = v
            model.glm.transformer.transformer_block.linear_biases[idx].set_value(v)
        elif k.endswith("post_attention_layernorm.weight"):
            new_state_dict["fusemt.{}.ffn_ln_scale".format(idx)] = v
            model.glm.transformer.transformer_block.ffn_ln_scales[idx].set_value(v.astype("float32"))
        elif k.endswith("post_attention_layernorm.bias"):
            new_state_dict["fusemt.{}.ffn_ln_bias".format(idx)] = v
            model.glm.transformer.transformer_block.ffn_ln_biases[idx].set_value(v.astype("float32"))
        elif k.endswith("mlp.dense_h_to_4h.weight"):
            new_state_dict["fusemt.{}.ffn1_weight".format(idx)] = v
            model.glm.transformer.transformer_block.ffn1_weights[idx].set_value(v)
        elif k.endswith("mlp.dense_h_to_4h.bias"):
            new_state_dict["fusemt.{}.ffn1_bias".format(idx)] = v
            model.glm.transformer.transformer_block.ffn1_biases[idx].set_value(v)
        elif k.endswith("mlp.dense_4h_to_h.weight"):
            new_state_dict["fusemt.{}.ffn2_weight".format(idx)] = v
            model.glm.transformer.transformer_block.ffn2_weights[idx].set_value(v)
        elif k.endswith("mlp.dense_4h_to_h.bias"):
            new_state_dict["fusemt.{}.ffn2_bias".format(idx)] = v
            model.glm.transformer.transformer_block.ffn2_biases[idx].set_value(v)
        else:
            print("Unknow weight {}".format(k))
    return new_state_dict

def load_model(model_name_or_path: str, model_class: Type[PretrainedModel]):
    config = GLMConfig.from_pretrained(model_name_or_path)
    # dtype = "float32" if config.dtype is None else config.dtype
    dtype = "float16"
    paddle.set_default_dtype(dtype)

    # Detecting last checkpoint.
    config["enable_fuse_transformer"] = False
    config["use_cache"] = True
    config["use_pure_fp16"] = False

    # TODO(wj-Mcat): only support `mp_degree`, so world_size is equal to `world_size`
    world_size = paddle.distributed.get_world_size()

    if world_size == 1:
        with LazyGuard():
            model = model_class.from_pretrained(
                model_name_or_path, config=config,
                dtype=dtype,
                )

        weight_file = os.path.join(model_name_or_path, f"model_state.pdparams")
        state_dict = paddle.load(weight_file, return_numpy=True)
        
        if FUSE_MT:
            get_state_dict(config, state_dict, model)

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

    # support shard state_dict
    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"sharding model weight file<auto_dist{mp_rank}.pdparams> not found under <{model_name_or_path}>"
        )
    state_dict = paddle.load(weight_file, return_numpy=True)

    model.set_state_dict(state_dict)
    return model


class Predictor(object):
    def __init__(self, args):
        # self.tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained("/root/.paddlenlp/models/THUDM/glm-large-chinese/")
        self.batch_size = args.batch_size
        self.args = args
        path = args.model_path
        model = load_model(path, GLMForConditionalGeneration)
        self.model = model
        # self.model = AutoModelForConditionalGeneration.from_pretrained(args.model_path)
        # self.model.glm.init_weight_fuse_mt()
        self.model.eval()

    def preprocess(self, input_text):
        input_text = [text.strip() + "[gMASK]" for text in input_text]
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            add_special_tokens=True,
            padding=True,
            max_length=self.args.src_length,
            truncation=True,
            truncation_side="left",
        )
        inputs = self.tokenizer.build_inputs_for_generation(inputs, max_gen_length=self.args.tgt_length)
        inputs_tensor = {}
        for key, value in inputs.items():
            inputs_tensor[key] = paddle.to_tensor(value)
        return inputs_tensor

    def infer(self, inputs):
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.args.tgt_length,
            eos_token_id=self.tokenizer.eop_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    import random
    import numpy as np
    paddle.seed(100)
    random.seed(100)
    np.random.seed(100)
    
    args = parse_arguments()
    predictor = Predictor(args)
    all_texts = [
        "答案：年基准利率4.35%，上下文：从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。在已知答案的前提下，问题：",
        "答案：U系列，上下文：U系列是最好的，采用国际顶尖技术（由格力自主研发）双级变频压缩机，提高压缩机运转效率，制冷制热能力更强劲；1赫兹变频技术，使空调相当于一个15 W电灯泡，更加节能省电；送风面积广，风力大；生态风，净化空气。非常不错，现在国美在做活动，可以了解一下。在已知答案的前提下，问题：",
    ]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{}\n{}".format(text, result))
