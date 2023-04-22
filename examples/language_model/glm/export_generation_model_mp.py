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

from paddlenlp.transformers import GLMForConditionalGeneration
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger
from utils import load_model


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
        default="output_generate/splits_mp_01_sharding_01_500/",
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

def main():
    args = parse_args()
    token_path = "/root/.paddlenlp/models/THUDM/glm-10b-chinese"
    use_fp16 = args.data_type == "float16"
    if use_fp16:
        paddle.set_default_dtype("float16")
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    # model = AutoModelForConditionalGeneration.from_pretrained(args.model_path)
    path = "./glm_10b_mp/"
    model = load_model(path, GLMForConditionalGeneration, dtype=args.data_type)
    logger.info("load model done")

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

    current_rank = paddle.distributed.get_rank()
    output_path = f"{args.output_path}_mp{current_rank}"
    # # Save converted static graph model
    paddle.jit.save(model, output_path)
    # # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
