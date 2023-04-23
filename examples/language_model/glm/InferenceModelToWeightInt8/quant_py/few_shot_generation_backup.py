#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""ERNIE pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import json
import argparse
import numpy as np
import multiprocessing
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle.fluid.profiler as profiler
import paddle.fluid.contrib.mixed_precision.fp16_utils as fp16_utils
# import paddle.static.amp.fp16_utils as fp16_utils

import copy
from utils.decode import post_process, word_count
from metrics import metric

from reader.few_shot_reader import GenerationReader, GenerationControllableReader, GenerationMultiPromptReader
from model.ernie import ErnieModel, ErnieConfig
from utils.args import print_arguments
from utils.io import init_checkpoint, init_pretraining_params, checkpoint_rearrange, save_checkpoint, save_inference_model
from finetune_args import parser
from utils.topo import Topology
from edl_checkpoint_utils import _get_file_md5
from utils.inference_pipeline import HybridParallelInferenceHelper as HybridParallelInferenceHelper
from reader.ngram_mask import load_ngram_mask_ids


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

USE_LOCAL_HPI = True

os.environ['FLAGS_enable_parallel_graph'] = "0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.1"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
os.environ['FLAGS_check_nan_inf'] = "0"

paddle.enable_static()
fleet.init(is_collective=True)
np.set_printoptions(threshold=1e6)
args = parser.parse_args()

float_type = "float16" if args.use_amp else "float32"
if paddle.is_compiled_with_cuda():
    device = "gpu"
    ascend = False
    int_type = "int64"
    device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
elif paddle.is_compiled_with_npu():
    device = "npu"
    ascend = True
    int_type = 'int32'
    device_id = int(os.environ.get('FLAGS_selected_npus', 0))
else:
    raise Exception('paddle must compiled with cuda or npu')

# yapf: enable.


def create_model(pyreader_name, ernie_config, task_group, topo, acc_steps, program):
    shapes = [[-1, args.max_seq_len], [-1, args.max_seq_len],
              [-1, args.max_seq_len, args.max_seq_len + args.prompt_num * int(args.use_prefix_tuning)], [-1, 1]]
    dtypes = [int_type, int_type, float_type, int_type]
    names = ['src_ids', 'pos_ids', 'input_mask', 'qids']

    if args.use_2d_pos:
        # for GLM tasks
        shapes.append([-1, args.max_seq_len])
        dtypes.append(int_type)
        names.append("pos_ids_extra")

    if args.use_rope:
        shapes.append([2, -1, 1, args.max_seq_len, ernie_config["hidden_size"] // ernie_config["num_attention_heads"]])
        dtypes.append(float_type)
        names.append("rotary_pos_emb")  

    if args.use_prefix_tuning:
        # [2, -1, mp_n_head, max_seq_len, head_size],
        # for prefix-tuning (bsz, n_layers, 2, prompt_num, hidden_dim)
        shapes.append([-1, ernie_config["num_hidden_layers"], 2, args.prompt_num // topo.mp.size, ernie_config["hidden_size"]])
        dtypes.append(float_type)
        names.append('prefix_caches')

    if args.use_ngram_mask:
        ngram_mask_ids = load_ngram_mask_ids(args.vocab_path, args.ngram_mask_filepath)
        shapes.append(ngram_mask_ids.shape)
        dtypes.append("int32")
        names.append('ngram_mask_ids')

    # append tgt data
    if "topk_sampling" == args.decoding_strategy:
        extra_shapes = [[-1, 1], [-1, 1], [-1, 1], [-1], [-1, 1, args.max_seq_len + args.prompt_num * int(args.use_prefix_tuning)], [1], [-1], [-1, 1], [-1, 1], [-1, ernie_config['vocab_size_output']], [-1, 1], [-1, ernie_config['vocab_size_output']]]
        extra_dtypes = [int_type, int_type, "float32", int_type, "float32", int_type, int_type, float_type, float_type, float_type, int_type, float_type]
        extra_names = ['tgt_ids', 'tgt_pos', 'init_score', 'parent_idx',
        'tgt_generation_mask', 'max_dec_len', 'topk', 'temperature', 'penalty_score', 'token_penalty', 'min_dec_len', 'min_dec_token_penalty']
        extra_lod_levels = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]        
    elif "topk_topp_sampling" == args.decoding_strategy:
        extra_shapes = [[-1, 1], [-1, 1], [-1, 1], [-1], [-1, 1, args.max_seq_len + args.prompt_num * int(args.use_prefix_tuning)], [1], [-1], [-1, 1], [-1, 1], [-1, 1], [-1, ernie_config['vocab_size_output']], [-1, 1], [-1, ernie_config['vocab_size_output']]]
        extra_dtypes = [int_type, int_type, "float32", int_type, "float32", int_type, int_type, float_type, float_type, float_type, float_type, int_type, float_type]
        extra_names = ['tgt_ids', 'tgt_pos', 'init_score', 'parent_idx',
        'tgt_generation_mask', 'max_dec_len', 'topk', 'topp', 'temperature', 'penalty_score', 'token_penalty', 'min_dec_len', 'min_dec_token_penalty']
        extra_lod_levels = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif "topp_sampling" == args.decoding_strategy:
        extra_shapes = [[-1, 1], [-1, 1], [-1, 1], [-1], [-1, 1, args.max_seq_len + args.prompt_num * int(args.use_prefix_tuning)], [1], [-1, 1], [-1, 1], [-1, 1], [-1, ernie_config['vocab_size_output']], [-1, 1], [-1, ernie_config['vocab_size_output']]]
        extra_dtypes = [int_type, int_type, "float32", int_type, "float32", int_type, float_type, float_type, float_type, float_type, int_type, float_type]
        extra_names = ['tgt_ids', 'tgt_pos', 'init_score', 'parent_idx',
        'tgt_generation_mask', 'max_dec_len', 'topp', 'temperature', 'penalty_score', 'token_penalty', 'min_dec_len', 'min_dec_token_penalty']
        extra_lod_levels = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]        
    else:
        assert False, args.decoding_strategy


    if args.use_2d_pos:
        extra_shapes.append([-1, 1])
        extra_dtypes.append(int_type)
        extra_names.append('tgt_pos_extra')
        extra_lod_levels.append(2)
    # if args.is_mask_out_src_tokens or args.is_constraint_within_src_tokens:
    #     extra_shapes.append([-1, 30000])
    #     extra_dtypes.append(float_type)
    #     extra_names.append('gen_token_mask')
    #     extra_lod_levels.append(0)

    if args.use_batch_idx:
        extra_shapes.append([-1])
        extra_dtypes.append("int32")
        extra_names.append('batch_idx')
        extra_lod_levels.append(0)

    with fluid.device_guard(f"{device}:all"):
        inputs = [paddle.static.data(name=names[i], shape=shapes[i], dtype=dtypes[i]) for i in range(len(names))]
        extra_inputs = [paddle.static.data(name=extra_names[i], shape=extra_shapes[i], dtype=extra_dtypes[i],
                                           lod_level=extra_lod_levels[i]) for i in range(len(extra_names))]
        pyreader = None
        if not args.save_inference_model_then_exist:
            pyreader = fluid.io.DataLoader.from_generator(
                    feed_list=inputs + extra_inputs,
                    capacity=70, iterable=False, use_double_buffer=True)
            
    
    pos_ids_extra = None
    prefix_caches = None
    rotary_pos_emb = None
    src_ids, pos_ids, input_mask, qids = inputs[:4]
    if args.use_2d_pos:
        pos_ids_extra = inputs[4]
        if args.use_rope:
            rotary_pos_emb = inputs[5]
            if args.use_prefix_tuning:
                prefix_caches = inputs[6]
        else:
            if args.use_prefix_tuning:
                prefix_caches = inputs[5]
    else:
        if args.use_rope:
            rotary_pos_emb = inputs[4]
            if args.use_prefix_tuning:
                prefix_caches = inputs[5]
        else:
            if args.use_prefix_tuning:
                prefix_caches = inputs[4]

    print(inputs)

    with fluid.device_guard(f"{device}:0"):
        ernie = ErnieModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            input_mask=input_mask,
            max_dec_len=extra_inputs[5],
            eff_mask=None,
            eff_ratio=None,
            config=ernie_config,
            mem_len=args.mem_len,
            weight_sharing=args.weight_sharing,
            use_fp16=args.use_amp,
            need_cal_loss=None,
            is_training_server=args.training_server,
            use_vars=args.use_vars,
            topo=topo,
            is_nlg=None,
            is_nlu=None,
            fuse=args.fuse,
            device=device,
            use_2d_pos=args.use_2d_pos,
            args=args,
            program=program,
            position_ids_extra=pos_ids_extra,
            is_generation_task=True,
            prefix_caches=prefix_caches,
            rotary_pos_emb=rotary_pos_emb)

        generator_inputs_dict = {
           variable.name: variable for variable in inputs + extra_inputs
        }
        outputs = ernie.generator.inference(generator_inputs_dict)
        checkpoints = ernie.get_checkpoints()
        mems, new_mems = ernie.get_mem_output()
        
        graph_vars = [qids, outputs['finished_ids'], outputs['finished_scores']]

    feed_vars = [src_ids, pos_ids, input_mask]
    fetch_vars = {"graph_vars": graph_vars,
                  "pp_total_loss": None,
                  "checkpoints": checkpoints}
    feed_names = names + extra_names
    return pyreader, fetch_vars, mems, new_mems, feed_vars, feed_names


def debug_program(name, program):
    with open("{}.txt.{}".format(name, device_id), 'w') as f:
        f.write(str(program))


class PassUtils(object):
    passes = ['fuse_relu_depthwise_conv_pass',
            'fuse_bn_act_pass',
             'fuse_bn_add_act_pass',
             #'fusion_group_pass',                          # wrong
             'fuse_elewise_add_act_pass',

             # inference pass
             "is_test_pass",
             "simplify_with_basic_ops_pass",
             #"conv_affine_channel_fuse_pass",              # runtime
             #"conv_eltwiseadd_affine_channel_fuse_pass",   # runtime
             #"conv_bn_fuse_pass",                          # runtime
             #"conv_eltwiseadd_bn_fuse_pass",               # runtime
             #"embedding_eltwise_layernorm_fuse_pass",      # runtime
             #"multihead_matmul_fuse_pass_v2",              # runtime
             "gpu_cpu_squeeze2_matmul_fuse_pass",
             "gpu_cpu_reshape2_matmul_fuse_pass",
             "gpu_cpu_flatten2_matmul_fuse_pass",
             #"gpu_cpu_map_matmul_v2_to_mul_pass",          # runtime
             #"gpu_cpu_map_matmul_v2_to_matmul_pass",
             #"gpu_cpu_map_matmul_to_mul_pass",             # runtime
             #"fc_fuse_pass",                               # runtime
             #"fc_elementwise_layernorm_fuse_pass",         # runtime
             "transpose_flatten_concat_fuse_pass",

             #'runtime_context_cache_pass',
             'buffer_shared_inplace_pass',
             ]

    @staticmethod
    def apply_ir_passes(main_program):
        import paddle.fluid.core as core
        graph = core.Graph(main_program.desc)
        # graph attr set
        graph.set_not_owned('__param_scope__', fluid.global_scope())

        pass_builder = core.PassBuilder()
        for name in PassUtils.passes:
            ir_pass = pass_builder.append_pass(name)
            # set attr for pass
            ir_pass.set('use_cuda', True)
            ir_pass.set('use_gpu', True)

        trans_pass = pass_builder.append_pass('graph_to_program_pass')
        opt_program = fluid.Program()
        trans_pass.set_not_owned('program', opt_program.desc)
        for p in pass_builder.all_passes():
            p.apply(graph)
        opt_program.blocks = [
                paddle.fluid.framework.Block(opt_program, i)
                for i in range(opt_program.desc.num_blocks())]
        opt_program._sync_with_cpp()
        return opt_program

    @staticmethod
    def apply_program_pass(main_program, startup_program, pass_attrs):
        from paddle.fluid.ir import get_data_vars
        from paddle.fluid.framework import _apply_pass
        def update_attr(attrs, attr_types, name, value, typ=None):
            if name not in attrs:
                attrs[name] = value
            if typ:
                attr_types[name] = typ

        def apply_pass(name):
            attrs = dict(pass_attrs)
            attr_types = {}
            update_attr(attrs, attr_types, "nranks", 1, "size_t")
            update_attr(attrs, attr_types, "use_cuda", False, "bool")
            update_attr(attrs, attr_types, "mem_opt_skip_vars",
                        get_data_vars(main_program), "list[str]")
            _apply_pass(main_program, startup_program, name, attrs, attr_types)

        use_cuda = pass_attrs.get("use_cuda", False)
        for p in PassUtils.passes:
            apply_pass(p)


def train(args):
    def debug_config(ernie_config):
        n_layer = ernie_config["num_hidden_layers"]
        sharing_layer = ernie_config["num_sharing_layers"]
        task_branch_layer = n_layer - sharing_layer
        total_layer_in_fact = n_layer + task_branch_layer if args.mem_len > 0 else 0
        server_hidden_size = ernie_config["hidden_size"]
        task_hidden_size = ernie_config["branch_hidden_size"]
        print("the layers number of model is: ", n_layer)
        print("the layers number of server is: ", sharing_layer)
        print("the layers number of branch is: ", task_branch_layer)
        print("the hidden size of server is: ", server_hidden_size)
        print("the hidden size of branch is: ", task_hidden_size)
    print("pretraining start")
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    debug_config(ernie_config)

    with open(args.task_group_json) as f:
        task_group = json.load(f)

    print("args.is_distributed:", args.is_distributed)
    if args.is_distributed:
        # define distribution strategy
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.nccl_comm_num = 3
        dist_strategy.fuse_all_reduce_ops = args.fuse_all_reduce_ops
        dist_strategy.fuse_grad_size_in_MB = args.fuse_grad_size_in_MB
        dist_strategy.amp = args.use_amp
        white_list = ['gelu']
        black_list = ['layer_norm', 'softmax', 'log'] if device == "gpu" else ['layer_norm', 'softmax']
        # note: softmax must not be in black_list, will cause generation error for 105B model.
        black_list = ["beam_search_decode"]
        print(f"custom_white_list:{white_list}, custom_black_list: {black_list}")
        dist_strategy.amp_configs = {
            "custom_white_list": white_list,
            "custom_black_list": black_list,
            "init_loss_scaling": 32768, #32768,
            "decr_every_n_nan_or_inf": args.decr_every_n_nan_or_inf,
            "incr_every_n_steps": args.incr_every_n_steps,
            "incr_ratio": 2.0,
            "use_dynamic_loss_scaling": True,
            "decr_ratio": 0.5,
            "use_fp16_guard": False,
            "use_pure_fp16": False
        }
        dist_strategy.recompute = args.use_recompute
        dist_strategy.sharding = args.sharding
        dist_strategy.pipeline = args.num_pp > 1

        if args.fp16_allreduce:
            dist_strategy.fp16_allreduce = args.fp16_allreduce

        if args.fuse_grad_merge:
            dist_strategy.fuse_grad_merge = args.fuse_grad_merge

        if args.num_pp == 1 and args.num_dp == 1 and args.num_mp == 1:
            dist_strategy.without_graph_optimization = True
            dist_strategy.sharding = False

        # define topology structure for dp/pp/mp
        topo = Topology(rank=fleet.worker_index(),
                        world_size=fleet.worker_num(),
                        pp=args.num_pp,
                        dp=args.num_dp, #fleet.worker_num() // args.num_pp // args.num_mp,
                        mp=args.num_mp,
                        sharding=args.num_sharding)
        is_first = False
        is_last = False
        if topo.pp.rank == 0:
            is_first = True
        if topo.pp.rank == (topo.pp.size - 1):
            is_last = True
        dp_rank = topo.dp.rank * topo.sharding.size + topo.sharding.rank #topo.dp.rank
        dp_worldsize = topo.dp.size * topo.sharding.size
        bsz_per_dp = args.global_bsz // dp_worldsize

        micro_bsz = min(bsz_per_dp, args.micro_bsz)
        assert args.global_bsz >= micro_bsz * dp_worldsize, f"[micro bsz x dp_worldsize] larger than global bsz, global_bsz: {args.global_bsz} micro_bsz: {micro_bsz}, dp_worldsize: {dp_worldsize}"
        assert args.global_bsz % micro_bsz == 0, f"cannot do gradient accumulate, global_bsz: {args.global_bsz} micro_bsz: {micro_bsz}"
        acc_steps = bsz_per_dp // micro_bsz

        # sharding \ model parallel \ pipeline
        if args.num_mp == 1 and args.num_pp == 1 and args.num_sharding == 1:
            # single
            assert args.num_dp == 1, "normal data parallelism should not use sharding config"
            print(f'using sharding with mp and pp: {topo}')
        else:
            dist_strategy.sharding_configs = {"segment_broadcast_MB": 32,
                                              "sharding_degree": args.num_sharding,
                                              "mp_degree": args.num_mp,
                                              "pp_degree": args.num_pp,
                                              "dp_degree": args.num_dp,
                                              "gradient_merge_acc_step": acc_steps,
                                              "optimize_offload": False,
                                              "optimize_cast": args.optimizer_cast
                                             }
            dist_strategy.pipeline_configs = {"schedule_mode": "1F1B",
                                              "micro_batch_size": micro_bsz,
                                              "accumulate_steps": acc_steps,
                                             }
            print(f"using global_bsz: {args.global_bsz} micro_bsz: {micro_bsz}, acc_steps: {acc_steps}")
    else:
        dist_strategy = None

    if args.save_inference_model_then_exist or args.save_ckpt_then_exist:
        place = fluid.CPUPlace()
    else:
        if not ascend:
            place = fluid.CUDAPlace(device_id)
        else:
            place = fluid.NPUPlace(device_id)

    random_seed_dp = dp_rank + args.random_seed * args.random_seed_factor
    if args.task == "controllable_case":
        # controllable generation
        data_reader = GenerationControllableReader(
            trainer_id=dp_rank,
            trainer_num=dp_worldsize,
            random_seed=random_seed_dp,
            args=args,
            place=place)    
    elif args.task == "multi_prompt_task" or args.task == "finetune":
        data_reader = GenerationMultiPromptReader(
            trainer_id=dp_rank,
            trainer_num=dp_worldsize,
            random_seed=random_seed_dp,
            args=args,
            place=place,
            ernie_config=ernie_config)    
    else:
        data_reader = GenerationReader(
            trainer_id=dp_rank,
            trainer_num=dp_worldsize,
            random_seed=random_seed_dp,
            args=args,
            place=place,
            ernie_config=ernie_config)

    train_program, startup_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_program, startup_prog):
        with fluid.unique_name.guard():
            train_pyreader, fetch_vars, mems_train, new_mems_train, feed_vars, feed_names = create_model(
                pyreader_name='train_reader', ernie_config=ernie_config, task_group=task_group,
                topo=topo, acc_steps=acc_steps, program=train_program)
            graph_vars = fetch_vars["graph_vars"]

            if not args.save_inference_model_then_exist:
                train_pyreader.set_batch_generator(data_reader.data_generator(
                    input_file=args.input_file,
                    batch_size=args.batch_size,
                    epoch=1,
                    shuffle=False,
                    phase="dev"), place)

            # test fp16
            if args.use_amp:
                debug_program('fp16_origin_program', train_program)

                amp_lists = fp16_utils.AutoMixedPrecisionLists(
                    custom_white_list=set(white_list),
                    custom_black_list=set(black_list))
                from utils.inference_pipeline import cast_model_to_fp16_block
                fp16_var_names = cast_model_to_fp16_block(train_program, amp_lists, False)
                #fp16_var_names = fp16_utils.cast_model_to_fp16_block(train_program, amp_lists, False)
                #fp16_utils.rewrite_program(train_program, amp_lists)

    #if True:
    if False:
        PassUtils.apply_program_pass(train_program, startup_prog, {"use_cuda": True, 'use_gpu': True})
        #train_program = PassUtils.apply_ir_passes(train_program)

    debug_program('origin_program', train_program)

    if args.num_pp > 1:
        helper = HybridParallelInferenceHelper(
            startup_prog, train_program, num_mp=args.num_mp, num_pp=args.num_pp,
            micro_batch_size=micro_bsz,
            init_comm=fleet.worker_num() > 1 and not (args.save_ckpt_then_exist or args.save_inference_model_then_exist),
            beam_size=args.beam_size if args.decoding_strategy == "beam_search" else 1)
        print("USE_LOCAL_HPI", USE_LOCAL_HPI)

        sync_in_while_lastpp2firstpp_var_names = ['array_write_0.out', 'array_write_1.out', 'array_write_2.out', 'array_write_3.out', 'array_write_4.out']
        if args.use_2d_pos:
            sync_in_while_lastpp2firstpp_var_names.append("array_write_5.out")
        helper.gen_infer_program(
            sync_in_while_lastpp2firstpp_var_names=sync_in_while_lastpp2firstpp_var_names, 
            sync_in_while_var_names=['cond_int.tmp_0']
        )
    else:
        helper = HybridParallelInferenceHelper(
            startup_prog, train_program, num_mp=args.num_mp, num_pp=args.num_pp,
            micro_batch_size=micro_bsz,
            init_comm=fleet.worker_num() > 1 and not (args.save_ckpt_then_exist or args.save_inference_model_then_exist),
            beam_size=args.beam_size if args.decoding_strategy == "beam_search" else 1)

        helper.gen_infer_program()

    print('-'*20)
    print(place)
    print('-'*20)
    exe = fluid.Executor(place)
    seed = 2021 + random_seed_dp# should be the same for different mp due to topk sampling
    if args.num_mp > 1:
        paddle.seed(seed)

    debug_program('startup_prog', startup_prog)
    debug_program('train_program', train_program)

    #exit(0)
    #train_program = PassUtils.apply_ir_passes(train_program)

    steps = 0
    save_model_dir = os.path.join(args.checkpoints, 'saved_model_pp%dmp%d'%(args.num_pp, args.num_mp))
    if args.init_checkpoint and args.init_checkpoint != "":
        steps = args.init_checkpoint_step

    only_save_model = args.only_save_model
    if not only_save_model:
        exe.run(startup_prog)

        print("********************************************************")
        print("********* start load parameters first time ... *********")
        if args.init_checkpoint and args.init_checkpoint != "":
            print("worker_index is : ", fleet.worker_index())
            print("init pretraining params from ", args.init_checkpoint)
            idx = fleet.worker_index() % (fleet.worker_num() // args.num_dp)
            # if save_model_dir != args.init_checkpoint:
            #     print(f"[warning] {save_model_dir} != {args.init_checkpoint}, will do checkpoint_rearrange")
            #     checkpoint_rearrange(save_model_dir, args.init_checkpoint, idx, args.num_pp, args.num_mp, args.init_checkpoint_step, topo.dp.rank)

            init_checkpoint_path = os.path.join(save_model_dir, 'rank_' + str(idx), 'step_' + str(args.init_checkpoint_step))
            #init_checkpoint_path = os.path.join(save_model_dir, 'rank_' + str(idx))
            if args.num_pp > 1:
                if args.use_init_checkpoint:
                    init_checkpoint(exe, init_checkpoint_path, train_program)
                else:
                    init_pretraining_params(exe, init_checkpoint_path, train_program)
            else:
                if args.use_init_checkpoint:
                    init_checkpoint(exe, init_checkpoint_path, train_program)
                else:
                    init_pretraining_params(exe, init_checkpoint_path, train_program)
            steps = args.init_checkpoint_step

        if args.use_amp:
            fp16_utils.cast_parameters_to_fp16(place, train_program, to_fp16_var_names=fp16_var_names)
        import sys
        sys.path.append("..")
        from quant_py.weight_only_quant_all_fused_multi_transformer_pass import WeightOnlyQuantFusedMultiTransformerPass
        quant_pass = WeightOnlyQuantFusedMultiTransformerPass(fluid.global_scope())
        for block in train_program.blocks:
            quant_pass.apply(block)
    def remove_op_device(block):
        for op in block.ops:
            # op._set_attr('@ENABLE_CACHE_RUNTIME_CONTEXT@', True)
            if op.has_attr('is_test'):
                op._set_attr('is_test', True)

            if op.has_attr('op_device'):
                op._set_attr('op_device', '')
            if op.has_attr('sub_block'):
                sub_block_id = op.attr('sub_block').id
                remove_op_device(train_program.block(sub_block_id))

    remove_op_device(train_program.global_block())

    train_exe = exe

    fetch_list = []
    if is_last:
        fetch_list.extend([var for var in graph_vars])

    if args.save_ckpt_then_exist:
        save_path = os.path.join(save_model_dir, 'rank_' + str(fleet.worker_index()), 'step_' + str(steps))
        print("saving models to {}".format(save_path))
        save_checkpoint(exe, save_path, train_program, args.num_sharding, args.num_pp)

    if args.save_inference_model_then_exist:
        save_inference_model_dir = f'inference_model_pp{args.num_pp}mp{args.num_mp}'
        inference_save_path = os.path.join(save_inference_model_dir, 'rank_' + str(fleet.worker_index()), 'step_' + str(steps))
        print("saving inference models to {}".format(inference_save_path))
        qid_name_idx = feed_names.index("qids")
        feed_names_wo_qids = feed_names[:qid_name_idx] + feed_names[qid_name_idx+1:]
        if args.use_rope:
            feed_names_wo_qids.remove("pos_ids")
            if args.use_2d_pos:
                feed_names_wo_qids.remove("pos_ids_extra")
        save_inference_model(
            inference_save_path, 
            feed_names_wo_qids,
            [var.name for var in graph_vars[1:3]] if is_last else [], 
            exe, train_program, args.num_sharding, args.num_pp,
            is_first, is_last, only_save_model)

    if args.save_ckpt_then_exist or args.save_inference_model_then_exist:
        os._exit(0) 

    train_pyreader.start()
    qid_to_results = {}
    total_training_samples = data_reader.get_num_examples(args.input_file)
    time_begin = None
    while True: #steps < args.num_train_steps:
        try:
            steps += 1
            skip_steps = args.skip_steps
            if time_begin is None:
                time_begin = time.time()

            s = time.time()
            outputs = train_exe.run(fetch_list=fetch_list, program=train_program, use_program_cache=True, return_numpy=False)
            elpase = time.time() - s
            print("time elpase:", elpase, "qps:", 1 / elpase * args.batch_size, 'End Training: ', steps)

            if is_last:
                qids, finished_ids, finished_scores = outputs[:3]
                # print('finished_ids.lod()', finished_ids.lod(), 'finished_scores.lod()', finished_scores.lod())
                qids = np.array(qids).reshape(-1)
                finished_ids_np = np.array(finished_ids)
                finished_scores = np.array(finished_scores)
                bsz = len(finished_ids.lod()[0])-1
                for i in range(bsz):
                    qid = qids[i]
                    start = finished_ids.lod()[0][i]
                    end = finished_ids.lod()[0][i + 1]
                    for j in range(start, end):
                        sub_start = finished_ids.lod()[1][j]
                        sub_end = finished_ids.lod()[1][j + 1]

                        decode_score = float(finished_scores[sub_end - 1])
                        response_tokens = data_reader.tokenizer.convert_ids_to_tokens(finished_ids_np[sub_start:sub_end].tolist())[1:]
                        if qid not in qid_to_results or qid_to_results[qid][0] < decode_score:
                            qid_to_results[qid] = [decode_score, response_tokens]
                print(
                    "qid:", qid, 
                    'post strings:', "".join(post_process(response_tokens, args.aggressive_break, args.stop_token))
                )
                    

                if steps % 50 == 0:
                    print(f'[process] {steps * dp_worldsize * args.batch_size} / {total_training_samples}')
                
        except fluid.core.EOFException:
            train_pyreader.reset()
            break
    time_str = time.time()
    if is_last and fleet.worker_index() % args.num_mp == 0:
        total_num = len(qid_to_results)
        print('-'*20)
        print(f'finish processing {total_num} samples')
        print('-'*20)
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, f'predictions_dp{dp_rank}_seed{random_seed_dp}_{time_str}.txt'), 'w') as fout, \
            open(os.path.join(output_dir, f'predictions_dp{dp_rank}_seed{random_seed_dp}_{time_str}_ori.txt'), 'w') as fout_ori:
            qid_to_response_tokens = {}
            # fout.write("qid\tresponse\tresponse_length\n")
            fout.write("response\n")
            for qid in sorted(qid_to_results):
                qid_str = data_reader.qid_int_to_str[qid]
                response_str = "".join(post_process(qid_to_results[qid][1], aggressive_break=args.aggressive_break, stop_token=args.stop_token))
                qid_to_response_tokens[qid_str] = response_str
                # fout.write(f"{qid_str}\t{response_str}\t{word_count(qid_to_results[qid][1])}\n")
                fout.write(f"{response_str}\n")
                fout_ori.write(f"{qid_str}\t{qid_to_results[qid][1]}\n")
        with open(os.path.join(output_dir, f'predictions_dp{dp_rank}_seed{random_seed_dp}_{time_str}.json'), 'w') as fout:
            json.dump(qid_to_response_tokens, fout, ensure_ascii=False, indent=4)
        if args.multi_prompt_metric != "":
            ret = eval(f"metric.{args.multi_prompt_metric}")(qid_to_response_tokens, args.input_file)
            with open(os.path.join(output_dir, f'metric_dp{dp_rank}_seed{random_seed_dp}_{time_str}.json'), 'w') as fout:
                json.dump(ret, fout, indent=4)
        
        print("time elpase:", time.time() - time_begin)

        if dp_worldsize > 1:
            print(f"[waring, dp_worldsize={dp_worldsize}] For now, you should manually merge predictions.txt from different machine ")


if __name__ == '__main__':
    print_arguments(args)
    train(args)
