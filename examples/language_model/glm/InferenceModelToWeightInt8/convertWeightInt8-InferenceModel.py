import os
import argparse
import paddle
import paddle.fluid as fluid
# from paddle.fluid import debugger
from paddle.fluid import core
paddle.enable_static()
import sys
sys.path.append('./quant_py/')
from quant_py.weight_only_quant_all_fused_multi_transformer_pass_gpu import WeightOnlyQuantFusedMultiTransformerPass

def parse_args():
  parser = argparse.ArgumentParser(description='convert fp16 inference model to weight only int8 model')
  parser.add_argument('--input_dir',type=str, default='/root/paddlejob/workspace/env_run/fhq/dev_0017/PaddleNLP/examples/language_model/glm/inference/')
  parser.add_argument('--input_model_name',type=str, default = 'glm')
  parser.add_argument('--output_dir',type=str, default='/root/paddlejob/workspace/env_run/fhq/dev_0017/PaddleNLP/examples/language_model/glm/int8_weightonly/')
  args = parser.parse_args()
  return args, parser


def load_inference_model(model_path, model_name, param_name, exe):
    '''
    '''
    model_abs_path = os.path.join(model_path, model_name)
    param_abs_path = os.path.join(model_path, param_name)
    print(model_abs_path)
    print(param_abs_path)
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, model_name, param_name)
    else:
        return fluid.io.load_inference_model(model_path, exe)



if __name__=='__main__':
    args, parser = parse_args()
    model_file_name=args.input_model_name+'.pdmodel'
    param_file_name=args.input_model_name+'.pdiparams'
    load_path=args.input_dir
    save_path=args.output_dir
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    quant_pass = WeightOnlyQuantFusedMultiTransformerPass(place,scope)

    with fluid.scope_guard(scope):
        [net_program,
        feed_target_names,
        fetch_targets] = load_inference_model(load_path, model_file_name, param_file_name, exe)
        for block in net_program.blocks:
            quant_pass.apply(block)
        fluid.io.save_inference_model(save_path, feeded_var_names=feed_target_names, target_vars=fetch_targets, executor=exe, main_program=net_program, model_filename=model_file_name,params_filename=param_file_name)