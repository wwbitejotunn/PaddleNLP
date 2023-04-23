import collections
import numpy as np
import paddle

from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from paddle.fluid.framework import IrNode
from paddle.fluid.framework import Operator
from paddle.fluid import unique_name

from paddle.fluid.framework import Program, program_guard, default_startup_program
from paddle.fluid.data import data
from paddle.fluid.executor import scope_guard
from paddle.fluid.framework import _get_paddle_place

def clip_and_round(x):
  return np.clip(np.around(x), -127, 127).astype('int8')

class QuantFusedMultiTransformerPass:
  def __init__(self, scope):
    self._scope = scope
    self._place =  paddle.fluid.libpaddle.CPUPlace()

  def _load_var(self, name):
    return np.array(self._scope.find_var(name).get_tensor())

  def _store_var(self, name, array, dtype):
    tensor = self._scope.find_var(name).get_tensor()
    tensor.set(array.astype(dtype), self._place)

  def _quant_qkv_weight(self, weight, num_head, dim_head, dim_embed):
    weight = np.resize(weight, [3 * num_head * dim_head, dim_embed])
    max_value = np.max(np.abs(weight), axis=1).astype('float32')
    print(max_value.shape)

    quanted_weight = weight.transpose((1,0))
    quanted_weight = clip_and_round(quanted_weight * 127.0 / max_value)
    quanted_weight = quanted_weight.transpose((1,0))

    quanted_weight.resize([3, num_head, dim_head, dim_embed])
    return quanted_weight, max_value / 127.0

  def _quant_weight(self, weight, k, n):
    weight.resize([k, n])
    max_value = np.max(np.abs(weight), axis=0).astype('float32')
    quanted_weight = clip_and_round(weight * 127.0 / max_value)
    quanted_weight = quanted_weight.transpose((1,0))
    return quanted_weight, max_value / 127.0


  def apply(self, block):
    
    # Find fused_mt op and rename it
    for op in block.ops:
      print(op.type)
      if op.type == "fused_multi_transformer":
        op.desc.set_type("fused_multi_transformer_dyquant")

        qkv_weight_names = op.input("QKVW")

        qkv_out_scale_names = []

        # Quant QKVWeight and calculate scale
        for weight_name in qkv_weight_names:
          print("process ", weight_name)
          weight_tensor = self._scope.find_var(weight_name).get_tensor()
          assert weight_tensor is not None
          shape = np.array(weight_tensor).shape
          print(shape)
          dim_embed = shape[3]
          num_head, dim_head = shape[1], shape[2]

          out_scale_name = weight_name + "_out_scale"
          qkv_out_scale_names.append(out_scale_name)

          if self._scope.find_var(out_scale_name) is None:
            # np.save(weight_name+ "_ori", np.array(weight_tensor))
            quanted_weight_data, out_scale_data = self._quant_qkv_weight(np.array(weight_tensor), num_head, dim_head, dim_embed)
            # np.save(weight_name+ "_int8", quanted_weight_data)
            # np.save(weight_name+ "_scale", out_scale_data)
            weight_tensor.set(quanted_weight_data, self._place)

            out_scale_tensor = self._scope.var(out_scale_name).get_tensor()
            block.create_parameter(shape=out_scale_data.shape, dtype='float32', name=out_scale_name)
            out_scale_tensor.set(out_scale_data, self._place)
          
        op.desc.set_input("QKVOutScale", qkv_out_scale_names)

        # Quant / Transpose other weight and caculate scale

        # weight_names = []
        # weight_names += op.input("OutLinearW")
        # weight_names += op.input("FFN1Weight")
        # weight_names += op.input("FFN2Weight")

        weight_var_names = ["OutLinearW", "FFN1Weight", "FFN2Weight"]
        out_scale_var_names =["OutLinearOutScale", "FFN1OutScale", "FFN2OutScale"]

        for i in range(3):
          weight_names = op.input(weight_var_names[i])
          out_scale_var_name = out_scale_var_names[i]

          out_scale_names = []

          for weight_name in weight_names:
            print("process ", weight_name)
            weight_tensor = self._scope.find_var(weight_name).get_tensor()
            shape = np.array(weight_tensor).shape
            k = shape[0]
            n = shape[1]

            out_scale_name = weight_name + "_out_scale"
            out_scale_names.append(out_scale_name)

            if self._scope.find_var(out_scale_name) is None:
              # np.save(weight_name+ "_ori", np.array(weight_tensor))
              quanted_weight_data, out_scale_data = self._quant_weight(np.array(weight_tensor), k, n)
              # np.save(weight_name+ "_int8", quanted_weight_data)
              # np.save(weight_name+ "_scale", out_scale_data)
              weight_tensor.set(quanted_weight_data, self._place)

              out_scale_tensor = self._scope.var(out_scale_name).get_tensor()
              block.create_parameter(shape=out_scale_data.shape, dtype='float32', name=out_scale_name)
              out_scale_tensor.set(out_scale_data, self._place)
          op.desc.set_input(out_scale_var_name, out_scale_names)