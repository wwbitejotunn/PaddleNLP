import collections
import numpy as np
import cupy as cp
import paddle

from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from paddle.fluid.framework import IrNode
from paddle.fluid.framework import Operator
from paddle.fluid import unique_name

from paddle.fluid.framework import Program, program_guard, default_startup_program
# from paddle.fluid.data import data
from paddle.fluid.executor import scope_guard
from paddle.fluid.framework import _get_paddle_place
import os
import weight_only_process
def clip_and_round(x):
  return cp.clip(cp.around(x), -127, 127).astype('int8')

class WeightOnlyQuantFusedMultiTransformerPass:
  def __init__(self, place, scope=None,):
    if scope is not None:
      self._scope = scope
      self.device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
      self._place = place
      self._processed_params_name=[]

  def _load_var(self, name):
    return cp.array(self._scope.find_var(name).get_tensor())

  def _store_var(self, name, array, dtype):
    tensor = self._scope.find_var(name).get_tensor()
    tensor.set(array.astype(dtype), self._place)
    self._processed_params_name.append(name)


    # for cutalss fpA int8 B gemm kernel
    # we need to permute the rows of weight as below map
    # each group of 16 rows is permuted using the map below:
    # 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15  
  def _fpA_int8B_weight_permute(self, weight):
    weight=weight.astype('int8')+128
    weight=weight.astype('int8')
    # weight_shape = [k,n]
    weight_shape = weight.shape
    dim=weight_shape[0]
    threeNH=weight_shape[1]
    numel=dim*threeNH
    assert len(weight_shape)==2
    permute_matrix = cp.zeros([weight_shape[0],weight_shape[0]])
    permute_rows_map = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    for i in range(permute_matrix.shape[0]):
      permute_matrix[i,permute_rows_map[i%16]+i//16*16]=1
    permuted_weight=cp.matmul(permute_matrix.astype('float32'),weight.astype('float32')).astype('int8')
    permuted_weight=cp.transpose(permuted_weight,(1,0))
    permuted_weight_linear_cpu=np.ascontiguousarray(np.resize((cp.asnumpy(permuted_weight)),numel).astype('int8'))
    permuted_weight_interleave_column_major_cpu=np.ascontiguousarray(np.zeros(numel,dtype=np.int8))
    weight_only_process.interleave_column_major_tensor(int(permuted_weight_interleave_column_major_cpu.ctypes.data),
                                                       int(permuted_weight_linear_cpu.ctypes.data),
                                                       [int(dim),int(threeNH)])
    weight_only_process.add_bias_and_interleave_int8s_inplace(
      int(permuted_weight_interleave_column_major_cpu.ctypes.data),
      int(numel)
    )
    return permuted_weight_interleave_column_major_cpu
    # origin qkv weight is [3*nh*dim,dim]
    # trans_qkvw=true
  def _quant_qkv_weight(self, weight_np, num_head, dim_head, dim_embed):
    weight=cp.asarray(weight_np)
    # [dim_embed, 3, num_head, dim_head] -> [num_head, 3, dim_head, dim_embed]
    # weight = cp.transpose(weight, (2, 1, 3, 0))
    weight = cp.reshape(weight,[3 * num_head * dim_head, dim_embed])
    max_value = cp.max(cp.abs(weight), axis=1).astype('float32')
    print(max_value.shape)

    quanted_weight = cp.transpose(weight,(1,0))
    quanted_weight = clip_and_round(quanted_weight * 127.0 / max_value)
    quanted_weight = self._fpA_int8B_weight_permute(quanted_weight)
    quanted_weight = np.resize(quanted_weight,[3, num_head, dim_head, dim_embed])

    # [3, num_head, dim_head, dim_embed]
    
    max_value = cp.asnumpy(max_value)
    return (quanted_weight.astype('int8')), ((max_value / 127.0).astype('float16'))


  def _quant_weight(self, weight_np, k, n):
    weight=cp.asarray(weight_np)
    weight=cp.reshape(weight,[k, n])
    max_value = cp.max(cp.abs(weight), axis=0).astype('float32')
    quanted_weight = clip_and_round(weight * 127.0 / max_value)
    quanted_weight = self._fpA_int8B_weight_permute(quanted_weight)
    quanted_weight = np.resize(quanted_weight,[n,k])
    max_value = cp.asnumpy(max_value)
    return (quanted_weight.astype('int8')), ((max_value / 127.0).astype('float16'))

  def apply(self, block):
    # Find fused_mt op and rename it
    for op in block.ops:
      print(op.type)
      if op.type == "fused_multi_transformer":
        op._set_attr("quant_weight",True)
        op._set_attr("trans_qkvw",True)
        qkv_weight_names = op.input("QKVW")
        qkv_out_scale_names = []

        # Quant QKVWeight and calculate scale
        for i, weight_name in enumerate(qkv_weight_names):
          if(weight_name in self._processed_params_name):
            continue
          print("process ", weight_name)
          weight_tensor = self._scope.find_var(weight_name).get_tensor()
          assert weight_tensor is not None
          shape = weight_tensor._get_dims()
          print(shape)
          # import pdb;pdb.set_trace()
          dim_embed = shape[3]
          num_head, dim_head = shape[1], shape[2]
          # dim_embed = shape[0]
          # num_head, dim_head = shape[2], shape[3]

          out_scale_name = weight_name + "_weight_only_scale"
          qkv_out_scale_names.append(out_scale_name)
          if self._scope.find_var(out_scale_name) is None:
            quanted_weight_data, out_scale_data = self._quant_qkv_weight(cp.array(weight_tensor), num_head, dim_head, dim_embed)
            weight_tensor.set(quanted_weight_data, self._place)

            out_scale_tensor = self._scope.var(out_scale_name).get_tensor()
            block.create_parameter(shape=out_scale_data.shape, dtype='float16', name=out_scale_name)
            out_scale_tensor.set(out_scale_data, self._place)
            del quanted_weight_data
            del out_scale_data 
        op.desc.set_input("QKVWScale", qkv_out_scale_names)

        weight_var_names = ["OutLinearW", "FFN1Weight", "FFN2Weight"]
        out_scale_var_names = ["OutLinearWScale", "FFN1WeightScale", "FFN2WeightScale"]

        for i in range(3):
          weight_names = op.input(weight_var_names[i])
          out_scale_var_name = out_scale_var_names[i]

          out_scale_names = []

          for weight_name in weight_names:
            if(weight_name in self._processed_params_name):
              continue
            print("process ", weight_name)

            weight_tensor = self._scope.find_var(weight_name).get_tensor()
            shape = weight_tensor._get_dims()
            k = shape[0]
            n = shape[1]

            out_scale_name = weight_name + "_weight_only_scale"
            out_scale_names.append(out_scale_name)

            if self._scope.find_var(out_scale_name) is None:
              quanted_weight_data, out_scale_data = self._quant_weight(cp.array(weight_tensor), k, n)
              weight_tensor.set(quanted_weight_data, self._place)
              out_scale_tensor = self._scope.var(out_scale_name).get_tensor()
              block.create_parameter(shape=out_scale_data.shape, dtype='float16', name=out_scale_name)
              out_scale_tensor.set(out_scale_data, self._place)
              del quanted_weight_data
              del out_scale_data
          op.desc.set_input(out_scale_var_name, out_scale_names)
if __name__ == '__main__':
  # test
  print('@@@ start test')
  device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
  quant_pass=WeightOnlyQuantFusedMultiTransformerPass(paddle.fluid.libpaddle.CUDAPlace(device_id))
  
  qkv_weight_np=np.ones([12288,3*8*192])
  for i in range(qkv_weight_np.shape[0]):
    for j in range(qkv_weight_np.shape[1]):
      qkv_weight_np[i,j]=i
  qkv_weight_np=qkv_weight_np.transpose((1,0))
  qkv=paddle.fluid.libpaddle.Tensor()
  qkv.set(qkv_weight_np,paddle.fluid.libpaddle.CUDAPlace(device_id))

  processed_qkv=quant_pass._quant_qkv_weight(cp.array(qkv),8,192,12288)

