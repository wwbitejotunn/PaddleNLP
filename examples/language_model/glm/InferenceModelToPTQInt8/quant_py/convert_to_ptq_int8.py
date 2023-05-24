import collections
import numpy as np
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
import json

class load_act_scale_json():
  def __init__(self, json_file_path="glm_10b_act_scales.json"):
    with open(json_file_path) as json_file:
      self.scale_dict = json.load(json_file)
    self.key_map={
      "qkv_in_scale":"glm.transformer.layers.#.attention.query_key_value",
      "out_linear_in_scale":"glm.transformer.layers.#.attention.dense",
      "ffn1_in_scale":"glm.transformer.layers.#.mlp.dense_h_to_4h",
      "ffn2_in_scale":"glm.transformer.layers.#.mlp.dense_4h_to_h",
    }
    self.scale={}
    for scale_type, key_template in self.key_map.items():
      num_layer = 0
      for i in range(len(self.scale_dict.keys())):
        if(key_template.replace("#",str(i)) in self.scale_dict.keys()):
          num_layer=num_layer+1
      
      self.scale[scale_type]=np.array([
        1/self.scale_dict[key_template.replace("#",str(i))] for i in range(num_layer)
      ])

class load_weight_scale_json():
  def __init__(self, json_file_path="glm_10b_weight_scales.json"):
    with open(json_file_path) as json_file:
      self.scale_dict = json.load(json_file)
    self.key_map={
      "qkv_weight_scale":"glm.transformer.layers.#.attention.query_key_value",
      "out_linear_weight_scale":"glm.transformer.layers.#.attention.dense",
      "ffn1_weight_scale":"glm.transformer.layers.#.mlp.dense_h_to_4h",
      "ffn2_weight_scale":"glm.transformer.layers.#.mlp.dense_4h_to_h",
    }
    self.scale={}
    for scale_type, key_template in self.key_map.items():
      num_layer = 0
      for i in range(len(self.scale_dict.keys())):
        if(key_template.replace("#",str(i)) in self.scale_dict.keys()):
          num_layer=num_layer+1
      
      self.scale[scale_type]=np.array([
        self.scale_dict[key_template.replace("#",str(i))] for i in range(num_layer)
      ])

class ptq_int8_converter:
  def __init__(self, place, scope=None,):
    if scope is not None:
      self._scope = scope
      self.device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
      self._place = place
      self._processed_params_name=[]

  def _load_var(self, name):
    return np.array(self._scope.find_var(name).get_tensor())

  def _store_var(self, name, array, dtype):
    tensor = self._scope.find_var(name).get_tensor()
    tensor.set(array.astype(dtype), self._place)
    self._processed_params_name.append(name)



  def apply(self, block):
    # Find fused_mt op and rename it
    act_scales = load_act_scale_json()
    weight_scales = load_weight_scale_json()
    for op in block.ops:
      print(op.type)
      if op.type == "fused_multi_transformer":
        op.desc.set_type("fused_multi_transformer_int8")
        # assert(op._get_attr("trans_qkvw"))

        # op._set_attr("trans_qkvw",True)
        ## TODO(wangbojun), check the quant_round_type
        op._set_attr("quant_round_type",1)
        op._set_attr("quant_max_bound",127.0)
        op._set_attr("quant_min_bound",-127.0)
        qkv_weight_names = op.input("QKVW")
        # set in scale as attr
        op._set_attr("qkv_in_scale",act_scales.scale["qkv_in_scale"])
        op._set_attr("out_linear_in_scale",act_scales.scale["out_linear_in_scale"])
        op._set_attr("ffn1_in_scale",act_scales.scale["ffn1_in_scale"])
        op._set_attr("ffn2_in_scale",act_scales.scale["ffn2_in_scale"])

        # qkv
        qkv_out_scale_names=[]
        for i, weight_name in enumerate(qkv_weight_names):
          qkv_out_scale_name = weight_name+"_out_scale"
          if(weight_name in self._processed_params_name):
            qkv_out_scale_name = weight_name+"_out_scale"
            qkv_out_scale_names.append(qkv_out_scale_name)
            qkv_out_scale_tensor = self._scope.find_var(qkv_out_scale_name).get_tensor()
            qkv_out_scale_data = np.array(qkv_out_scale_tensor)
            block.create_parameter(shape=qkv_out_scale_data.shape, dtype='float32', name=qkv_out_scale_name)
            qkv_out_scale_tensor.set(qkv_out_scale_data,self._place)
            continue
          print("process ", weight_name)
          self._processed_params_name.append(weight_name)
          weight_tensor = self._scope.find_var(weight_name).get_tensor()
          shape = weight_tensor._get_dims()
          num_head, dim_head = shape[1], shape[2]
          out_dim=3*num_head* dim_head
          weight_data = np.array(weight_tensor)
          weight_data_int8 = weight_data.astype("int8")
          weight_tensor.set(weight_data_int8, self._place)
          out_scale_tensor = self._scope.var(qkv_out_scale_name).get_tensor()
          out_scale_data=weight_scales.scale["qkv_weight_scale"][i]/(127.0*act_scales.scale["qkv_in_scale"][i]*127.0)
          # import pdb;pdb.set_trace()
          if out_scale_data.ndim==0:
            out_scale_data=np.repeat(out_scale_data,out_dim)
          out_scale_data=out_scale_data.astype('float32')
          block.create_parameter(shape=out_scale_data.shape, dtype='float32', name=qkv_out_scale_name)
          out_scale_tensor.set(out_scale_data, self._place)
          qkv_out_scale_names.append(qkv_out_scale_name)
        op.desc.set_input("QKVOutScale", qkv_out_scale_names)
        # out_linear, ffn1, ffn2
        weight_var_names = ["OutLinearW", "FFN1Weight", "FFN2Weight"]
        out_scale_var_names = ["OutLinearOutScale", "FFN1OutScale", "FFN2OutScale"]
        out_scale_keys = ["out_linear_weight_scale","ffn1_weight_scale","ffn2_weight_scale"]
        for i_gemm in range(3):
          weight_names = op.input(weight_var_names[i_gemm])
          out_scale_var_name = out_scale_var_names[i_gemm]
          out_scale_names = []
          for i,weight_name in enumerate(weight_names):
            if(weight_name in self._processed_params_name):
              out_scale_name = weight_name+"_out_scale"
              out_scale_names.append(out_scale_name)
              out_scale_tensor = self._scope.find_var(out_scale_name).get_tensor()
              out_scale_data = np.array(out_scale_tensor)
              block.create_parameter(shape=out_scale_data.shape, dtype='float32', name=out_scale_name)
              out_scale_tensor.set(out_scale_data,self._place)
              continue
            print("process ", weight_name)
            self._processed_params_name.append(weight_name)
            weight_tensor = self._scope.find_var(weight_name).get_tensor()
            shape = weight_tensor._get_dims()
            out_dim = shape[1]
            weight_data = np.array(weight_tensor)
            weight_data_int8 = weight_data.astype("int8")
            weight_data_int8 = np.transpose(weight_data_int8, (1,0))
            weight_tensor.set(weight_data_int8, self._place)
            out_scale_name = weight_name + "_out_scale"
            out_scale_tensor = self._scope.var(out_scale_name).get_tensor()
            out_scale_data=weight_scales.scale[out_scale_keys[i_gemm]][i]/(127.0*act_scales.scale[out_scale_keys[i_gemm].replace("_weight_","_in_")][i]*127.0)
            if out_scale_data.ndim==0:
              out_scale_data=np.repeat(out_scale_data,out_dim)
            out_scale_data=out_scale_data.astype('float32')
            block.create_parameter(shape=out_scale_data.shape, dtype='float32', name=out_scale_name)
            out_scale_tensor.set(out_scale_data, self._place)
            out_scale_names.append(out_scale_name)
          op.desc.set_input(out_scale_var_name, out_scale_names)

