

import paddle
from paddlenlp.transformers import *
import os

FUSE_MT = os.getenv("FUSE_MT") == "1"
def _generate_cache(batch_size, max_length):
    num_layers = 24
    num_attention_head = 16
    hidden_size = 1024
    head_dim = hidden_size // num_attention_head

    cache_kvs = []
    cache_kv_size = (2, batch_size, num_attention_head, max_length, head_dim)
    
    for _ in range(num_layers):
        # import pdb;pdb.set_trace()
        cache_kv = paddle.zeros(cache_kv_size, dtype=paddle.get_default_dtype())
        cache_kvs.append(cache_kv)
    return cache_kvs

import random
import numpy as np
paddle.seed(100)
random.seed(100)
np.random.seed(100)

def func(self, *args, **kwargs):
    return

# 屏蔽init_weights 
GLMModel.init_weights = func

model = GLMModel.from_pretrained(
    "THUDM/glm-large-chinese",
    # dtype="float16",
)
model.init_weight_fuse_mt()
model.eval()

max_length = 10
caches = _generate_cache(1, max_length)
input_ids = paddle.arange(100, 110, dtype="int64").reshape([1, -1])
seq_len = input_ids.shape[1]
attention_mask = paddle.zeros((1, 1, max_length, max_length), dtype=paddle.int32)
attention_mask[:, :, :seq_len, :seq_len] = 1

out = model(
    input_ids=input_ids, 
    cache=caches if FUSE_MT else None,
    attention_mask=attention_mask if FUSE_MT else None
)
# for i in range(24):
#     ret = out.hidden_states[i][:,:,1]
#     print(ret)
print("output:", out.logits)