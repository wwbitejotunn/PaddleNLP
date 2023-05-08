# python3 predict_generation.py --model_path /root/paddlejob/workspace/env_run/fhq/models/glm/checkpoint-100/
export PYTHONPATH=/root/paddlejob/workspace/env_run/wangbojun/glm_10b_int8_ptq/PaddleNLP/:${PYTHONPATH}
export RING_ID=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export FUSE_MT=0
python3 -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" predict_generation_mp.py > unfused.log 2>&1

export FUSE_MT=1
export GLOG_v=1
python3 -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" predict_generation_mp.py > fuse_mt.log 2>&1
