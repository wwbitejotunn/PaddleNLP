# FUSE_MT=1 python3 export_generation_model.py --model_path /root/paddlejob/workspace/env_run/fhq/models/glm/checkpoint-100
export PYTHONPATH=/root/paddlejob/workspace/env_run/fhq/paddlenlp/PaddleNLP:${PYTHONPATH}
export FUSE_MT=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7"  export_generation_model_mp.py --model_path /root/paddlejob/workspace/env_run/fhq/paddlenlp/PaddleNLP/examples/language_model/glm/glm_10b_mp/ --data_type float16