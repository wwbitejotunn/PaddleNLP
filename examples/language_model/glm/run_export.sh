# FUSE_MT=1 python3 export_generation_model.py --model_path /root/paddlejob/workspace/env_run/fhq/models/glm/checkpoint-100
export PYTHONPATH=/root/paddlejob/workspace/env_run/fhq/paddlenlp/PaddleNLP:${PYTHONPATH}
export FUSE_MT=1
python3 export_generation_model.py --model_path /root/paddlejob/workspace/env_run/fhq/models/glm_10b --data_type float16
