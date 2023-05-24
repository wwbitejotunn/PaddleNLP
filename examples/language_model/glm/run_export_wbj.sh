# FUSE_MT=1 python3 export_generation_model.py --model_path /root/paddlejob/workspace/env_run/fhq/models/glm/checkpoint-100
export PYTHONPATH=//root/paddlejob/workspace/env_run/wangbojun/ptq/glm/paddlenlp//:${PYTHONPATH}
export FUSE_MT=1
python3 export_generation_model.py --model_path ./glm_10b_quant_model --data_type float16 --output_path inference_from_ptq/glm
