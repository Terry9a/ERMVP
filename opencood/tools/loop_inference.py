import os   

for index in range(41,60,2):
  cmd = f"CUDA_VISIBLE_DEVICES=1 python /home/gaojing/zjy/cvpr/V2V4Real/opencood/tools/inference.py --eval_epoch {index} "
  print(f"Running command: {cmd}")
  os.system(cmd)