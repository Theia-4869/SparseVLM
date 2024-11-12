CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/vizwiz.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_cn.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
