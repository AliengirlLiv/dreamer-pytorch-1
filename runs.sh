CUDA_VISIBLE_DEVICES=5 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id 1 --distribution_shift mass
CUDA_VISIBLE_DEVICES=5 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id 2 --distribution_shift mass
CUDA_VISIBLE_DEVICES=6 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id 3 --distribution_shift mass
CUDA_VISIBLE_DEVICES=6 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id 4 --distribution_shift mass