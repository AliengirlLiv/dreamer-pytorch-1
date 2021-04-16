CUDA_VISIBLE_DEVICES=5 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id mass1 --distribution_shift mass
CUDA_VISIBLE_DEVICES=6 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id mass2 --distribution_shift mass
CUDA_VISIBLE_DEVICES=7 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id mass3 --distribution_shift mass


CUDA_VISIBLE_DEVICES=6 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug --distribution_shift mass