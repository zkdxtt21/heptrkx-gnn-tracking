from crayai import hpo
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run_pbt', action='store_true')
parser.add_argument('--gens_per_epoch', type=int, default=1)
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--nodes_per_eval', type=int, default=1)
parser.add_argument('--alloc_job_id', type=int, default=0)
parser.add_argument('--pop_size', type=int, default=16)
parser.add_argument('--generations', type=int, default=100)
parser.add_argument('--mutation_rate', type=float, default=0.05)
parser.add_argument('--crossover_rate', type=float, default=0.33)
parser.add_argument('--checkpoint', type=str, default='./checkpoints')
args = parser.parse_args()

#loss_funcs = ["binary_cross_entropy", "binary_cross_entropy_with_logits"]

#optimizers = ["Adam", "Adadelta", "Adagrad", "Adamax", "RMSprop", "SGD"]

if args.run_pbt:
    params = hpo.Params([["--lr", 0.001, (1e-6, 0.1)]])
    config_file = 'configs/agnn_pbt.yaml'
else:
    params = hpo.Params([["--hidden-dim", 64, (32, 512)],
                         ["--n_iters", 4, (1, 16)],
                         ["--lr", 0.001, (1e-6, 0.1)],
                         ["--real-weight", 3, (1, 5)]])
    config_file = 'configs/agnn.yaml'

cmd = "python train.py %s --resume --ranks-per-node 1 -v --gpu 0" % config_file
if args.run_pbt:
    cmd += " --load_checkpoint @checkpoint --save_checkpoint @checkpoint"

timestr = time.strftime("%Y%m%d-%H%M%S")
evaluator = hpo.Evaluator(cmd,
                          run_path="./runs/run%s" % timestr,
                          nodes=args.nodes,
                          launcher='wlm',
                          verbose=True, 
                          nodes_per_eval=args.nodes_per_eval,
                          checkpoint=args.checkpoint if args.run_pbt else '',
                          alloc_args="-J agnn-heptrkx --exclusive --time=24:00:00 -C P100 --gres=gpu",
                          alloc_jobid=args.alloc_job_id)

optimizer = hpo.genetic.Optimizer(evaluator,
                                  gens_per_epoch=args.gens_per_epoch,
                                  pop_size=args.pop_size,
                                  num_demes=1,
                                  generations=args.generations,
                                  mutation_rate=args.mutation_rate,
                                  crossover_rate=args.crossover_rate,
                                  verbose=True)

optimizer.optimize(params)
