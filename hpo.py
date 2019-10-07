import os
import argparse
import time

from crayai import hpo

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/agnn_hpo.yaml')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes to run optimization over, total')
    parser.add_argument('--nodes-per-eval', type=int, default=1,
                        help='Number of nodes per individual evaluation')
    parser.add_argument('--demes', type=int, default=4,
                        help='Number of populations')
    parser.add_argument('--pop-size', type=int, default=4,
                        help='Size of the genetic population')
    parser.add_argument('--generations', type=int, default=4,
                        help='Number of generations to run')
    parser.add_argument('--mutation-rate', type=float, default=0.05,
                        help='Mutation rate between generations of genetic optimization')
    parser.add_argument('--crossover-rate', type=float, default=0.33,
                        help='Crossover rate between generations of genetic optimization')
    #parser.add_argument('--output-dir', default='./run',
    #                    help='Directory to store all outputs and checkpoints')
    parser.add_argument('--alloc-args', default='-J hpo -C gpu --gres=gpu:8 -c 10 --exclusive -t 8:00:00')
    return parser.parse_args()

def main():

    args = parse_args()

    # Hyperparameters
    params = hpo.Params([
        ['--lr', 0.001, (1e-6, 0.01)],
        ['--hidden-dim', 32, [16, 32, 64]],
        ['--weight-decay', 1.e-4, (0., 0.01)],
        ['--n-graph-iters', 4, (1, 16)],
        ['--real-weight', 3., (1., 6.)]
    ])
    
    # Define the command to be run by the evaluator
    cmd = 'python train.py %s' % args.config
    cmd += ' --rank-gpu -d ddp-file'
    cmd += ' --fom best --n-epochs 16'
    
    # Define the evaluator
    #result_dir = os.path.expandvars('$SCRATCH/heptrkx/results/hpo_%s' %
    #                                time.strftime('%Y%m%d_%H%M%S'))
    evaluator = hpo.Evaluator(cmd,
                              #run_path=result_dir,
                              nodes=args.nodes,
                              launcher='wlm',
                              verbose=True, 
                              nodes_per_eval=args.nodes_per_eval,
                              #checkpoint='checkpoints',
                              alloc_args=args.alloc_args)
    
    # Define the Optimizer
    results_file = 'hpo.log'
    optimizer = hpo.GeneticOptimizer(evaluator,
                                     pop_size=args.pop_size,
                                     num_demes=args.demes,
                                     generations=args.generations,
                                     mutation_rate=args.mutation_rate,
                                     crossover_rate=args.crossover_rate,
                                     verbose=True,
                                     log_fn=results_file)
    
    # Run the Optimizer
    optimizer.optimize(params)

if __name__ == '__main__':
    main()
