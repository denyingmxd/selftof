# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function



from options import MonodepthOptions
import sys
import torch

options = MonodepthOptions()

if sys.argv.__len__() == 2:
    # arg_filename_with_prefix = '@' + sys.argv[1]
    arg_filename_with_prefix = sys.argv[1]
    args = options.parser.parse_args([arg_filename_with_prefix])
else:
    args = options.parser.parse_args()
print(args.model_name)
print(sys.argv[1])
assert args.model_name in sys.argv[1]

if __name__ == "__main__":
    if args.distributed:
        from trainer_ddp import train_ddp
        import torch.multiprocessing as mp
        # args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]


        mp.set_start_method('forkserver')

        print(args.rank)
        port = args.port
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

        ngpus_per_node = torch.cuda.device_count()

        args.ngpus_per_node = ngpus_per_node
        # args.batch_size = args.n_batch
        if args.distributed:
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
