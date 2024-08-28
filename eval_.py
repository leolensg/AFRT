
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from torchvision.datasets.utils import download_url
from tqdm import tqdm

import models
import utils
from datasets.samplers import CategoriesSampler

from afrt import AFRT
####################################################################
USE_WANDB = False

if USE_WANDB:
    import wandb
    # Note: Make sure to specify your username for correct logging
    WANDB_USER = 'username'
####################################################################


def get_args_parser():
    parser = argparse.ArgumentParser('evaluate', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'deit_tiny', 'deit_small', 'swin_tiny'],
                        help="""Name of architecture you want to evaluate.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base), but is used in swin to compute the 
            predictor size (8*patch_size vs. pred_size = patch_size in ViT) -- so be wary!. 
            If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
                            This config is only valid for Swin Transformer and is ignored for vanilla ViT architectures.""")

    # FSL task/scenario related parameters
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)  # number of query image per class ( -> paper claims 16?)
    parser.add_argument('--set', type=str, default='test', choices=['train', 'val', 'test'],
                        help="""Part of the dataset on which to run the evaluation. Default is 'test', but eval could 
                            also be run on 'train' or 'val' if desired.""")
    parser.add_argument('--num_test_episodes', type=int, default=600,
                        help="""Number of episodes used for testing. Results will be average over all. Classic FSL
                            default is 600 for both 5-shot and 1-shot experiments, but recent works have moved towards
                            increasing this to lower the variance: 600/5k for DeepEMD, 10k/10k for FEAT.""")

    # FSL adaptation component related parameters
    parser.add_argument('--block_mask_1shot', default=5, type=int, help="""Number of patches to mask around each 
                            respective patch during online-adaptation in 1shot scenarios: masking along main diagonal,
                            e.g. size=5 corresponds to masking along the main diagonal of 'width' 5.""")
    parser.add_argument('--similarity_temp_init', type=float, default=0.0421,
                        help="""Temperature for scaling the logits of the path embedding similarity matrix. Logits 
                            will be divided by that temperature, i.e. temp<1 scales up. 
                            'similarity_temp_init' must be positive.  0.051031036307982884 0.0421""")
    parser.add_argument('--disable_peiv_optimisation', type=utils.bool_flag, default=False,
                        help="""Disable the patch embedding importance vector (peiv) optimisation/adaptation.
                                This means that inference is performed simply based on the cosine similarity of support 
                                and query set sample embeddings, w/o any further adaptation.""")

    # Adaptation component -- Optimisation related parameters
    parser.add_argument('--optimiser_online', default='SGD', type=str, choices=['SGD'],
                        help="""Optimiser to be used for adaptation of patch embedding importance vector.""")
    parser.add_argument('--lr_online', default=0.1, type=float, help="""Learning rate used for online optimisation.""")
    parser.add_argument('--optim_steps_online', default=20, type=int,
                        help="""Number of update steps to take to optimise the patch embedding importance vector.""")

    # Dataset related parameters
    parser.add_argument('--image_size', type=int, default=224,
                        help="""Size of the squared input images, 224 for imagenet-style.""")
    parser.add_argument('--dataset', default='miniimagenet', type=str,
                        choices=['miniimagenet', 'tieredimagenet', 'fc100', 'cifar_fs'],
                        help='Please specify the name of the dataset to be used for training.')
    parser.add_argument('--data_path', default='D:/project_file/python/dataset/miniimagenet', type=str,
                        help='Please specify path to the root folder containing the training dataset(s). If dataset '
                         'cannot be loaded, check naming convention of the folder in the corresponding dataloader.'
                             ' D:/project_file/python/dataset/miniimagenet'
                        'cifar fc100 D:/project_file/python/dataset')

    # Misc
    # Checkpoint to load
    parser.add_argument('--wandb_mdl_ref', required=False, type=str,
                        help="""Complete path/tag of model at wandb if it is to be loaded from there.""")
    parser.add_argument('--mdl_checkpoint_path', default='meta/miniimagenet_224/vit_small/1shot-op20-window0-去参数-val-cls2', required=False, type=str,
                        help="""Path to checkpoint of model to be loaded. Actual checkpoint given via chkpt_epoch.""")
    parser.add_argument('--mdl_url', required=False, type=str,
                        help='URL from where to load the model weights or checkpoint, e.g. from pre-trained publically '
                              'available ones.')
    parser.add_argument('--trained_model_type', default="metaft", type=str, choices=['metaft', 'pretrained'],
                        help="""Type of model to be evaluated -- either meta fine-tuned (metaft) or pretrained.""")
    parser.add_argument('--chkpt_epoch', default=1600, type=int, help="""Number of epochs of pretrained 
                                    model to be loaded for evaluation. Irrelevant if metaft is selected.""")

    # Misc
    parser.add_argument('--output_dir', default="", type=str, help="""Root path where to save correspondence images. 
                            If left empty, results will be stored in './eval_results/...'.""")
    parser.add_argument('--seed', default=10, type=int, help="""Random seed.""")
    parser.add_argument('--num_workers', default=2, type=int, help="""Number of data loading workers per GPU.""")

    # new
    parser.add_argument('-temperature_attn', type=float, default=2.0, metavar='gamma',
                        help='temperature for softmax in computing cross-attention')
    return parser


def set_up_dataset(args):
    # Datasets and corresponding number of classes
    if args.dataset == 'miniimagenet':
        # (Vinyals et al., 2016), (Ravi & Larochelle, 2017)
        # train num_class = 64
        from datasets.dataloaders.miniimagenet.miniimagenet import MiniImageNet as dataset
    elif args.dataset == 'tieredimagenet':
        # (Ren et al., 2018)
        # train num_class = 351
        from datasets.dataloaders.tieredimagenet.tieredimagenet import tieredImageNet as dataset
    elif args.dataset == 'fc100':
        # (Oreshkin et al., 2018) Fewshot-CIFAR 100 -- orig. images 32x32
        # train num_class = 60
        from datasets.dataloaders.fc100.fc100 import DatasetLoader as dataset
    elif args.dataset == 'cifar_fs':
        # (Bertinetto et al., 2018) CIFAR-FS (100) -- orig. images 32x32
        # train num_class = 64
        from datasets.dataloaders.cifar_fs.cifar_fs import DatasetLoader as dataset
    else:
        raise ValueError('Unknown dataset. Please check your selection!')
    return dataset


def get_patch_embeddings(model, data, args):
    """Function to retrieve all patch embeddings of provided data samples, split into support and query set samples;
    Data arranged in 'aaabbbcccdddeee' fashion, so must be split appropriately for support and query set"""
    # Forward pass through backbone model;
    # Important: This contains the [cls] token at position 0 ([:,0]) and the patch-embeddings after that([:,1:end]).
    # We thus remove the [cls] token
    # patch_embeddings = model(data)[:, 1:]
    patch_embeddings = model(data)
    # temp size values for reshaping
    bs, seq_len, emb_dim = patch_embeddings.shape[0], patch_embeddings.shape[1], patch_embeddings.shape[2]
    # Split the data accordingly into support set and query set!  E.g. 5|75 (5way,1-shot); 25|75 (5way, 5-shot)
    patch_embeddings = patch_embeddings.view(args.n_way, -1, seq_len, emb_dim)
    emb_support, emb_query = patch_embeddings[:,:args.k_shot], patch_embeddings[:,args.k_shot:]
    return emb_support.reshape(-1, seq_len, emb_dim), emb_query.reshape(-1, seq_len, emb_dim)


def eval(args, eval_run=None):
    # Function is built upon elements from DeepEMD, CloserFSL, DINO and iBOT
    # Set seed and display args used for evaluation
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Check variables for suitability
    if args.data_path is None:
        raise ValueError("No path to dataset provided. Please do so to run experiments!")

    if args.similarity_temp_init is not None:
        assert args.similarity_temp_init > 0., "Error: Provided similarity temperature is negative or zero."

    # ============ Setting up dataset and dataloader ... ============
    DataSet = set_up_dataset(args)
    dataset = DataSet(args.set, args)
    sampler = CategoriesSampler(dataset.label, args.num_test_episodes, args.n_way, args.k_shot + args.query)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    tqdm_gen = tqdm(loader)
    print(f"\nEvaluating {args.n_way}-way {args.k_shot}-shot learning scenario.")
    print(f"Using the {args.set} set of {args.dataset} to run evaluation, averaging over "
          f"{args.num_test_episodes} episodes.")
    print(f"Data successfully loaded: There are {len(dataset)} images available to sample from.")

    # ============ Building model for evaluation and loading parameters from provided checkpoint ============
    # DeiT and ViT are the same architecture, so we change the name DeiT to ViT to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    # if the network is using multi-scale features (i.e. swin_tiny, ...)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        model = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True
        )
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, ...)
    elif args.arch in models.__dict__.keys():
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}. Please choose one that is supported.")

    # Move model to GPU
    model = model.cuda()

    # Add arguments to model for easier access
    model.args = args

    # Load weights from a checkpoint of the model to evaluate -- Note that artifact information has to be adapted!
    if args.wandb_mdl_ref:
        assert USE_WANDB, 'Enable wandb to load artifacts.'
        print('\nLoading model file from wandb artifact...')
        artifact = eval_run.use_artifact('AFRT/' + args.wandb_mdl_ref, type='model')
        artifact_dir = artifact.download()
        if args.trained_model_type == 'pretrained':
            chkpt = torch.load(artifact_dir + f'/checkpoint_ep{args.chkpt_epoch}.pth')
            # Adapt and load state dict into current model for evaluation
            chkpt_state_dict = chkpt['teacher']
        elif args.trained_model_type == 'metaft':
            chkpt = torch.load(artifact_dir + f'/meta_best.pth')
            # Adapt and load state dict into current model for evaluation
            chkpt_state_dict = chkpt['params']
            try:
                args.similarity_temp_init = chkpt['temp_sim']
            except:
                print("Could not recover (learnt) similarity temperature from model. Using provided value instead.")
        else:
            raise ValueError(f"Model type '{args.trained_model_type}' does not exist. Please check! ")
    elif args.mdl_checkpoint_path:
        print('Loading model from provided path...')
        if args.trained_model_type == 'pretrained':
            chkpt = torch.load(args.mdl_checkpoint_path + f'/checkpoint{args.chkpt_epoch:04d}.pth')
            chkpt_state_dict = chkpt['teacher']
        elif args.trained_model_type == 'metaft':
            chkpt = torch.load(args.mdl_checkpoint_path + f'/meta_best.pth')
            chkpt_state_dict = chkpt['params']
            chkpt_fsl = torch.load(args.mdl_checkpoint_path + f'/fsl_meta_best.pth')
            chkpt_state_dict_fsl = chkpt_fsl['params']
            try:
                args.similarity_temp_init = chkpt['temp_sim']
            except:
                print("Could not recover (learnt) similarity temperature from model. Using provided value instead.")
        else:
            raise ValueError(f"Model type '{args.trained_model_type}' does not exist. Please check! ")
    elif args.mdl_url:
        mdl_storage_path = os.path.join(utils.get_base_path(), 'downloaded_chkpts', f'{args.arch}', f'outdim_{out_dim}')
        download_url(url=args.mdl_url, root=mdl_storage_path, filename=os.path.basename(args.mdl_url))
        chkpt_state_dict = torch.load(os.path.join(mdl_storage_path, os.path.basename(args.mdl_url)))['state_dict']
    else:
        raise ValueError("Checkpoint not provided or provided one could not found.")
    # Adapt and load state dict into current model for evaluation
    msg = model.load_state_dict(utils.match_statedict(chkpt_state_dict), strict=False)
    seqlen_key, seqlen_qu = get_sup_emb_seqlengths(args)
    if args.arch == 'swin_tiny':
        fsl_mod_inductive = AFRT(args, seqlen_key, seqlen_qu, 768).cuda()
    else:
        fsl_mod_inductive = AFRT(args, seqlen_key, seqlen_qu).cuda()

    fsl_mod_inductive.load_state_dict(utils.match_statedict(chkpt_state_dict_fsl), strict=False)

    if args.wandb_mdl_ref or args.mdl_checkpoint_path:
        try:
            eppt = chkpt["epoch"]
        except:
            print("Epoch not recovered from checkpoint, probably using downloaded one...")
            eppt = args.chkpt_epoch
    else:
        eppt = 'unknown'
    trd = 'pretrained' if args.trained_model_type == 'pretrained' else 'meta fine-tuned'
    print(f'Parameters successfully loaded from checkpoint. '
          f'Model to be evaluated has been {trd} for {eppt} epochs.')
    if args.trained_model_type == 'metaft':
        print(f'Model has achieved a validation accuracy of {chkpt["val_acc"]:.2f}+-{chkpt["val_conf"]:.4f}!')
    print(f'Using a similarity temperature of {args.similarity_temp_init:.4f} to rescale logits.')
    print(f'Info on loaded state dict and dropped head parameters: \n{msg}')
    print("Note: If unexpected_keys other than parameters relating to the discarded 'head' exist, go and check!")
    # Save args of loaded checkpoint model into folder for reference!
    try:
        with (Path(args.output_dir) / "args_checkpoint_model.txt").open("w") as f:
            f.write(json.dumps(chkpt["args"].__dict__, indent=4))
    except:
        print("Arguments used during training not available, and will thus not be stored in eval folder.")

    # Set model to evaluation mode and freeze -- not updating of main parameters at inference time
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # Partially based on DeepEMD data loading / labelling strategy:
    # label of query images  -- Note: the order of samples provided is AAAAABBBBBCCCCCDDDDDEEEEE...!
    ave_acc = utils.Averager()
    test_acc_record = np.zeros((args.num_test_episodes,))
    label_query = torch.arange(args.n_way).repeat_interleave(args.query)
    label_query = label_query.type(torch.cuda.LongTensor)
    label_support = torch.arange(args.n_way).repeat_interleave(args.k_shot)
    label_support = label_support.type(torch.cuda.LongTensor)

    len_tqdm = len(tqdm_gen)

    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]

            # Retrieve the patch embeddings for all samples, both support and query from our backbone
            emb_support, emb_query = get_patch_embeddings(model, data, args)

            fsl_mod_inductive.eval()
            query_pred_logits, _ = fsl_mod_inductive(emb_query, emb_support, args)
            # with torch.enable_grad():
            #     # fsl_mod_inductive = PatchFSL(args, emb_support.shape[1], emb_support.shape[1])
            #     seqlen_key, seqlen_qu = get_sup_emb_seqlengths(args)
            #     fsl_mod_inductive = FART(args, seqlen_key, seqlen_qu).cuda()
            #     # Patch-based module, online adaptation using support set info, followed by prediction of query classes
            #     # query_pred_logits = fsl_mod_inductive(emb_support, emb_support, emb_query, label_support)
            #     query_pred_logits, _ = fsl_mod_inductive(emb_query, emb_support, args)

            acc = utils.count_acc(query_pred_logits, label_query) * 100
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            m, pm = utils.compute_confidence_interval(test_acc_record[:i])
            tqdm_gen.set_description('Batch {}/{}: This episode:{:.2f}  average: {:.4f}+{:.4f}'
                                     .format(i, len_tqdm, acc, m, pm))

        m, pm = utils.compute_confidence_interval(test_acc_record)
        result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
        result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print(result_list[1])

        # Upload results to wandb after evaluation is finished
        if args.use_wandb:
            eval_run.log({f'{args.set}-accuracy_mean': m, f'{args.set}-accuracy_conf': pm})
            name_str = args.dataset + f'_{args.image_size}-' + args.arch + f'-outdim_{args.out_dim}' \
                                      f'-{args.set}-ep{args.num_test_episodes}-bs{args.batch_size_total}'
            model_config = args.__dict__
            mdl_art = wandb.Artifact(name=name_str, type="results",
                                     description="Results of the evaluation.",
                                     metadata=model_config)
            try:
                with mdl_art.new_file('args_checkpoint_model.txt', 'w') as file:
                    file.write(json.dumps(chkpt["args"].__dict__, indent=4))
            except:
                print("Arguments used during training not available, and will thus not be stored in eval folder.")

            if args.trained_model_type == 'pretrained':
                res_str = f"Results achieved with {args.arch} pretrained for {args.chkpt_epoch} epochs on the " \
                          f"{args.dataset} dataset, using a batch size of {args.batch_size_total}"
            else:
                res_str = f"Results achieved with a meta fine-tuned {args.arch} architecture on {args.dataset}"
            with mdl_art.new_file('accuracy.txt', 'w') as file:
                file.write(f"{res_str}. \n"
                           f"Accuracy on {args.set} set, evaluated on {args.num_test_episodes} tasks: \t {result_list[1]}")
            # Upload to wandb
            eval_run.log_artifact(mdl_art)
            print("Artifact and results uploaded to wandb server.")
    return

def get_sup_emb_seqlengths(args):
    if args.arch in ['vit_tiny', 'vit_small', 'deit_tiny', 'deit_small']:
        raw_emb_len = (args.image_size // args.patch_size) ** 2
        support_key_seqlen = raw_emb_len
        support_query_seqlen = raw_emb_len
    elif args.arch == 'swin_tiny':
        raw_emb_len = 49
        support_key_seqlen = raw_emb_len
        support_query_seqlen = raw_emb_len
    else:
        raise NotImplementedError("Architecture currently not supported. Please check arguments, or get in contact!")
    return support_key_seqlen, support_query_seqlen


if __name__ == '__main__':
    # Parse arguments for current evaluation
    parser = argparse.ArgumentParser('evaluate', parents=[get_args_parser()])
    args = parser.parse_args()

    args.__dict__.update({'use_wandb': True} if USE_WANDB else {'use_wandb': False})

    # Create appropriate path to store evaluation results, unique for each parameter combination
    if args.wandb_mdl_ref:
        out_dim = args.wandb_mdl_ref.split('outdim_')[-1].split(':')[0]
        total_bs = args.wandb_mdl_ref.split('bs_total_')[-1].split(':')[0]
    elif args.mdl_checkpoint_path:
        out_dim = args.mdl_checkpoint_path.split('outdim_')[-1].split('/')[0]
        total_bs = args.mdl_checkpoint_path.split('bs_')[-1].split('/')[0]
    elif args.mdl_url:
        out_dim = 'unknown'
        total_bs = 'unknown'
    else:
        raise ValueError("No checkpoint provided. Cannot run evaluation!")
    args.__dict__.update({'out_dim': out_dim})
    args.__dict__.update({'batch_size_total': total_bs})

    # Set up wandb to be able to download checkpoints
    if USE_WANDB:
        eval_run = wandb.init(config=args.__dict__, project="eval", entity=WANDB_USER)
    else:
        eval_run = None

    if args.output_dir == "":
        args.output_dir = os.path.join(utils.get_base_path(), 'eval_results')

    if args.trained_model_type == 'pretrained':
        assert args.chkpt_epoch is not None, 'Checkpoint epoch must be provided for pretrained-only models!'
        epckpt = f'ep_{int(args.chkpt_epoch)+1}'
    else:
        args.chkpt_epoch = 'metaft'
        epckpt = 'metaft'
    # Creating hash to uniquely identify parameter setting for run, but w/o elements that are non-essential and
    # might change due to moving the dataset to different path, using different server, etc.
    non_essential_keys = ['num_workers', 'output_dir', 'data_path']
    exp_hash = utils.get_hash_from_args(args, non_essential_keys)
    args.output_dir = os.path.join(args.output_dir, args.dataset+f'_{args.image_size}', args.arch,
                                   f'{epckpt}', f'bs_{total_bs}', f'outdim_{out_dim}', exp_hash)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Start evaluation
    eval(args, eval_run)

    print('Evaluating your model has finished! We hope it proved successful!')

