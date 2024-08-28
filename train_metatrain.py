
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from pathlib import Path
from torchvision.datasets.utils import download_url
from tqdm import tqdm

import models
import utils
from datasets.samplers import CategoriesSampler

from torch.utils.tensorboard import SummaryWriter
from afrt import AFRT
####################################################################
USE_WANDB = False

if USE_WANDB:
    import wandb
    # Note: Make sure to specify your username for correct logging
    WANDB_USER = 'username'
####################################################################


def get_args_parser():
    parser = argparse.ArgumentParser('metatrain', add_help=False)

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
                            This config is only valid for Swin Transformer and is ignored for vanilla ViT 
                            architectures.""")

    # FSL task/scenario related parameters
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    # parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--num_episodes_per_epoch', type=int, default=100,
                        help="""Number of episodes used for 1 epoch of meta fine tuning. """)
    parser.add_argument('--num_epochs', type=int, default=100,  # 200 stated in paper, but converges much faster
                        help="""Maximum number of epochs for meta fine tuning. """)
    parser.add_argument('--num_validation_episodes', type=int, default=100,
                        help="""Number of episodes used for validation.   600""")

    # Optimisation outer loop -- meta fine-tuning
    parser.add_argument('--meta_lr', type=float, default=0.0002,
                        help="""Learning rate for meta finetuning outer loop.   def 0.0002 0.0005 0.00008""")
    parser.add_argument('--meta_momentum', type=float, default=0.9,
                        help="""Momentum for meta finetuning outer loop. """)
    parser.add_argument('--meta_weight_decay', type=float, default=0.0005,
                        help="""Weight decay for meta finetuning outer loop. """)
    parser.add_argument('--meta_lr_scale_mult', type=int, default=25,
                        help="""Multiplicative scaling of meta_lr to retrieve lr for 2nd parameter group (temp).""")
# def 25
    parser.add_argument('--meta_lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'multistep'],
                        help="""Learning rate scheduler for meta finetuning outer loop.""")
    parser.add_argument('--meta_lr_step_size', type=int, default=20,
                        help="""Step size used for step or multi-step lr scheduler. Currently not really in use.""")

    # FSL adaptation component related parameters
    parser.add_argument('--block_mask_1shot', default=5, type=int, help="""Number of patches to mask around each 
                        respective patch during online-adaptation in 1shot scenarios: masking along main diagonal,
                        e.g. size=5 corresponds to masking along the main diagonal of 'width' 5.""")

    parser.add_argument('--similarity_temp_init', type=float, default=0.0421,
                        help="""Initial value of temperature used for scaling the logits of the path embedding 
                            similarity matrix. Logits will be divided by that temperature, i.e. temp<1 scales up. 
                            'similarity_temp' must be positive. 0.051031036307982884  0.0421""")
    parser.add_argument('--meta_learn_similarity_temp', type=utils.bool_flag, default=False,
                        help="""If true, the temperature rescaling the logits in the patch similarity matrix will be 
                             learnt; staring from the initial value provided before.""")

    # Adaptation component -- Optimisation related parameters
    parser.add_argument('--optimiser_online', default='SGD', type=str, choices=['SGD'],
                        help="""Optimiser to be used for adaptation of patch embedding importance vector.""")
    parser.add_argument('--lr_online', default=0.1, type=float, help="""Learning rate used for online optimisation.""")
    # default=0.1
    parser.add_argument('--optim_steps_online', default=20, type=int, help="""Number of update steps to take to
                                optimise the patch embedding importance vector.""")
    # True default=False
    parser.add_argument('--disable_peiv_optimisation', type=utils.bool_flag, default=False,
                        help="""Disable the patch embedding importance vector (peiv) optimisation/adaptation.
                                This means that inference is performed simply based on the cosine similarity of support and
                                query set sample embeddings, w/o any further adaptation.""")

    # Dataset related parameters
    parser.add_argument('--image_size', type=int, default=224,
                        help="""Size of the squared input images, 224 for imagenet-style.""")
    parser.add_argument('--dataset', default='miniimagenet', type=str,
                        choices=['miniimagenet', 'tieredimagenet', 'fc100', 'cifar_fs'],
                        help='Please specify the name of the dataset to be used for training.')
    parser.add_argument('--data_path', required=False, type=str, default= 'D:/project_file/python/dataset/miniimagenet',
                        help='Please specify path to the root folder containing the training dataset(s). If dataset '
                             'cannot be loaded, check naming convention of the folder in the corresponding dataloader.'
                             ' D:/project_file/python/dataset/miniimagenet'
                             'cifar fc100 tired D:/project_file/python/dataset')

    # Misc
    # Checkpoint to load
    parser.add_argument('--wandb_mdl_ref', required=False, type=str,
                        help="""Complete path/tag of model at wandb if it is to be loaded from there.""")
    parser.add_argument('--mdl_checkpoint_path', required=False, type=str, default='initialization/miniimagenet',
                        help="""Path to checkpoint of model to be loaded. Actual checkpoint given via chkpt_epoch.
                        CIFAR-FS miniimagenet tieredimagenet tieredimagenet_swin fc100""")
    parser.add_argument('--mdl_url', required=False, type=str,
                        help='URL from where to load the model weights or checkpoint, e.g. from pre-trained publically '
                             'available ones.')
    parser.add_argument('--chkpt_epoch', default=1600, type=int, help="""Number of epochs of pretrained 
                            model to be loaded for evaluation.  1600  800""")

    # Misc
    parser.add_argument('--output_dir', default="", type=str, help="""Root path where to save correspondence images. 
                            If left empty, results will be stored in './meta/...'.""")
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
    patch_embeddings = model(data)
    # patch_embeddings = model(data)[:, 1:]
    # temp size values for reshaping
    bs, seq_len, emb_dim = patch_embeddings.shape[0], patch_embeddings.shape[1], patch_embeddings.shape[2]
    # Split the data accordingly into support set and query set!  E.g. 5|75 (5way,1-shot); 25|75 (5way, 5-shot)
    patch_embeddings = patch_embeddings.view(args.n_way, -1, seq_len, emb_dim)
    emb_support, emb_query = patch_embeddings[:, :args.k_shot], patch_embeddings[:, args.k_shot:]
    return emb_support.reshape(-1, seq_len, emb_dim), emb_query.reshape(-1, seq_len, emb_dim)


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


def run_validation(model, patchfsl, data_loader, args, epoch):
    model.eval()
    patchfsl.eval()
    # Create labels and loggers
    label_query = torch.arange(args.n_way).repeat_interleave(args.query)  # Samples arranged in an 'aabbccdd' fashion
    label_query = label_query.type(torch.cuda.LongTensor)
    label_support = torch.arange(args.n_way).repeat_interleave(args.k_shot)  # Samples arranged in an 'aabbccdd' fashion
    label_support = label_support.type(torch.cuda.LongTensor)
    val_ave_acc = utils.Averager()
    val_acc_record = np.zeros((args.num_validation_episodes,))
    val_ave_loss = utils.Averager()
    val_loss_record = np.zeros((args.num_validation_episodes,))
    val_tqdm_gen = tqdm(data_loader)
    # Run validation
    with torch.no_grad():
        for i, batch in enumerate(val_tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]
            # Retrieve the patch embeddings for all samples, both support and query from Transformer backbone
            emb_support, emb_query = get_patch_embeddings(model, data, args)

            query_pred_logits, _ = patchfsl(emb_query, emb_support, args)
            # with torch.enable_grad():
            #     # optimise patch importance weights based on support set information and predict query logits
            #     # query_pred_logits = patchfsl(emb_support, emb_support, emb_query, label_support)
            #     query_pred_logits, _ = patchfsl(emb_query, emb_support, args)

            loss = F.cross_entropy(query_pred_logits, label_query)

            val_acc = utils.count_acc(query_pred_logits, label_query) * 100
            val_ave_acc.add(val_acc)
            val_acc_record[i - 1] = val_acc
            m, pm = utils.compute_confidence_interval(val_acc_record[:i])
            val_ave_loss.add(loss)
            val_loss_record[i - 1] = loss
            m_loss, _ = utils.compute_confidence_interval(val_loss_record[:i])
            val_tqdm_gen.set_description(
                'Ep {} | batch {}: Loss epi:{:.2f} avg: {:.4f} | Acc: epi:{:.2f} avg: {:.4f}+{:.4f}'
                .format(epoch, i, loss, m_loss, val_acc, m, pm))
        # Compute stats of finished epoch
        m, pm = utils.compute_confidence_interval(val_acc_record)
        m_loss, _ = utils.compute_confidence_interval(val_loss_record)
        result_list = ['Ep {} | Overall Validation Loss {:.4f} | Validation Acc {:.4f}'
                        .format(epoch, val_ave_loss.item(), val_ave_acc.item())]
        result_list.append(
            'Ep {} | Validation Loss {:.4f} | Validation Acc {:.4f} + {:.4f}'.format(epoch, m_loss, m, pm))
        print(f'{result_list[1]}')
        # Return validation accuracy for this epoch
        return m, pm, m_loss


def metatrain(args, wandb_run):
    # Function is built upon elements from DeepEMD, CloserFSL, DINO and iBOT
    # Set seed and display args used for evaluation
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Check variables for suitability
    if args.data_path is None:
        raise ValueError("No path to dataset provided. Please do so to run experiments!")

    if args.similarity_temp_init is not None:
        assert args.similarity_temp_init > 0., "Error: Provided initial similarity temperature is negative or zero."

    # ============ Setting up dataset and dataloader ... ============
    DataSet = set_up_dataset(args)
    train_dataset = DataSet('train', args)
    train_sampler = CategoriesSampler(train_dataset.label, args.num_episodes_per_epoch, args.n_way,
                                               args.k_shot + args.query)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                               num_workers=args.num_workers, pin_memory=True)
    print(f"\nTraining {args.n_way}-way {args.k_shot}-shot learning scenario.")
    print(f"Using {args.num_episodes_per_epoch} episodes per epoch, training for {args.num_epochs} epochs.")
    print(f"Data successfully loaded: There are {len(train_dataset)} images available for training.")

    val_dataset = DataSet('val', args)
    # val_dataset = DataSet('test', args)
    val_sampler = CategoriesSampler(val_dataset.label, args.num_validation_episodes, args.n_way,
                                             args.k_shot + args.query)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler,
                                             num_workers=args.num_workers, pin_memory=True)
    print(f"\nValidating using {args.num_validation_episodes} episodes.")

    # ============ Building model loading parameters from provided checkpoint ============
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

    # Load weights from a checkpoint of the model to meta train -- Note that artifact information has to be adapted!
    if args.wandb_mdl_ref:
        assert USE_WANDB, 'Enable wandb to load artifacts.'
        print('\nLoading model file from wandb artifact...')
        artifact = wandb_run.use_artifact('AFRT/' + args.wandb_mdl_ref, type='model')
        artifact_dir = artifact.download()
        chkpt = torch.load(artifact_dir + f'/checkpoint_ep{args.chkpt_epoch}.pth')
        # Adapt and load state dict into current model for evaluation
        chkpt_state_dict = chkpt['teacher']
    elif args.mdl_checkpoint_path:
        print('Loading model from provided path...')
        chkpt = torch.load(args.mdl_checkpoint_path + f'/checkpoint{args.chkpt_epoch:04d}.pth')
        chkpt_state_dict = chkpt['teacher']
    elif args.mdl_url:
        mdl_storage_path = os.path.join(utils.get_base_path(), 'downloaded_chkpts', f'{args.arch}', f'outdim_{out_dim}')
        download_url(url=args.mdl_url, root=mdl_storage_path, filename=os.path.basename(args.mdl_url))
        chkpt_state_dict = torch.load(os.path.join(mdl_storage_path, os.path.basename(args.mdl_url)))['state_dict']
    else:
        raise ValueError("Checkpoint not provided or provided one could not be found.")
    # Adapt and load state dict into current model for evaluation
    msg = model.load_state_dict(utils.match_statedict(chkpt_state_dict), strict=False)
    # # =============================去掉位置编码==========================================================
    # model.pos_embed = None
    if args.wandb_mdl_ref or args.mdl_checkpoint_path:
        try:
            eppt = chkpt["epoch"]
        except:
            print("Epoch not recovered from checkpoint, probably using downloaded one...")
            eppt = args.chkpt_epoch
    else:
        eppt = 'unknown'
    print(f'Parameters successfully loaded from checkpoint. '
          f'Model to be meta fine-tuned has been trained for {eppt} epochs.')
    print(f'Info on loaded state dict and dropped head parameters: \n{msg}')
    print("Note: If unexpected_keys other than parameters relating to the discarded 'head' exist, go and check!")
    # Save args of loaded checkpoint model into folder for reference!
    try:
        with (Path(args.output_dir) / "args_checkpoint_pretrained_model.txt").open("w") as f:
            f.write(json.dumps(chkpt["args"].__dict__, indent=4))
    except:
        print("Arguments used during training not available, and will thus not be stored in eval folder.")

    # ============= Building the patchFSL online adaptation and classification module =================================
    seqlen_key, seqlen_qu = get_sup_emb_seqlengths(args)
    # fsl_mod_inductive = PatchFSL(args, seqlen_key, seqlen_qu)
    if args.arch == 'swin_tiny':
        fsl_mod_inductive = AFRT(args, seqlen_key, seqlen_qu, 768).cuda()
    else:
        fsl_mod_inductive = AFRT(args, seqlen_key, seqlen_qu).cuda()

    # ============= Building the optimiser for meta fine-tuning and assigning the parameters ==========================
    param_to_meta_learn = [{'params': model.parameters()}]
    if args.meta_learn_similarity_temp:
        param_to_meta_learn.append(
            {'params': fsl_mod_inductive.log_tau_c, 'lr': args.meta_lr * args.meta_lr_scale_mult})
        print("Meta learning the temperature for scaling the logits.")
    meta_optimiser = torch.optim.SGD(param_to_meta_learn,
                                     lr=args.meta_lr,
                                     momentum=args.meta_momentum,
                                     nesterov=True,
                                     weight_decay=args.meta_weight_decay)
    # ==============================================  add  ===================================
    fsl_optimiser = torch.optim.SGD(fsl_mod_inductive.parameters(),
                                     lr=args.meta_lr * args.meta_lr_scale_mult,
                                     momentum=args.meta_momentum,
                                     nesterov=True,
                                     weight_decay=args.meta_weight_decay)

    if args.meta_lr_scheduler == 'step':
        meta_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            meta_optimiser,
            step_size=int(args.meta_lr_step_size),
            gamma=0.5
            # gamma=args.gamma
        )
    elif args.meta_lr_scheduler == 'multistep':
        meta_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            meta_optimiser,
            milestones=[int(_) for _ in args.meta_lr_step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.meta_lr_scheduler == 'cosine':  # default for our application
        meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimiser,
            # T_max=args.num_epochs,
            T_max=args.num_epochs * args.num_episodes_per_epoch,
            # T_max=50 * args.num_episodes_per_epoch,
            eta_min=0
        )
        # =====================================add=====================================
        fsl_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            fsl_optimiser,
            # T_max=args.num_epochs,
            T_max=args.num_epochs * args.num_episodes_per_epoch,
            # T_max=50 * args.num_episodes_per_epoch,
            eta_min=0
        )
    else:
        raise ValueError('No Such Scheduler')

    # Partially based on DeepEMD data loading / labelling strategy:
    # label of query images  -- Note: the order of samples provided is AAAAABBBBBCCCCCDDDDDEEEEE...!
    label_query = torch.arange(args.n_way).repeat_interleave(args.query)
    label_query = label_query.type(torch.cuda.LongTensor)
    label_support = torch.arange(args.n_way).repeat_interleave(args.k_shot)
    label_support = label_support.type(torch.cuda.LongTensor)

    best_val_acc = 0.
    # ============================================================
    print(args.output_dir)
    writer = SummaryWriter(comment=args.output_dir)
    global_count = 0

    for epoch in range(1, args.num_epochs + 1):
        train_tqdm_gen = tqdm(train_loader)
        model.train()
        fsl_mod_inductive.train()
        train_ave_acc = utils.Averager()
        train_acc_record = np.zeros((args.num_episodes_per_epoch,))
        train_ave_loss = utils.Averager()
        train_loss_record = np.zeros((args.num_episodes_per_epoch,))
        ttl_num_batches = len(train_tqdm_gen)

        for i, batch in enumerate(train_tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]
            # Retrieve the patch embeddings for all samples, both support and query from Transformer backbone
            emb_support, emb_query = get_patch_embeddings(model, data, args)
            # Run patch-based module, online adaptation using support set info, followed by prediction of query classes
            # query_pred_logits = fsl_mod_inductive(emb_support, emb_support, emb_query, label_support)
            query_pred_logits, _ = fsl_mod_inductive(emb_query, emb_support, args)

            loss = F.cross_entropy(query_pred_logits, label_query)
            meta_optimiser.zero_grad()
            fsl_optimiser.zero_grad()
            loss.backward()
            meta_optimiser.step()
            fsl_optimiser.step()
            meta_lr_scheduler.step()
            fsl_lr_scheduler.step()

            train_acc = utils.count_acc(query_pred_logits, label_query) * 100
            train_ave_acc.add(train_acc)
            train_acc_record[i - 1] = train_acc
            m, pm = utils.compute_confidence_interval(train_acc_record[:i])
            train_ave_loss.add(loss)
            train_loss_record[i - 1] = loss
            m_loss, _ = utils.compute_confidence_interval(train_loss_record[:i])
            train_tqdm_gen.set_description(
                'Ep {} | bt {}/{}: Loss epi:{:.2f} avg: {:.4f} | Acc: epi:{:.2f} avg: {:.4f}+{:.4f}'.format(epoch, i,
                                                                                                            ttl_num_batches,
                                                                                                            loss,
                                                                                                            m_loss,
                                                                                                            train_acc,
                                                                                                            m, pm))
            # # =============================== 太密集了
            # global_count = global_count + 1
            # writer.add_scalar('data/loss', float(loss), global_count)
            # writer.add_scalar('data/train_acc', float(train_acc), global_count)

        m, pm = utils.compute_confidence_interval(train_acc_record)
        m_loss, _ = utils.compute_confidence_interval(train_loss_record)

        # ==========================================================================
        global_count = global_count + 1
        writer.add_scalar('data/loss', float(m_loss), global_count)
        writer.add_scalar('data/train_acc', float(m), global_count)

        result_list = ['Ep {} | Overall Train Loss {:.4f} | Train Acc {:.4f}'.format(epoch, train_ave_loss.item(),
                                                                                    train_ave_acc.item())]
        result_list.append('Ep {} | Train Loss {:.4f} | Train Acc {:.4f} + {:.4f}'.format(epoch, m_loss, m, pm))
        print(result_list[1])
        if args.meta_learn_similarity_temp:
            print(f'Ep {epoch} | Temperature {np.exp(fsl_mod_inductive.log_tau_c.item()):.4f}')
        else:
            print(f'Ep {epoch} | Using fixed temperature {np.exp(fsl_mod_inductive.log_tau_c.item()):.4f}')

        # Log stats to wandb
        if args.use_wandb:
            log_stats = {'acc_train_mean': m, 'acc_train_conf': pm, 'loss_train': m_loss, 'epoch': epoch}
            # if args.meta_learn_similarity_temp:   # Decided to always log it, even if it's constant
            log_stats.update({'sim_temp': np.exp(fsl_mod_inductive.log_tau_c.item())})
            log_stats.update({'meta_lr': meta_lr_scheduler.get_last_lr()[0]})

        print("Validating model...")
        val_acc, val_conf, val_loss = run_validation(model, fsl_mod_inductive, val_loader, args, epoch)
        # ===================================================================
        writer.add_scalar('data/val_loss', float(val_loss), epoch)
        writer.add_scalar('data/val_acc', float(val_acc), epoch)

        if val_acc > best_val_acc:
            torch.save(
                dict(params=model.state_dict(), val_acc=val_acc, val_conf=val_conf, val_loss=val_loss, epoch=epoch,
                     temp_sim=np.exp(fsl_mod_inductive.log_tau_c.item())),
                os.path.join(args.output_dir, 'meta_best.pth'))
            torch.save(dict(params=fsl_mod_inductive.state_dict()),
                       os.path.join(args.output_dir, 'fsl_meta_best.pth'))
            with (Path(args.output_dir) / "val_acc.txt").open("w") as f:
                f.write(json.dumps(dict(val_acc=val_acc, val_conf=val_conf, val_loss=val_loss, epoch=epoch), indent=4))
            best_val_acc = val_acc
            print(f"Best validation acc: {val_acc} +- {val_conf}")
        print("Finished validation, running next epoch.\n")
        # Log to wandb
        if args.use_wandb:
            log_stats.update({'acc_val_mean': val_acc, 'acc_val_conf': val_conf, 'loss_val': val_loss})
            wandb_run.log(log_stats)

    # ===========================================================================
    writer.close()

    # Upload results to wandb after fine-tuning is finished
    if args.use_wandb:
        name_str = args.dataset + f'_{args.image_size}-' + f'{args.k_shot}shot-' + args.arch + \
                   f'-outdim_{args.out_dim}' + f'-bs{args.batch_size_total}'
        model_config = args.__dict__
        mdl_art = wandb.Artifact(name=name_str, type="results",
                                 description="Results of the evaluation.",
                                 metadata=model_config)
        try:
            with mdl_art.new_file('args_pretrained_model.txt', 'w') as file:
                file.write(json.dumps(chkpt["args"].__dict__, indent=4))
        except:
            pass
        # Load best validation checkpoint (previously stored to disk)
        mdl_final = torch.load(os.path.join(args.output_dir, 'meta_best.pth'))
        with mdl_art.new_file(f'meta_best.pth', 'wb') as file:
            torch.save(mdl_final, file)
        with mdl_art.new_file('accuracy.txt', 'w') as file:
            file.write(
                f"Results achieved with {args.arch} pretrained for {args.chkpt_epoch} using a batch size "
                f"of {args.batch_size_total}. \n"
                f"Best validation accuracy has been achieved after {mdl_final['epoch']} episodes of fine-tuning. \n"
                f"Validation acc: {mdl_final['val_acc']} +- {mdl_final['val_conf']}.")
        # Upload to wandb
        wandb_run.log_artifact(mdl_art)
        print("Artifact and results uploaded to wandb server.")
    return


if __name__ == '__main__':
    # Parse arguments for current evaluation
    parser = argparse.ArgumentParser('metatrain', parents=[get_args_parser()])
    args = parser.parse_args()

    # Check whether wandb is going to be used
    args.__dict__.update({'use_wandb': True} if USE_WANDB else {'use_wandb': False})

    # Create appropriate path to store evaluation results, unique for each parameter combination
    if args.wandb_mdl_ref:
        assert args.use_wandb, 'Wandb must be active to load models from existing artifacts!'
        out_dim = args.wandb_mdl_ref.split('outdim_')[-1].split(':')[0]
        total_bs = args.wandb_mdl_ref.split('bs_total_')[-1].split(':')[0]
    elif args.mdl_checkpoint_path:
        out_dim = args.mdl_checkpoint_path.split('outdim_')[-1].split('/')[0]
        total_bs = args.mdl_checkpoint_path.split('bs_')[-1].split('/')[0]
    elif args.mdl_url:
        out_dim = 'unknown'
        total_bs = 'unknown'
    else:
        raise ValueError("No checkpoint provided. Cannot run meta fine-tuning procedure!")
    args.__dict__.update({'out_dim': out_dim})
    args.__dict__.update({'batch_size_total': total_bs})

    # Set output directory
    if args.output_dir == '':
        # args.output_dir = os.path.join(utils.get_base_path(), 'meta')
        args.output_dir = os.path.join('meta')

    # Creating hash to uniquely identify parameter setting for run, but w/o elements that are non-essential and
    # might change due to moving the dataset to different path, using different server, etc.
    non_essential_keys = ['num_workers', 'output_dir', 'data_path']
    exp_hash = utils.get_hash_from_args(args, non_essential_keys)
    args.output_dir = os.path.join(args.output_dir, args.dataset + f'_{args.image_size}', args.arch,
                                   exp_hash)
    # args.output_dir = os.path.join(args.output_dir, args.dataset + f'_{args.image_size}', args.arch,
    #                                f'ep_{args.chkpt_epoch+1}', f'bs_{total_bs}', f'outdim_{out_dim}', exp_hash)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set up wandb to be able to download checkpoints
    if USE_WANDB:
        wandb_run = wandb.init(config=args.__dict__, project="meta", entity=WANDB_USER)
    else:
        wandb_run = None

    # DataSet = set_up_dataset(args)
    # train_dataset = DataSet('train', args)
    # train_sampler = CategoriesSampler(train_dataset.label, args.num_episodes_per_epoch, args.n_way,
    #                                   args.k_shot + args.query)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
    #                                            num_workers=args.num_workers, pin_memory=True)
    # print(len(train_loader))

    # Start meta training
    metatrain(args, wandb_run)

    print('Meta-training has finished! Happy testing!')

