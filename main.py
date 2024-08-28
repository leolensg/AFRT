import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cpea import CPEA
from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from utils import pprint, ensure_path, Averager, count_acc, compute_confidence_interval
from tensorboardX import SummaryWriter
import utils


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    # parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00001)
    # parser.add_argument('--lr', type=float, default=0.02)
    # parser.add_argument('--lr', type=float, default=0.000002)
    # 0.00001
    parser.add_argument('--lr_mul', type=float, default=100)
    # parser.add_argument('--lr_mul', type=float, default=10000)
    # 100
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='CIFAR-FS')
    parser.add_argument('--init_weights', type=str, default='./initialization/CIFAR-FS/checkpoint1600.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--exp', type=str, default='CPEA-5way-5shot')

    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'deit_tiny', 'deit_small', 'swin_tiny'],
                        help="""Name of architecture you want to evaluate.""")
    parser.add_argument('--image_size', type=int, default=224,
                        help="""Size of the squared input images, 224 for imagenet-style.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
                        of input square patches - default 16 (for 16x16 patches). Using smaller
                        values leads to better performance but requires more memory. Applies only
                        for ViTs (vit_tiny, vit_small and vit_base), but is used in swin to compute the 
                        predictor size (8*patch_size vs. pred_size = patch_size in ViT) -- so be wary!. 
                        If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--block_mask_1shot', default=5, type=int, help="""Number of patches to mask around each 
                        respective patch during online-adaptation in 1shot scenarios: masking along main diagonal,
                        e.g. size=5 corresponds to masking along the main diagonal of 'width' 5.""")
    parser.add_argument('--disable_peiv_optimisation', type=utils.bool_flag, default=False,
                        help="""Disable the patch embedding importance vector (peiv) optimisation/adaptation.
                        This means that inference is performed simply based on the cosine similarity of support and
                        query set sample embeddings, w/o any further adaptation.""")
    parser.add_argument('--optimiser_online', default='SGD', type=str, choices=['SGD'],
                        help="""Optimiser to be used for adaptation of patch embedding importance vector.""")
    parser.add_argument('--optim_steps_online', default=15, type=int, help="""Number of update steps to take to
                        optimise the patch embedding importance vector.""")
    # parser.add_argument('--lr_online', default=0.00001, type=float, help="""Learning rate used for online optimisation.""")
    parser.add_argument('--lr_online', default=0.1, type=float, help="""Learning rate used for online optimisation.""")
    parser.add_argument('--similarity_temp_init', type=float, default=0.0421,
                        help="""Initial value of temperature used for scaling the logits of the path embedding 
                        similarity matrix. Logits will be divided by that temperature, i.e. temp<1 scales up. 
                        'similarity_temp' must be positive. 0.051031036307982884""")

    parser.add_argument('-temperature_attn', type=float, default=2.0, metavar='gamma',
                        help='temperature for softmax in computing cross-attention')
    parser.add_argument('-temperature', type=float, default=0.2, metavar='tau', help='temperature for metric-based loss')


    args = parser.parse_args()
    pprint(vars(args))

    save_path = '-'.join([args.exp, args.dataset, args.model_type])
    args.save_path = osp.join('./results', save_path)
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CIFAR-FS':
        from dataloader.cifarfs import CIFARFS as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    workers = 2
    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=workers, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 100, args.test_way, args.shot + args.query)
    # val_sampler = CategoriesSampler(valset.label, 500, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=workers, pin_memory=True)

    model = BackBone(args)
    seqlen_key, seqlen_qu = get_sup_emb_seqlengths(args)
    # fsl_mod_inductive = PatchFSL(args, seqlen_key, seqlen_qu)
    dense_predict_network = CPEA(args, seqlen_key, seqlen_qu)

    optimizer = torch.optim.Adam([{'params': model.encoder.parameters()}], lr=args.lr, weight_decay=0.001)
    print('Using {}'.format(args.model_type))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    dense_predict_network_optim = torch.optim.Adam(dense_predict_network.parameters(), lr=args.lr * args.lr_mul, weight_decay=0.001)
    dense_predict_network_scheduler = torch.optim.lr_scheduler.StepLR(dense_predict_network_optim, step_size=args.step_size, gamma=args.gamma)

    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    print(model_dict.keys())
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights, map_location='cpu')['teacher']
        print(pretrained_dict.keys())
        pretrained_dict = {k.replace('backbone', 'encoder'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        # model = torch.nn.DataParallel(model)
        dense_predict_network = dense_predict_network.cuda()


    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=dense_predict_network.state_dict()), osp.join(args.save_path, name + '_dense_predict.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    global_count = 0
    writer = SummaryWriter(comment=args.save_path)
    # criterion = torch.nn.NLLLoss().cuda()

    for epoch in range(1, args.max_epoch + 1):
        # lr_scheduler.step()
        # dense_predict_network_scheduler.step()
        model.train()
        dense_predict_network.train()
        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):  # batch->[80,3,224,224]  [80]
            # zero gradient
            optimizer.zero_grad()
            dense_predict_network_optim.zero_grad()

            # forward and backward
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            feat_shot, feat_query = model(data_shot, data_query)
# feat_shot 5,197,384 , feat_query 75,197,384

            results, _ = dense_predict_network(feat_query, feat_shot, args)
            # results = torch.cat(results, dim=0)  # Q x S
            label = torch.arange(args.way).repeat(args.query).long().to('cuda')

            eps = 0.1
            one_hot = torch.zeros_like(results).scatter(1, label.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (args.way - 1)
            log_prb = F.log_softmax(results, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            # loss = criterion(log_prb, label)

            acc = count_acc(results.data, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            loss_total = loss

            loss_total.backward()
            optimizer.step()
            dense_predict_network_optim.step()

        lr_scheduler.step()
        dense_predict_network_scheduler.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()
        dense_predict_network.eval()

        vl = Averager()
        va = Averager()

        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]
                feat_shot, feat_query = model(data_shot, data_query)

                results, _ = dense_predict_network(feat_query, feat_shot, args)  # Q x S

                # results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]

                # results = torch.cat(results, dim=0)  # Q x S
                label = torch.arange(args.test_way).repeat(args.query).long().to('cuda')

                loss = F.cross_entropy(results, label)
                acc = count_acc(results.data, label)
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

    writer.close()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 300, args.test_way, args.shot + args.query)
    # sampler = CategoriesSampler(test_set.label, 1000, args.test_way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=workers, pin_memory=True)
    test_acc_record = np.zeros((300,))
    # test_acc_record = np.zeros((1000,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    dense_predict_network.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '_dense_predict.pth'))['params'])
    dense_predict_network.eval()

    ave_acc = Averager()
    label = torch.arange(args.test_way).repeat(args.query)

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.test_way * args.shot
            data_shot, data_query = data[:k], data[k:]
            feat_shot, feat_query = model(data_shot, data_query)

            results, _ = dense_predict_network(feat_query, feat_shot, args)  # Q x S
            # results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]
            # results = torch.cat(results, dim=0)  # Q x S
            label = torch.arange(args.test_way).repeat(args.query).long().to('cuda')

            acc = count_acc(results.data, label)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            print('batch {}: acc {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Epoch {}, Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
