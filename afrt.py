import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.fc_norm1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFRT(nn.Module):
    def __init__(self, args, sup_emb_key_seq_len, sup_emb_query_seq_len, in_dim=384):
        super(AFRT, self).__init__()

        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim / 4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)

        # self.fc2 = Mlp(in_features=196 ** 2, hidden_features=256, out_features=1)
        # self.fc2 = Mlp(in_features=196 ** 2, hidden_features=196, out_features=1)

        self.way = args.n_way
        self.shot = args.k_shot
        # self.total_len_support_key = self.way * sup_emb_key_seq_len  # nl
        self.total_len_support_key = self.way * self.shot * sup_emb_key_seq_len  # nkl
        # Mask to prevent image self-classification during adaptation
        if args.k_shot > 1:  # E.g. for 5-shot scenarios, use 'full' block-diagonal logit matrix to mask entire image
            self.block_mask = torch.block_diag(*[torch.ones(sup_emb_key_seq_len, sup_emb_query_seq_len) * -100.
                                                 for _ in
                                                 range(args.n_way * args.k_shot)]).cuda()  # nk×l×l沿对角线拼接为nkl×nkl
        else:  # 1-shot experiments require diagonal in-image masking, since no other samples available
            self.block_mask = torch.ones(sup_emb_key_seq_len * args.n_way * args.k_shot,
                                         sup_emb_query_seq_len * args.n_way * args.k_shot).cuda()
            self.block_mask = (self.block_mask - self.block_mask.triu(diagonal=args.block_mask_1shot)
                               - self.block_mask.tril(diagonal=-args.block_mask_1shot)) * -100.


        self.v = torch.zeros(self.total_len_support_key, requires_grad=True,
                             device='cuda')  # nkl task-specific token importance weights

        self.log_tau_c = torch.tensor([np.log(args.similarity_temp_init)], requires_grad=True, device='cuda')

        self.peiv_init_state = True
        self.disable_peiv_optimisation = args.disable_peiv_optimisation
        self.optimiser_str = args.optimiser_online
        self.opt_steps = args.optim_steps_online
        self.lr_online = args.lr_online

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.bifC = BIFC(hidden_size=in_dim, inner_size=in_dim, num_patch=sup_emb_key_seq_len, drop_prob=0.1)
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.dim = in_dim


    def forward(self, feat_query, feat_shot, args):
        # query: Q x n x C
        # feat_shot: KS x n x C
        _, n, c = feat_query.size()
        # print(feat_query.size())

        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query  # Q x n x C
        feat_shot = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot  # KS x n x C
        feat_query = self.fc_norm1(feat_query)
        feat_shot = self.fc_norm1(feat_shot)

        query_class = feat_query[:, 0, :].unsqueeze(1)  # Q x 1 x C
        query_image = feat_query[:, 1:, :]  # Q x L x C

        support_class = feat_shot[:, 0, :].unsqueeze(1)  # KS x 1 x C
        support_image = feat_shot[:, 1:, :]  # KS x L x C

        feat_query = query_image + 2.0 * query_class  # Q x L x C 原本的分类token取下来，加到原来的图像特征中
        # feat_query = query_image  # Q x L x C 原本的分类token取下来，加到原来的图像特征中
        feat_shot = support_image + 2.0 * support_class  # KS x L x C
        # feat_shot = support_image  # KS x L x C

        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)

        # feat_shot = feat_shot.contiguous().reshape(args.shot, -1, n - 1, c)  # K x S x n-1 x C
        # feat_shot = feat_shot.mean(dim=0)  # S x n-1 x C  S是类别5
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)

        label_support = torch.arange(self.way).repeat(self.shot).long().to('cuda')
        # label_support = torch.arange(self.way).repeat(self.shot)
        # label_support = label_support.type(torch.cuda.LongTensor)
        results = self.patchfsl(feat_shot, feat_shot, feat_query, label_support)
        # way,Q

        return results, None

    def patchfsl(self, support_emb_key, support_emb_query, query_emb, support_labels):
        # Check whether patch importance vector has been reset to its initialisation state
        if not self.peiv_init_state:
            self._reset_peiv()
        # Run optimisation on peiv    patch embedding importance vector
        if not self.disable_peiv_optimisation:
            self._optimise_peiv(support_emb_key, support_emb_query, support_labels)
        # Retrieve the predictions of query set samples
        pred_query = self._predict(support_emb_key, query_emb, phase='infer')
        return pred_query

    def _reset_peiv(self):
        """Reset the patch embedding importance vector to zeros"""
        # Re-create patch importance vector (and add to optimiser in _optimise_peiv() -- might exist a better option)
        self.v = torch.zeros(self.total_len_support_key, requires_grad=True, device="cuda")

    def _predict(self, support_emb, query_emb, phase='infer'):
        """Perform one forward pass using the provided embeddings as well as the module-internal
        patch embedding importance vector 'peiv'. The phase parameter denotes whether the prediction is intended for
        adapting peiv ('adapt') using the support set, or inference ('infer') on the query set."""
        sup_emb_seq_len = support_emb.shape[1]
        # Compute patch embedding similarity
        if phase == 'infer':
            # ###########################faafbb  euclide
            # support_emb = torch.add(support_emb.view(self.way * sup_emb_seq_len, -1),
            #                         self.v.unsqueeze(1)).view(self.way, self.shot * sup_emb_seq_len, -1)

            support_emb = torch.add(support_emb.view(self.way * self.shot * sup_emb_seq_len, -1),
                                    self.v.unsqueeze(1)).view(self.way, self.shot * sup_emb_seq_len, -1)
            # # 去TRSR
            # support_emb = support_emb.view(self.way, self.shot * sup_emb_seq_len, -1)

            sq_similarity, qs_similarity = self.bifC(support_emb, query_emb)
            # sq_similarity = F.normalize(sq_similarity, dim=-1)
            # qs_similarity = F.normalize(qs_similarity, dim=-1)
            l2_dist = self.w1 * sq_similarity + self.w2 * qs_similarity  # N_query x N_way
            pred = l2_dist / self.dim * self.scale

        # Mask out block diagonal during adaptation to prevent image patches from classifying themselves and neighbours
        if phase == 'adapt':
            C = compute_emb_cosine_similarity(support_emb, query_emb)  # nkl×nkl
            # C = C + self.block_mask
            # Add patch embedding importance vector (logits, corresponds to multiplication of probabilities)
            # C = C.view(self.total_len_support_key, -1)
            pred = torch.add(C, self.v.unsqueeze(1))  # using broadcasting
            # =========
            # Rearrange the patch dimensions to combine the embeddings
            pred = pred.view(self.way, self.shot * sup_emb_seq_len,
                             query_emb.shape[0], query_emb.shape[1]).transpose(2, 3)
            # Reshape to combine all embeddings related to one query image
            pred = pred.reshape(self.way, self.shot * sup_emb_seq_len * query_emb.shape[1], query_emb.shape[0])
            # Temperature scaling
            pred = pred / torch.exp(self.log_tau_c)
            # pred = pred / self.dim * self.scale
            # Gather log probs of all patches for each image
            pred = torch.logsumexp(pred, dim=1)
            pred = pred.transpose(0, 1)
            # Q,way
            # Return the predicted logits
        return pred

    def _optimise_peiv(self, support_emb_key, support_emb_query, supp_labels):
        # Detach, we don't want to compute computational graph w.r.t. model
        support_emb_key = support_emb_key.detach()
        support_emb_query = support_emb_query.detach()
        supp_labels = supp_labels.detach()
        params_to_optimise = [self.v]  # nkl维的0  task-specific token importance weights
        # Perform optimisation of patch embedding importance vector v; embeddings should be detached here!
        self.optimiser_online = torch.optim.SGD(params=params_to_optimise, lr=self.lr_online)
        self.optimiser_online.zero_grad()
        # Run for a specific number of steps 'self.opt_steps' using SGD
        for s in range(self.opt_steps):
            support_pred = self._predict(support_emb_key, support_emb_query, phase='adapt')
            loss = self.loss_fn(support_pred, supp_labels)  # (5,5) (5)
            loss.requires_grad_(True)
            loss.backward()
            self.optimiser_online.step()
            self.optimiser_online.zero_grad()
        # Set initialisation/reset flag to False since peiv is no longer 'just' initialised
        self.peiv_init_state = False
        return


def compute_emb_cosine_similarity(support_emb: torch.Tensor, query_emb: torch.Tensor):
    """Compute the similarity matrix C between support and query embeddings using the cosine similarity.
       We reformulate the cosine sim computation as dot product using matrix mult and transpose, due to:
       cossim = dot(u,v)/(u.norm * v.norm) = dot(u/u.norm, v/v.norm);    u/u.norm = u_norm, v/v.norm = v_norm
       For two matrices (tensors) U and V of shapes [n,b] & [m,b] => torch.mm(U_norm,V_norm.transpose)
       This returns a matrix showing all pairwise cosine similarities of all entries in both matrices, i.e. [n,m]"""

    # Note: support represents the 'reference' embeddings, query represents the ones that need classification;
    #       During adaptation of peiv, support set embeddings will be used for both to optimise the peiv
    support_shape = support_emb.shape  # 1,way,head*shot*l*c
    support_emb_vect = support_emb.reshape(support_shape[0] * support_shape[1], -1)  # shape e.g. [4900, 384] nkl×d
    # Robust version to avoid division by zero
    support_norm = torch.linalg.norm(support_emb_vect, dim=1).unsqueeze(dim=1)
    # support_norm = torch.linalg.vector_norm(support_emb_vect, dim=1).unsqueeze(dim=1)
    support_norm = support_emb_vect / torch.max(support_norm, torch.ones_like(support_norm) * 1e-8)

    query_shape = query_emb.shape  # (N_query x N_way x head*N-shot*l*C)
    query_emb_vect = query_emb.reshape(query_shape[0] * query_shape[1], -1)  # shape e.g. [14700, 384]
    # Robust version to avoid division by zero
    query_norm = query_emb_vect.norm(dim=1).unsqueeze(dim=1)
    query_norm = query_emb_vect / torch.max(query_norm, torch.ones_like(query_norm) * 1e-8)

    return torch.matmul(support_norm, query_norm.transpose(0, 1))  # shape e.g. [4900, 14700]


# nld nld  [nl,nl]

class BIFC(nn.Module):
    """
    Feature Mutual Reconstruction Module
    """

    def __init__(self, hidden_size, inner_size=None, num_patch=25, drop_prob=0.):
        super(BIFC, self).__init__()

        self.hidden_size = hidden_size
        self.inner_size = inner_size if inner_size is not None else hidden_size // 8
        self.num_patch = num_patch

        dim_per_head = inner_size
        self.num_heads = 6
        # self.num_heads = 1
        inner_dim = self.inner_size
        # inner_dim = self.inner_size * self.num_heads
        self.to_qkv = nn.Sequential(
            nn.Linear(self.hidden_size, inner_dim * 3, bias=False),
        )

        self.dropout = nn.Dropout(drop_prob)

    def compute_distances(self, query_a, key_a, value_a, query_b, key_b, value_b, features_a, features_b):
        # 1) feature reconstruction
        value_a = value_a.unsqueeze(0)  # 1,way*shot,head,l,c 根据输入的不同shot的位置不同  1,way,head,shot*l,c
        value_b = value_b.unsqueeze(1)  # query,1,num_heads,l,c

        n_way = value_a.size(1)
        n_query = value_b.size(0)
        n_patch_shot = value_a.size(3)
        n_patch = value_b.size(3)
        k_shot = int(n_patch_shot / n_patch)
        n_dim = value_b.size(-1)
        # s_patch = value_a.size(3)

        # Reconstructed features B
        att_scores = torch.matmul(query_b.unsqueeze(1), key_a.unsqueeze(0).transpose(-1, -2).contiguous())
        # query,1,num_heads,l,c  1,shot*way,head,c,l -> query,shot*way,head,l,l
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)
        # shot*way     (N_query x N-shot*N_way x 1 x l x l) x (1 x N-shot*N_way x 1 x l x C) -> (N_query x N-shot*N_way x h x l x C)
        reconstructed_features_b = torch.matmul(att_probs, value_a)
        # shot*l         query,way,head,l,shot*l    1,way,head,shot*l,c      query,way,head,l,c

        # Reconstructed features A
        att_scores = torch.matmul(query_a.unsqueeze(0), key_b.unsqueeze(1).transpose(-1, -2).contiguous())
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)

        # (N_query x N-shot*N_way x h x l x l) x (N_query x 1 x h x l x C) -> (N_query x N-shot*N_way x h x l x C)
        reconstructed_features_a = torch.matmul(att_probs, value_b)

        assert reconstructed_features_a.size(-1) * self.num_heads == self.inner_size
        assert reconstructed_features_b.size(-1) * self.num_heads == self.inner_size
        assert value_a.size(-1) * self.num_heads == self.inner_size
        assert value_b.size(-1) * self.num_heads == self.inner_size

        ############### sq_euclide_simi  add  ###############
        value_a = F.normalize(value_a, dim=-1)
        value_b = F.normalize(value_b, dim=-1)
        reconstructed_features_a = F.normalize(reconstructed_features_a, dim=-1)
        reconstructed_features_b = F.normalize(reconstructed_features_b, dim=-1)

        # 2) compute the Euclide distance
        # 1,way,head*shot*l*c   (N_query x N_way x head*N-shot*l*C)    N_query x N_way
        # query,1,num_heads,l,c  query,way,num_heads,l,c
        sq_similarity = -torch.sum((value_a.view(value_a.size(0), value_a.size(1), -1) - reconstructed_features_a.view(
            reconstructed_features_a.size(0), reconstructed_features_a.size(1), -1)) ** 2, dim=-1)
        qs_similarity = -torch.sum((value_b.view(value_b.size(0), value_b.size(1), -1) - reconstructed_features_b.view(
            reconstructed_features_b.size(0), reconstructed_features_b.size(1), -1)) ** 2, dim=-1)

        return sq_similarity, qs_similarity

    def forward(self, features_a, features_b):
        # projection of features a
        # features_a = features_a.view(features_a.size(0), features_a.size(1), -1).permute(0, 2, 1).contiguous()
        # way,c,shot,h,w->way,c,shot*h*w->way,shot*h*w,c

        b_a, l_a, d_a = features_a.shape  # (self.way * self.shot, sup_emb_seq_len, -1)

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv_a = self.to_qkv(features_a)
        # (3,b,num_heads,l,dim_per_head)
        qkv_a = qkv_a.view(b_a, l_a, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        query_a, key_a, value_a = qkv_a.chunk(3)
        query_a, key_a, value_a = query_a.squeeze(0), key_a.squeeze(0), value_a.squeeze(0)  # way*shot,head,l,c

        # projection of features b
        # features_b = features_b.view(features_b.size(0), features_b.size(1), -1).permute(0, 2, 1).contiguous()
        # query,c,h,w->query,h*w,c
        b_b, l_b, d_b = features_b.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv_b = self.to_qkv(features_b)
        # (3,b,num_heads,l,dim_per_head)
        qkv_b = qkv_b.view(b_b, l_b, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        query_b, key_b, value_b = qkv_b.chunk(3)
        query_b, key_b, value_b = query_b.squeeze(0), key_b.squeeze(0), value_b.squeeze(0)  # query,num_heads,l,c

        # compute the total spatial similarity
        distances = self.compute_distances(query_a, key_a, value_a, query_b, key_b, value_b, features_a, features_b)

        return distances
