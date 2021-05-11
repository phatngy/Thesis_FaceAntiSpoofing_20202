
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

P_INF = float('inf')
N_INF = float('-inf')

class CRLoss(nn.Module):
    def __init__(self, nclass, comparison='relative', p=None, k=None, margin=1., device='cpu'):
        super(CRLoss, self).__init__()
        self.p = p or 1./nclass
        self.k = k
        self.nclass = nclass
        self.margin = margin
        if comparison == 'relative':     #['relative', 'absolute', 'distribution']
            self.func = self._triplet_loss
        elif comparison == 'absolute':
            self.func = self._constrastive_loss
        elif comparison == 'distribution':
            self.func = self._distribution_loss
        else:
            self.func = self._triplet_loss
        self.device = device
    
    def _get_minor_samples(self, target, onehot=False):
        '''
        Params:
            target: groundtruth shape | Tensor(batch_size, )
        Return:
            mask_minor_samples: binary tensor set for samples in minor class | Tensor(batch_size, nclass)
        '''
        batch_size = target.size()[0]
        mask_class = torch.Tensor([False]*self.nclass).bool().to(self.device)
        boundary = int(self.p*batch_size)
        uni, cnt = torch.unique(target, return_counts=True)
        mask_class[uni] = (cnt < boundary)# & (cnt > 1)
        mask_sparse = mask_class[target]
        if not onehot:
            return mask_sparse
        mask_onehot = torch.zeros((batch_size, self.nclass)).bool().to(self.device)
        mask_onehot.scatter_(1,target.view(-1,1),True)
        mask_minor_samples = mask_onehot & mask_sparse.view(-1,1)
        
        return mask_minor_samples

    def _triplet_loss(self, mask_ap, mask_an, distance_ap, distance_an):
        # Compute a 3D/4D tensor of size (batch_size, batch_size, batch_size[, nclass])
        # triplet_loss[i, j, k[, c]] will contain the triplet loss of anchor=i, positive=j, negative=k[, class=c]
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1[, nclass])
        # and the 2nd (batch_size, 1, batch_size[, nclass])
        distance_ap = distance_ap.unsqueeze(2)
        distance_an = distance_an.unsqueeze(1)
        triplet_loss = distance_ap - distance_an + self.margin

        # Put to zero the invalid triplets
        mask_ap = mask_ap.unsqueeze(2)
        mask_an = mask_an.unsqueeze(1)
        mask = mask_ap & mask_an
        triplet_loss = mask.float() * triplet_loss


        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss > 1e-7
        num_positive_triplets = valid_triplets.sum().float()

        # num_valid_triplets = mask.sum()
        # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        loss = triplet_loss.sum() / (num_positive_triplets + 1e-7)

        return loss #, fraction_positive_triplets
    

    def _constrastive_loss(self, mask_ap, mask_an, distance_ap, distance_an):
        loss_pos = distance_ap**2
        loss_pos = mask_ap.float() * loss_pos
        loss_pos = loss_pos.sum() / mask_ap.sum().float()
        
        loss_neg = self.margin - distance_an
        loss_neg[loss_neg < 0] = 0
        loss_neg = loss_neg**2
        loss_neg = mask_an.float() * loss_neg
        positive_an = loss_neg > 1e-7
        loss_neg = loss_neg.sum() / positive_an.sum().float()

        return (torch.sqrt(loss_pos) + torch.sqrt(loss_neg)) / 2

    def _distribution_loss(self, mask_ap, mask_an, distance_ap, distance_an):
        pass

    def forward(self, *input):
        raise NotImplementedError
    
    def _get_anchor_hard_sample_mask(self, *input):
        raise NotImplementedError

    def _get_anchor_sample_distance(self, *input):
        raise NotImplementedError

class CRClassLoss(CRLoss):
    def __init__(self, size_average=True, **kwargs):
        super(CRClassLoss, self).__init__(**kwargs)
        self.size_average = size_average

    def forward(self, input, target):
        '''
        Params:
            input: logits prediction | Tensor(batch_size, nclass)
            target: groundtruth | Tensor(batch_size, )
        Return:
            value: Loss value | tensor()
        '''
        
        # anchor-hard sample pairwise-classwise distance
        distance_anchor_positive, distance_anchor_negative = \
                self._get_anchor_sample_distance(input)

        # get mask pair of anchor and hard_sample
        mask_anchor_positive, mask_anchor_negative = \
                self._get_anchor_hard_sample_mask(input, target)

        # calculate loss 
        loss = self.func(mask_anchor_positive,
                        mask_anchor_negative,
                        distance_anchor_positive,
                        distance_anchor_negative)

        return loss

    def _get_topk_hard_positives(self, score, target, k=None):
        '''
        Params:
            score: logits prediction | Tensor(batch_size, nfeat)
            target: groundtruth | Tensor(batch_size, )
            k: number of samples would be returned
        Return:
            mask_hard_samples | Tensor(batch_size, nfeat)
        '''
        # batch_size, nfeat = score.size()
        mask_hard_samples = torch.zeros_like(score).bool()
        onehot_target = torch.zeros_like(score).bool()
        onehot_target.scatter_(1,target.view(-1,1),True)
        
        if k:
            valid_score = score.clone()
            valid_score[~onehot_target] = P_INF                     # score=inf is invalid
            values, indices = valid_score.topk(k, 0, False)         # choose k smallest scores
            mask_hard_samples.scatter_(0,indices,True)              # mark position in mask
            mask_hard_samples = mask_hard_samples & onehot_target   # remove invalid samples

        else:   # get all
            mask_hard_samples = onehot_target
        return mask_hard_samples

    def _get_topk_hard_negatives(self, score, target, k=None):
        '''
        Params:
            score: logits prediction | Tensor(batch_size, nfeat)
            target: groundtruth | Tensor(batch_size, )
            k: number of samples would be returned
        Return:
            mask_hard_samples | Tensor(batch_size, nfeat)
        '''
        batch_size, nfeat = score.size()
        mask_hard_samples = torch.zeros_like(score).bool()
        onehot_target = torch.zeros_like(score).bool()
        onehot_target.scatter_(1,target.view(-1,1),True)
        if k:
            valid_score = score.clone()                                     
            valid_score[onehot_target] = N_INF                          # score=-inf is invalid
            values, indices = valid_score.topk(k, 0, True)              # choose k largest scores
            mask_hard_samples.scatter_(0,indices,True)                  # mark position in mask
            mask_hard_samples = mask_hard_samples & ~onehot_target      # remove invalid samples

        else:   # get all
            mask_hard_samples = ~onehot_target
        return mask_hard_samples

    def _get_anchor_hard_sample_mask(self, score, target):
        batch_size, nclass = score.size()
        # get topk hard samples
        mask_hard_positives = self._get_topk_hard_positives(score, target, self.k)
        mask_hard_negatives = self._get_topk_hard_negatives(score, target, self.k)
        # get minor sample
        mask_anchors = self._get_minor_samples(target, onehot=True)
        # anchor-hard sample pairwise-classwise mask
        mask_anchor_positive = mask_anchors.unsqueeze(1) & mask_hard_positives.unsqueeze(0)
        indices_not_equal = ~torch.eye(batch_size).bool().unsqueeze(-1).to(self.device)
        mask_anchor_positive = indices_not_equal & mask_anchor_positive
        mask_anchor_negative = mask_anchors.unsqueeze(1) & mask_hard_negatives.unsqueeze(0)
        
        return mask_anchor_positive, mask_anchor_negative

    def _get_anchor_sample_distance(self, input):
        input = F.softmax(input, dim=-1)
        distance_anchor_positive = torch.abs(input.unsqueeze(1)-input.unsqueeze(0))
        distance_anchor_negative = input.unsqueeze(1)-input.unsqueeze(0)
        return distance_anchor_positive, distance_anchor_negative
    

class CRInstanceLoss(CRLoss):
    def __init__(self, size_average=True, **kwargs):
        super(CRInstanceLoss, self).__init__(**kwargs)
        self.size_average = size_average
        
    def forward(self, input, target):
        '''
        Params:
            input: logits prediction | Tensor(batch_size, nclass)
            target: groundtruth | Tensor(batch_size, )
        Return:
            value: Loss value | tensor()
        '''

        # anchor-hard sample pairwise-classwise distance
        distance_anchor_positive, distance_anchor_negative = \
                self._get_anchor_sample_distance(input)
        # get mask pair of anchor and hard_sample
        mask_anchor_positive, mask_anchor_negative = \
                self._get_anchor_hard_sample_mask(distance_anchor_positive, target)
        
        # calculate loss 
        loss = self.func(mask_anchor_positive,
                        mask_anchor_negative,
                        distance_anchor_positive,
                        distance_anchor_negative,)
        return loss

    def _get_topk_hard_positives(self, score, target, k=None):
        '''
        Params:
            score: distance matrix | Tensor(batch_size, batch_size)
            target: groundtruth | Tensor(batch_size, )
            k: number of samples would be returned
        Return:
            mask_hard_samples | Tensor(batch_size, batch_size)
        '''
        batch_size, nfeat = score.size()
        mask_hard_samples = torch.zeros_like(score).bool()
        similarity_target = target.unsqueeze(0) == target.unsqueeze(1)

        if k:
            valid_score = score.clone()
            valid_score[~similarity_target] = N_INF                     # score=-inf is invalid
            values, indices = valid_score.topk(k, 0, True)              # choose k largest scores
            mask_hard_samples.scatter_(0,indices,True)                  # mark position in mask
            mask_hard_samples = mask_hard_samples & similarity_target   # remove invalid samples

        else:   # get all
            mask_hard_samples = similarity_target
        return mask_hard_samples

    def _get_topk_hard_negatives(self, score, target, k=None):
        '''
        Params:
            score: distance matrix | Tensor(batch_size, nfeat)
            target: groundtruth | Tensor(batch_size, )
            k: number of samples would be returned
        Return:
            mask_hard_samples | Tensor(batch_size, nfeat)
        '''
        batch_size, nfeat = score.size()
        mask_hard_samples = torch.zeros_like(score).bool()
        similarity_target = target.unsqueeze(0) == target.unsqueeze(1)
        if k:
            valid_score = score.clone()                                     
            valid_score[similarity_target] = P_INF                      # score=inf is invalid
            values, indices = valid_score.topk(k, 0, False)             # choose k smallest scores
            mask_hard_samples.scatter_(0,indices,True)                  # mark position in mask
            mask_hard_samples = mask_hard_samples & ~similarity_target  # remove invalid samples

        else:   # get all
            mask_hard_samples = ~similarity_target
            # print(mask_hard_samples.sum())
        return mask_hard_samples

    def _get_anchor_hard_sample_mask(self, score, target):
        batch_size, nclass = score.size()
        # get topk hard samples
        mask_hard_positives = self._get_topk_hard_positives(score, target, self.k)
        mask_hard_negative = self._get_topk_hard_negatives(score, target, self.k)
        # get minor sample
        mask_anchors = self._get_minor_samples(target, onehot=False)   # batchsize,
        # anchor-hard sample pairwise mask
        mask_anchor_positive = mask_anchors.unsqueeze(1) & mask_hard_positives
        indices_not_equal = ~torch.eye(batch_size).bool().to(self.device)
        mask_anchor_positive = indices_not_equal & mask_anchor_positive
        mask_anchor_negative = mask_anchors.unsqueeze(1) & mask_hard_negative
        
        return mask_anchor_positive, mask_anchor_negative

    def _get_anchor_sample_distance(self, input):
        pairwise_distances = self._pairwise_distances(input, squared=False)
        return pairwise_distances, pairwise_distances
    
    def _pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 -mask) * torch.sqrt(distances)

        return distances
    


if __name__ == "__main__":
    
    device = 'cuda'
    weights = torch.tensor([1,5,10,60,2,10,5,3,2,2], dtype=torch.float)
    target = torch.multinomial(weights, 100, replacement=True)
    # print(target)
    input = (target.view(-1,1) == torch.arange(10).view(1,-1)).float()
    input += (torch.rand_like(input)-0.5)
    input = input.to(device)
    target = target.to(device)
    onehot = target.unsqueeze(1) == torch.arange(10).view(1,-1).to(device)
    emb = torch.rand(100,128).to(device)*0.1
    emb[:,:10] += onehot

    # class-level, relative
    # l1 = CRClassLoss(nclass=10, comparison='absolute', p=0.5, k=None, margin=.5, device=device)
    # v1 = l1(input, target)
    # print(v1)
    # instance-level, relative
    l2 = CRInstanceLoss(nclass=10, comparison='absolute', p=0.5, k=10, margin=1.5, device=device)
    v2 = l2(emb, target)
    print(v2)
