import copy
import math
import numpy as np
from select import select
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from .AL import AL


class ALFAMix(AL):
    def __init__(self, model, unlabeled_dst, ul, train, train_dataset, logit_scale, best_head, args, n_class, device, **kwargs):
        super().__init__(model, unlabeled_dst, n_class, **kwargs)
        self.device = device
        self.ul = ul
        self.train_data = train_dataset
        self.logit_scale = logit_scale
        self.classifier_head = best_head
        self.args = args
        self.alpha_cap = 0.2
        self.alpha_opt = True
        self.alpha_closed_form_approx = True
        self.alpha_learning_rate = 0.1
        self.alpha_clf_coef = 1.0
        self.alpha_learning_iters = 5
        self.alpha_learn_batch_size = 1000000
        self.labeled_in_set = train

    def predict_prob(self, unlabel_idxs):
        self.model.eval()
        probs = torch.zeros([len(unlabel_idxs), self.n_class])
        with torch.no_grad():
            for i, data in enumerate(self.unlabeled_dst):
                inputs = data[0].to(self.device)
                features = self.model.encode_image(inputs)
                out = self.classifier_head(features) * self.logit_scale.exp()
                for j in range(len(inputs)):
                    for a in range(len(unlabel_idxs)):
                        if i * 128 + j == unlabel_idxs[a]:
                            probs[a] = torch.nn.functional.softmax(out, dim=1)[j].cpu().data
        return probs

    def get_embedding(self, dataloader, data):
        self.model.eval()
        if 'rn50' in self.args.model_cfg:
            embDim = 1024  # rn50
        else:
            embDim = 512  # rn101/vitb16/vitb32
        embedding = torch.zeros([len(data), embDim])

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs = data[0].to(self.device)
                features = self.model.encode_image(inputs)
                out = self.classifier_head(features) * self.logit_scale.exp()
                for j in range(len(inputs)):
                    embedding[i * 128 + j] = features[j].data.cpu()

        return embedding

    def query(self, n_query):
        unlabel_idxs = np.array(range(len(self.ul)))
        label_idxs = np.array(range(len(self.train_data)))

        ulb_probs = self.predict_prob(unlabel_idxs)
        org_ulb_embedding = self.get_embedding(self.unlabeled_dst, unlabel_idxs)

        _, probs_sort_idxs = ulb_probs.sort(descending=True)
        pred_1 = probs_sort_idxs[:, 0]

        org_lb_embedding = self.get_embedding(self.labeled_in_set, label_idxs)

        ulb_embedding = org_ulb_embedding
        lb_embedding = org_lb_embedding

        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

        if self.alpha_closed_form_approx:
            var_emb = Variable(ulb_embedding, requires_grad=True).to(self.args.device)
            out = self.classifier_head(var_emb) * self.logit_scale.exp()
            loss = F.cross_entropy(out, pred_1.to(self.device))
            grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
            del loss, var_emb, out
        else:
            grads = None

        alpha_cap = 0.
        while alpha_cap < 1.0:
            alpha_cap += self.alpha_cap

            tmp_pred_change, tmp_min_alphas = \
                self.find_candidate_set(
                    lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
                    Y=self.get_labels(self.labeled_in_set),
                    grads=grads)

            is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

            min_alphas[is_changed] = tmp_min_alphas[is_changed]
            candidate += tmp_pred_change

            print('With alpha_cap set to %f, number of inconsistencies: %d' % (
                alpha_cap, int(tmp_pred_change.sum().item())))

            if candidate.sum() > n_query:
                break

        if candidate.sum() > 0:
            print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

            print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
            print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
            print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

            c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()

            selected_idxs = self.sample(min(n_query, candidate.sum().item()), feats=c_alpha)
            # u_selected_idxs = candidate.nonzero(as_tuple=True)[0][selected_idxs]
            selected_idxs = unlabel_idxs[candidate][selected_idxs]
        else:
            selected_idxs = np.array([], dtype=np.int)

        if len(selected_idxs) < n_query:
            remained = n_query - len(selected_idxs)

            not_selected_idxs = np.array(list(set(unlabel_idxs) - set(selected_idxs)))
            selected_idxs = np.concatenate([selected_idxs, np.random.choice(not_selected_idxs, remained)])
            print('picked %d samples from RandomSampling.' % (remained))

        return list(selected_idxs)

    def get_labels(self, data_idxs):
        with torch.no_grad():
            for i, data in enumerate(data_idxs):
                if i == 0:
                    labels = copy.deepcopy(data[-1])
                else:
                    labels = torch.cat([labels, copy.deepcopy(data[-1])], dim=0)

        return labels.numpy()

    def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):

        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

        if self.alpha_closed_form_approx:
            alpha_cap /= math.sqrt(embedding_size)
            grads = grads.to(self.device)

        for i in range(self.n_class):
            y_idx = (Y == i).reshape(-1, )
            emb = lb_embedding[y_idx]
            if emb.size(0) == 0:
                emb = lb_embedding
            anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

            if self.alpha_closed_form_approx:
                embed_i, ulb_embed = anchor_i.to(self.device), ulb_embedding.to(self.device)
                alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

                embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
                out = self.classifier_head(embedding_mix) * self.logit_scale.exp()
                out = out.detach().cpu()
                alpha = alpha.cpu()

                pc = out.argmax(dim=1) != pred_1
            else:
                alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)
                if self.alpha_opt:
                    alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,
                                                 log_prefix=str(i))
                else:
                    embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
                    out = self.classifier_head(embedding_mix.to(self.device)) * self.logit_scale.exp()
                    out = out.detach().cpu()

                    pc = out.argmax(dim=1) != pred_1

            torch.cuda.empty_cache()

            alpha[~pc] = 1.
            pred_change[pc] = True
            is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
            min_alphas[is_min] = alpha[is_min]

        return pred_change, min_alphas

    def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
        z = (lb_embedding - ulb_embedding)  # * ulb_grads
        alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (
                z + 1e-8)

        return alpha

    def sample(self, n, feats):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(feats)

        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (feats - centers) ** 2
        dis = dis.sum(axis=1)
        return np.array(
            [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
             (cluster_idxs == i).sum() > 0])

    def retrieve_anchor(self, embeddings, count):
        return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)

    def generate_alpha(self, size, embedding_size, alpha_cap):
        alpha = torch.normal(
            mean=alpha_cap / 2.0,
            std=alpha_cap / 2.0,
            size=(size, embedding_size))

        alpha[torch.isnan(alpha)] = 1

        return self.clamp_alpha(alpha, alpha_cap)

    def clamp_alpha(self, alpha, alpha_cap):
        return torch.clamp(alpha, min=1e-8, max=alpha_cap)

    def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''):
        labels = labels.to(self.device)
        min_alpha = torch.ones(alpha.size(), dtype=torch.float)
        pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        self.net.linear.eval()

        for i in range(self.alpha_learning_iters):
            tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
            for b in range(math.ceil(float(alpha.size(0)) / self.alpha_learn_batch_size)):
                self.net.linear.zero_grad()
                start_idx = b * self.alpha_learn_batch_size
                end_idx = min((b + 1) * self.alpha_learn_batch_size, alpha.size(0))

                l = alpha[start_idx:end_idx]
                l = torch.autograd.Variable(l.to(self.args.device), requires_grad=True)
                opt = torch.optim.Adam([l], lr=self.alpha_learning_rate / (
                    1. if i < self.alpha_learning_iters * 2 / 3 else 10.))
                e = org_embed[start_idx:end_idx].to(self.args.device)
                c_e = anchor_embed[start_idx:end_idx].to(self.args.device)
                embedding_mix = (1 - l) * e + l * c_e

                out = self.net.linear(embedding_mix)

                label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

                tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.args.device)
                tmp_pc[start_idx:end_idx] = label_change
                pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

                tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (
                        l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.args.device))
                min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

                clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.args.device))

                l2_nrm = torch.norm(l, dim=1)

                clf_loss *= -1

                loss = self.alpha_clf_coef * clf_loss + self.alpha_l2_coef * l2_nrm
                loss.sum().backward(retain_graph=True)
                opt.step()

                l = self.clamp_alpha(l, alpha_cap)

                alpha[start_idx:end_idx] = l.detach().cpu()

                tot_clf_loss += clf_loss.mean().item() * l.size(0)
                tot_loss += loss.mean().item() * l.size(0)
                tot_nrm += l2_nrm.mean().item() * l.size(0)

                del l, e, c_e, embedding_mix
                torch.cuda.empty_cache()

        count = pred_changed.sum().item()

        return min_alpha.cpu(), pred_changed.cpu()
