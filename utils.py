import torch
import matplotlib.pyplot as plt
from torch import nn
from matplotlib.colors import ListedColormap
import numpy as np
import torch.nn.functional as F
from itertools import chain
import random

def calculate_loss_acc(data, labels, model, loss_func, batch_size=None):
    if batch_size is None:
        pred = model(data)  # pred.shape = (# of examples, # model counts , output_dim)
    else:
        pred = []
        for i in range(0, len(data), batch_size):
            pred_cur = model(data[i:min(i+batch_size, len(data))])
            pred.append(pred_cur)
        pred = torch.cat(pred, dim=0)
    n, m, o = pred.shape
    loss = loss_func(pred.view(n * m, o), labels.repeat_interleave(m)).view(n, m).mean(dim=0)
    acc = (pred.view(n * m, o).argmax(dim=1) == labels.repeat_interleave(m)).view(n, m).float().mean(dim=0)
    return loss, acc


def make_permutation_invariant(m1, m2):
    # shape (1, model_count, out_d, in_d)
    sort_idx = m1[:, :, 0:1, :].sort(dim=3).indices
    new_m1 = torch.gather(m1, dim=3, index=sort_idx.repeat(1, 1, m1.shape[2], 1))
    new_m2 = torch.gather(m2, dim=2, index=sort_idx[:, :, 0, :, None])
    return new_m1, new_m2

def change_minimas_to_matrices(minimas, hidden_units):
    matrix1 = minimas[:, 0:hidden_units*2].reshape(1, -1, 2, hidden_units)
    matrix2 = minimas[:, hidden_units*2:hidden_units*3].reshape(1, -1, hidden_units, 1)
    bias2 = minimas[:, hidden_units*3:].reshape(1, -1, 1, 1)
    return matrix1, matrix2, bias2

def change_matrices_to_minimas(m1, m2, b2, hidden_units):
    minimas = torch.cat([
        m1.reshape(-1, 2*hidden_units),
        m2.reshape(-1, hidden_units*1),
        b2.reshape(-1, 1)], dim=1)
    return minimas

def visualize_decision_boundary(models_list, data=None, xlims=(-2,2), ylims=(-2, 2), filename='test.png'):
    model_count = models_list[0].model_count
    fig, axes = plt.subplots(nrows=model_count//3,
                             ncols=len(models_list)*3,
                             figsize=(len(models_list) * 3*3, (model_count) * 3//3))
    axes = np.reshape(axes, (model_count//3, len(models_list), 3))
    axes = np.transpose(axes, (0, 2, 1))
    axes = np.reshape(axes, (model_count, len(models_list)))
    axes = axes.T

    # X.reshape(3, 5, 3).permute(0, 2, 1).reshape(9, 5)
    # axes = axes.T
    for row_i, models in enumerate(models_list):
        models = models.cuda()
        xx, yy = np.meshgrid(np.arange(xlims[0], xlims[1], 0.01), np.arange(ylims[0], ylims[1], 0.01))
        grid_data = torch.cat([torch.tensor(xx.ravel())[:, None], torch.tensor(yy.ravel())[:, None]], dim=1).float().cuda()
        batch_size = 30
        predictions = []
        with torch.no_grad():
            for i in range(0, len(grid_data), batch_size):
                predictions.append(models(grid_data[i:min(i+batch_size, len(grid_data))]))
            predictions = torch.cat(predictions, dim=0)
        predictions = torch.softmax(predictions, dim=2).round()
        cm = plt.cm.hot
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])

        for model_i in range(model_count):
            grid_score = predictions[:, model_i, 1].cpu().detach().reshape(xx.shape)
            axes[row_i, model_i].contourf(xx, yy, grid_score, cmap=cm, alpha=0.8, vmin=0.4, vmax=0.6, rasterized=True)
            axes[row_i, model_i].set_axis_off()
            if data is not None:
                x, y = data

                axes[row_i, model_i].scatter(
                    x[y==0, 0].cpu(), x[y==0, 1].cpu(), c=np.zeros((len(x[y==0]), 1)), cmap=cm_bright,
                    edgecolors="k", rasterized=True, s=5
                )
                axes[row_i, model_i].scatter(
                    x[y==1, 0].cpu(), x[y==1, 1].cpu(), c=np.ones((len(x[y==1]), 1)), cmap=cm_bright,
                    edgecolors="k", marker='x', rasterized=True, s=5
                )
    # place some rectangular patches
            pad_x=0.004
            pad_y=0.02
            x0 = axes[row_i, 6].get_position().x0
            y0 = axes[row_i, 6].get_position().y0
            w = axes[row_i, 2].get_position().x1-x0
            h = axes[row_i, 2].get_position().y1-y0
            rect = plt.Rectangle(
                # (lower-left corner), width, height
                (x0-pad_x, y0-pad_y), w+pad_x*2, h+pad_y*2, fill=False, color="k", lw=6, 
                zorder=1000, transform=fig.transFigure, figure=fig
            )
            fig.patches.extend([rect])
    fig.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()

def calculate_sharpness_random_gaussian(m1, m2, b2, data, sigma=1, sample_count=100):
    # calculate stability with respect to gaussian noise

    # m1.shape (1, model_count, 2, hidden_units)
    # m2.shape (1, model_count, 2, hidden_units)
    model_count = m1.shape[1]
    x, y = data

    # add noise to the model

    train_accs = []
    for i in range(sample_count):
        m1_noise = torch.randn_like(m1) * sigma
        m2_noise = torch.randn_like(m2) * sigma
        b2_noise = torch.randn_like(b2) * sigma

        m1 += m1_noise
        m2 += m2_noise
        b2 += b2_noise
        # calculate prediction accuracy
        # repeat
        predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
        predictions = predictions > 0.5
        # predictions.shape (data count, model_count, 1, 1)
        train_acc = (y == predictions).float().mean(dim=0)[:, 0]
        train_accs.append(train_acc)
        # train_accs.shape (model_count, 1)
        m1 -= m1_noise
        m2 -= m2_noise
        b2 -= b2_noise
    train_accs = torch.cat(train_accs, dim=1)
    train_accs_mean = train_accs.mean(dim=1)
    return train_accs_mean

def calculate_norm(m1, m2, b2):
    m1_normsq = (m1 ** 2).sum(dim=(2, 3), keepdim=True)
    m2_normsq = (m2 ** 2).sum(dim=(2, 3), keepdim=True)
    b2_normsq = (b2 ** 2).sum(dim=(2, 3), keepdim=True)
    total_normsq = m1_normsq + m2_normsq + b2_normsq
    total_norm = total_normsq ** 0.5
    return total_norm

def calculate_sharpness_random_dir(m1, m2, b2, data, sample_count=10):
    # calculate stability with respect to gaussian noise

    # m1.shape (1, model_count, 2, hidden_units)
    # m2.shape (1, model_count, 2, hidden_units)
    model_count = m1.shape[1]
    x, y = data

    # add noise to the model

    train_accs = []
    biggest_rs = []
    for i in range(sample_count):
        biggest_r = torch.tensor(0.0, device=m1.device, dtype=torch.float)

        m1_noise_unit = torch.randn_like(m1)
        m2_noise_unit = torch.randn_like(m2)
        b2_noise_unit = torch.randn_like(b2)
        total_norm = calculate_norm(m1_noise_unit, m2_noise_unit, b2_noise_unit)
        m1_noise_unit = m1_noise_unit / total_norm
        m2_noise_unit = m2_noise_unit / total_norm
        b2_noise_unit = b2_noise_unit / total_norm

        for r in np.linspace(0, 3, 100):
            m1_noise = m1_noise_unit * r
            m2_noise = m2_noise_unit * r
            b2_noise = b2_noise_unit * r

            m1 += m1_noise
            m2 += m2_noise
            b2 += b2_noise
            # calculate prediction accuracy
            # repeat
            predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
            predictions = predictions > 0.5
            # predictions.shape (data count, model_count, 1, 1)
            train_acc = (y == predictions).float().mean(dim=0)[:, 0]
            # train_accs.shape (model_count, 1)
            biggest_r = torch.where(train_acc == 1,
                                    torch.tensor(r, device=train_acc.device, dtype=torch.float),
                                    biggest_r)


            m1 -= m1_noise
            m2 -= m2_noise
            b2 -= b2_noise
            if (train_acc==1).sum() == 0:
                break
        nan_tensor = torch.tensor(float('NaN'), device=biggest_r.device, dtype=torch.float)
        biggest_r = torch.where(biggest_r==3, nan_tensor, biggest_r)
        biggest_rs.append(biggest_r)
    biggest_rs = torch.cat(biggest_rs, dim=1)
    biggest_rs_mean = biggest_rs.nanmean(dim=1)
    nan_count = (biggest_rs == float('nan')).sum(dim=1)
    return biggest_rs_mean, nan_count

def calculate_sharpness_sam(m1, m2, b2, data, rho=1):
    # calculate stability with respect to gaussian noise

    # m1.shape (1, model_count, 2, hidden_units)
    # m2.shape (1, model_count, 2, hidden_units)
    model_count = m1.shape[1]
    x, y = data

    # add noise to the model

    train_accs = []
    m1.requires_grad = True
    m2.requires_grad = True
    b2.requires_grad = True
    # backward pass
    predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
    losses_ori = ((predictions - y)**2).mean(dim=0)
    loss_ori = losses_ori.sum()
    m1_grad, m2_grad, b2_grad = torch.autograd.grad(loss_ori, [m1, m2, b2])
    grad_norm = calculate_norm(m1_grad, m2_grad, b2_grad)
    m1_grad = m1_grad/grad_norm*rho
    m2_grad = m2_grad/grad_norm*rho
    b2_grad = b2_grad/grad_norm*rho

    with torch.no_grad():
        m1 += m1_grad
        m2 += m2_grad
        b2 += b2_grad

        # predictions.shape (data count, model_count, 1, 1)
        predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
        losses_attacked = ((predictions - y)**2).mean(dim=0)[:, 0, 0]
        train_acc_attacked = (y == (predictions>0.5)).float().mean(dim=0)[:, 0, 0]
        # train_accs.shape (model_count, 1)
        m1 -= m1_grad
        m2 -= m2_grad
        b2 -= b2_grad
    return train_acc_attacked, losses_attacked

def test_acc_by_bin(test_acc, bin_metric, bin_count=10):
    bin_metric = bin_metric.cpu()
    intervals = np.linspace(bin_metric.min(), bin_metric.max(), bin_count+1)
    for l, u in zip(intervals[:-1], intervals[1:]):
        idx = ((bin_metric >= l) & (bin_metric <= u))
        print(f"interval: {l.item(): 0.3f}, {u.item(): 0.3f}, count:{idx.sum().item()} "
              f"test accs: {test_acc[idx].mean().cpu().item(): 0.3f}")

class MLPModels(nn.Module):
    def __init__(self, input_dim, output_dim, layers, hidden_units, model_count, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.hidden_units = hidden_units
        self.device = device
        self.model_count = model_count
        self.weights = []
        for layer_i in range(layers+1):
            if layer_i == 0:
                self.weights.append(nn.Parameter(torch.rand((1, model_count, input_dim, hidden_units), device=device) * 2 - 1))
            elif layer_i == layers:
                self.weights.append(nn.Parameter(torch.rand((1, model_count, hidden_units, output_dim), device=device) * 2 - 1))
            else:
                self.weights.append(nn.Parameter(torch.rand((1, model_count, hidden_units, hidden_units), device=device) * 2 - 1))
        self.bias = nn.Parameter(torch.randn((1, model_count, output_dim), device=device) * 2 - 1)
        self.weights = torch.nn.ParameterList(self.weights)

    def reinitialize(self):
        for matrix in self.weights:
            torch.nn.init.uniform_(matrix.data, a=-1, b=1)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    @torch.no_grad()
    def reset_parameters(self):
        import math
        for weight in self.weights:
            stdv = 1. / math.sqrt(weight.shape[3])
            weight.data.uniform_(-stdv, stdv)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim]
        # outputs [# of examples, model_count, logit_count]
        x = x[:, None, None]
        for matrix in self.weights[:-1]:
            x = x @ matrix
            x = torch.clamp(x, 0)
        x = x @ self.weights[-1]
        x = x.squeeze(2)
        x = x + self.bias
        return x

    def get_feature(self, x, cat_one=False):
        x = x[:, None, None]
        for matrix in self.weights[:-1]:
            x = x @ matrix
            x = torch.clamp(x, 0)
        x = x.squeeze(2)
        if cat_one:
            x = torch.cat((x, torch.ones(*x.shape[:2], 1, device=x.device)), dim=2)
        return x

    @torch.no_grad()
    def get_grad_norms(self):
        grad_square = 0
        for weight in self.weights:
            grad_square += (weight.grad**2).sum(dim=(0,2,3))
        grad_square += (self.bias.grad ** 2).sum(dim=(0, 2))
        grad_norm = grad_square ** 0.5
        return grad_norm

    def zero_grad(self):
        for para in self.parameters():
            para.grad = None

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        return {name: para[:, idx].cpu() for name, para in self.state_dict().items()}

    def get_model_subsets(self, idx):
        model_count = len(idx)
        new_model = MLPModels(
            input_dim=self.input_dim, output_dim=self.output_dim,
            layers=self.layers, hidden_units=self.hidden_units,
            model_count=model_count, device=self.device)
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model

    @torch.no_grad()
    def normalize(self):
        cum_norm = 1
        for weight in self.weights:
            cur_norm = weight.norm(dim=(2,3), keepdim=True)
            weight.data /= cur_norm
            cum_norm *= cur_norm
        cum_norm = cum_norm.squeeze(3)
        self.bias.data /= cum_norm

    def forward_normalize(self, x):
        cum_norm = 1
        for weight in self.weights:
            cur_norm = weight.norm(dim=(2,3), keepdim=True)
            cum_norm *= cur_norm
        cum_norm = cum_norm.squeeze(3)
        return self.forward(x)/cum_norm

    @torch.no_grad()
    def make_permutation_invariant(self):
        weights = self.weights
        for i in range(len(weights)-1):
            sort_idx = weights[i][:, :, 0:1, :].sort(dim=3).indices
            weights[i].data.copy_(
                torch.gather(weights[i], dim=3, index=sort_idx.repeat(1, 1, weights[i].shape[2], 1))
            )
            weights[i+1].data.copy_(
                torch.gather(weights[i + 1], dim=2, index=sort_idx.permute(0, 1, 3, 2).repeat(1, 1, 1, weights[i+1].shape[3]))
            )

    @torch.no_grad()
    def shorten(self, count):
        idx = torch.arange(count)
        return self.get_model_subsets(idx)

    @torch.no_grad()
    def get_vectorized_weights(self):
        # return (# of models, # of parameters) as a tensor
        vectorized_weights = []
        for weight in chain(self.weights, [self.bias]):
            vectorized_weights.append(weight.data.reshape(self.model_count, -1).detach().cpu())
        vectorized_weights = torch.cat(vectorized_weights, dim=1)
        return vectorized_weights

class LeNetModels(nn.Module):
    def __init__(self, output_dim, width_factor, model_count, dataset, feature_dim=None):
        super(LeNetModels, self).__init__()
        self.model_count = model_count
        self.output_dim = output_dim
        self.width_factor = width_factor
        self.dataset = dataset
        if feature_dim is None:
            self.feature_dim = int(84 * width_factor)
        else:
            self.feature_dim = feature_dim
        if dataset == "cifar10":
            self.conv1 = nn.Conv2d(3*model_count,
                                    int(6*width_factor)*model_count,
                                    5, groups=model_count
                                   )

        elif dataset == "mnist":
            self.conv1 = nn.Conv2d(
                1*model_count,
                int(6*width_factor)*model_count,
                5, groups=model_count
            )
        self.conv2 = nn.Conv2d(int(6*width_factor)*model_count,
                               int(16*width_factor)*model_count,
                               5, groups=model_count)
        if dataset == "cifar10":
            self.fc1 = nn.Conv2d(int(16*width_factor)*5*5*model_count,
                                 int(120*width_factor)*model_count,
                                 1,
                                 groups=model_count)
        elif dataset == "mnist":
            self.fc1 = nn.Conv2d(int(16*width_factor)*4*4*model_count,
                                 int(120*width_factor)*model_count,
                                 1,
                                 groups=model_count)
        self.fc2 = nn.Conv2d(int(120*width_factor)*model_count,
                                 int(self.feature_dim*model_count),
                                 1,
                                 groups=model_count)
        self.fc3 = nn.Conv2d(int(self.feature_dim*model_count),
                                 output_dim*model_count,
                                 1,
                                 groups=model_count)
        self.basis_list = None
        self.curr_idx = 0
        self.radius= 1

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim, H, W]
        # outputs [# of examples, model_count, logit_count]
        x = x.repeat(1, self.model_count, 1, 1)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1, 1, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.view(out.size(0), self.model_count, self.output_dim)
        return out

    @torch.no_grad()
    def pattern_search(self, x, y, loss_func):
        import random
        if self.basis_list is None:
            self.basis_list = []
            for para in self.parameters():
                para_flatten = para.data.view(self.model_count, -1)
                for p in range(para_flatten.shape[1]):
                    self.basis_list.append((para_flatten, p, "+"))
                    self.basis_list.append((para_flatten, p, "-"))
        random.shuffle(self.basis_list)
        self.curr_idx = 0

        while True:
            # replicate the first model and duplicate the weights across models
            for para in self.parameters():
                original_shape = para.shape
                para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                para_reshaped[1:] = para_reshaped[0:1]


            # modify each model at one index location
            for i in range(1,self.model_count):
                if self.curr_idx >= len(self.basis_list):
                    import pdb; pdb.set_trace()
                para, p_i, op = self.basis_list[self.curr_idx]
                if op == "+":
                    para[i, p_i] += self.radius
                else:
                    para[i, p_i] -= self.radius
                self.curr_idx += 1
                if self.curr_idx >= len(self.basis_list):
                    print("went over everything")
                    random.shuffle(self.basis_list)
                    self.radius /= 2
                    self.curr_idx = 0
                    break

            # forward and select the model with the best losses, and it into index 0
            pred = self.forward_normalize(x)
            n, m, o = pred.shape
            loss = loss_func(pred.view(n * m, o), y.repeat_interleave(m)).view(n, m).mean(dim=0)

            best_idx = loss.min(dim=0).indices

            for para in self.parameters():
                original_shape = para.shape
                para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                para_reshaped[:] = para_reshaped[best_idx:best_idx+1]
            if best_idx != 0:
                break

    @torch.no_grad()
    def greedy_random(self, x, y, loss_func):
        for _ in range(30):
            iter_max = 100
            for i in range(iter_max):
                # add noise to the all models beside the zero indexed model
                for para in self.parameters():
                    original_shape = para.shape
                    para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                    para_reshaped[1:] = para_reshaped[0:1]
                    para_reshaped[1:] += torch.randn_like(para_reshaped[1:])*self.radius

                # forward and select the model with the best losses, and it into index 0
                pred = self.forward_normalize(x)
                n, m, o = pred.shape
                loss = loss_func(pred.view(n * m, o), y.repeat_interleave(m)).view(n, m).mean(dim=0)

                best_idx = loss.min(dim=0).indices

                for para in self.parameters():
                    original_shape = para.shape
                    para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                    para_reshaped[:] = para_reshaped[best_idx:best_idx + 1]
                if best_idx != 0:
                    return
            print(f"radius decreased to {self.radius/2}")
            self.radius /= 2




    def get_feature(self, x, cat_one=False):
        x = x.repeat(1, self.model_count, 1, 1)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1, 1, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), self.model_count, self.feature_dim)

        if cat_one:
            out = torch.cat((out, torch.ones(*out.shape[:2], 1, device=x.device)), dim=2)
        return out

    @torch.no_grad()
    def get_model_subsets(self, idx):
        model_count = len(idx)
        new_model = LeNetModels(output_dim=self.output_dim,
                                width_factor=self.width_factor,
                                model_count=model_count,
                                feature_dim=self.feature_dim,
                                dataset=self.dataset)
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model

    @torch.no_grad()
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @torch.no_grad()
    def reinitialize(self, mult=1):
        for para in self.parameters():
            torch.nn.init.uniform_(para.data, a=-mult, b=mult)


    @torch.no_grad()
    def reinitialize_sphere(self, mult=1):
        overall_norm_square = 0
        for para in self.parameters():
            torch.nn.init.normal_(para.data)
            original_shape = para.shape
            para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
            sum_dim = tuple((d for d in range(1, len(para_reshaped.shape))))
            overall_norm_square += (para_reshaped ** 2).sum(dim=sum_dim)
        overall_norm = overall_norm_square ** 0.5
        for para in self.parameters():
            original_shape = para.shape
            para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
            new_norm_shape = (-1, ) + tuple((1 for i in range(len(para_reshaped.shape)-1)))
            para_reshaped /= (overall_norm.view(new_norm_shape)/mult)
        
    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        weight_dict = {}
        for name, para in self.state_dict().items():
            original_shape = para.shape
            para_reshaped = para.reshape(self.model_count, -1, *original_shape[2:])
            para_selected = para_reshaped[idx]
            para_selected = para_selected.reshape(-1, *original_shape[1:])
            weight_dict[name] = para_selected.clone().detach().cpu()
        return weight_dict

    @torch.no_grad()
    def shorten(self, count):
        idx = torch.arange(count)
        return self.get_model_subsets(idx)

    @torch.no_grad()
    def normalize(self):
        cum_norm = 1
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            cur_weight = layer.weight
            original_shape = cur_weight.shape
            cur_weight = cur_weight.view(self.model_count, -1, *original_shape[2:])
            cur_norm = cur_weight.norm(p=2, dim=(1, 2, 3), keepdim=True) / 3
            cur_weight /= cur_norm
            cum_norm *= cur_norm.view(self.model_count, -1)
            biasview = layer.bias.data.view(self.model_count, -1)
            biasview /= cum_norm

    @torch.no_grad()
    def forward_normalize(self, x):
        x = self.forward(x)
        cum_norm = 1
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            cur_weight = layer.weight
            original_shape = cur_weight.shape
            cur_weight = cur_weight.view(self.model_count, -1, *original_shape[2:])
            cur_norm = cur_weight.norm(p=2, dim=(1, 2, 3), keepdim=True) /3
            cum_norm *= cur_norm.view(self.model_count, -1)
        x /= cum_norm
        return x

class LinearModels(nn.Module):
    def __init__(self, input_dim, output_dim, model_count, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.model_count = model_count
        self.weight = nn.Parameter(torch.rand((1, model_count, input_dim, output_dim), device=device) * 2 - 1)
        self.bias = nn.Parameter(torch.randn((1, model_count, output_dim), device=device) * 2 - 1)

    def reinitialize(self):
        torch.nn.init.uniform_(self.weight.data, a=-1, b=1)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    @torch.no_grad()
    def reset_parameters(self):
        import math
        stdv = 1. / math.sqrt(self.weight.shape[3])
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim]
        # outputs [# of examples, model_count, logit_count]
        x = x.view(x.shape[0], -1)
        x = x[:, None, None]
        x = x @ self.weight
        x = x.squeeze(2)
        x = x + self.bias
        return x

    def forward_normalize(self, x):
        cur_norm = self.weight.norm(dim=(2,3), keepdim=True).squeeze(3)
        return self.forward(x)/cur_norm

    @torch.no_grad()
    def normalize(self):
        weight_norm = self.weight.norm(dim=(2,3), keepdim=True)
        self.weight.data /= weight_norm
        self.bias.data /= weight_norm.squeeze(3)

    @torch.no_grad()
    def get_grad_norms(self):
        grad_square = 0
        grad_square += (self.weight.grad**2).sum(dim=(0,2,3))
        grad_square += (self.bias.grad ** 2).sum(dim=(0, 2))
        grad_norm = grad_square ** 0.5
        return grad_norm

    def zero_grad(self):
        for para in self.parameters():
            para.grad = None

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        return {name: para[:, idx].cpu() for name, para in self.state_dict().items()}

    def get_model_subsets(self, idx):
        model_count = len(idx)
        new_model = LinearModels(
            input_dim=self.input_dim, output_dim=self.output_dim,
            model_count=model_count, device=self.device)
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model
if __name__ == "__main__":
    model = MLPModels(input_dim=2, output_dim=2,
              layers=1, hidden_units=3,
              model_count=3000, device=torch.device('cuda:0'))

    x = torch.randn((10, 2))
    print("This should be (10, 3000, 2)", model(x.cuda()).shape)
    model = LeNetModels(output_dim=2, width_factor=1, model_count=10, dataset='mnist').cuda()
    x_ori = torch.randn((10, 1, 28, 28)).cuda()
    out = model(x_ori)
    print(f"This should be (10, 20, 2): {out.shape}")
    for i in range(10):
        print("===="*10)
        weight = model.conv1.weight[6*i:6*(i+1), :, :, :]
        bias = model.conv1.bias[6*i:6*(i+1)]
        x = F.conv2d(x_ori, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        weight = model.conv2.weight[16*i:16*(i+1)]
        bias = model.conv2.bias[16*i:16*(i+1)]
        x = F.conv2d(x, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.reshape(10, 16 * 4 * 4, 1, 1)
        x = F.conv2d(x, model.fc1.weight[120 * i:120 * (i + 1)], model.fc1.bias[120 * i:120 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc2.weight[84 * i:84 * (i + 1)], model.fc2.bias[84 * i:84 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc3.weight[2*i:2*(i+1)], model.fc3.bias[2*i:2*(i+1)])

        print(f"this should be close to zero: {(x.flatten() - out[:, i:i+1].flatten()).abs().max().cpu().item(): 0.3f}" )

    model = LeNetModels(output_dim=2, width_factor=1, model_count=10, dataset='cifar10').cuda()
    x_ori = torch.randn((10, 3, 32, 32)).cuda()
    out = model(x_ori)
    print(f"This should be (10, 20, 2): {out.shape}")
    for i in range(10):
        print("===="*10)
        weight = model.conv1.weight[6*i:6*(i+1), :, :, :]
        bias = model.conv1.bias[6*i:6*(i+1)]
        x = F.conv2d(x_ori, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        weight = model.conv2.weight[16*i:16*(i+1)]
        bias = model.conv2.bias[16*i:16*(i+1)]
        x = F.conv2d(x, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.reshape(10, 16 * 5 * 5, 1, 1)
        x = F.conv2d(x, model.fc1.weight[120 * i:120 * (i + 1)], model.fc1.bias[120 * i:120 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc2.weight[84 * i:84 * (i + 1)], model.fc2.bias[84 * i:84 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc3.weight[2*i:2*(i+1)], model.fc3.bias[2*i:2*(i+1)])

        print(f"this should be close to zero: {(x.flatten() - out[:, i:i+1].flatten()).abs().max().cpu().item(): 0.3f}" )



























