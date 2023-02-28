import torch
import pdb


class HelperOptimizer(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        super(HelperOptimizer, self).__init__(params, kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        inner_product = 0
        # print("first_step, gradnorm: ", grad_norm, grad_norm.device)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                # this is only active during analysis
                if self.analysis and "last_grad" in self.state[p]:
                    last_grad = self.state[p]["last_grad"]
                    current_grad = p.grad
                    last_grad = last_grad.reshape(1, -1).squeeze()
                    current_grad = current_grad.reshape(1, -1).squeeze()
                    inner_product += last_grad @ current_grad.T

        if zero_grad: self.zero_grad()
        if self.analysis:
            inner_product /= (grad_norm*self._grad_norm_last()) if grad_norm and self._grad_norm_last() else 1
            return inner_product

    @torch.no_grad()
    def save_parameters(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p][key] = p.data.clone().detach()

    @torch.no_grad()
    def remove_parameters(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                if key in self.state[p]:
                    del self.state[p][key]
    @torch.no_grad()
    def add_noise(self, sigma):
        for group in self.param_groups:
            for p in group["params"]:
                p.data += torch.randn_like(p.data)*abs(p.data)*sigma

    @torch.no_grad()
    def restore_parameters(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p][key].clone().detach()

    @torch.no_grad()
    def blend_parameters(self, key_target=None, key_source=None, alpha_target=0.5, alpha_source=0.5):
        for group in self.param_groups:
            for p in group["params"]:
                if key_target is not None and key_source is not None:
                    self.state[p][key_target].data = self.state[p][key_target].data*alpha_target + self.state[p][key_source]* alpha_source
                elif key_target is None and key_source is not None:
                    p.data = p.data*alpha_target + self.state[p][key_source]*alpha_source
                elif key_target is not None and key_source is None:
                    self.state[p][key_target].data = self.state[p][key_target].data * alpha_target + p.data * alpha_source
                else:
                    raise ValueError("Not allowed!")

    @torch.no_grad()
    def accumulate_parameters(self, key_target=None, key_source=None, alpha=1):
        for group in self.param_groups:
            for p in group["params"]:
                if key_target is not None and key_source is not None:
                    if key_target in self.state[p]:
                        self.state[p][key_target].data += self.state[p][key_source]*alpha
                    else:
                        self.state[p][key_target] = (self.state[p][key_source] * alpha).clone().detach()
                elif key_target is None and key_source is not None:
                    p.data += self.state[p][key_source]*alpha
                elif key_target is not None and key_source is None:
                    if key_target in self.state[p]:
                        self.state[p][key_target].data += p.data*alpha
                    else:
                        self.state[p][key_target] = (p.data * alpha).clone().detach()
                else:
                    raise ValueError("Not allowed!")

    @torch.no_grad()
    def divide_parameters(self, key_target=None, div=None):
        for group in self.param_groups:
            for p in group["params"]:
                if key_target is not None:
                    self.state[p][key_target].data /= div
                elif key_target is None:
                    p.data /= div
                else:
                    raise ValueError("Not allowed!")

    @torch.no_grad()
    def revert_to_saved_parameters(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    @torch.no_grad()
    def save_gradients(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p][key] = p.grad.clone().detach()

    @torch.no_grad()
    def accumulate_gradients(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if key in self.state[p]:
                    self.state[p][key] += p.grad.clone().detach()
                else:
                    self.state[p][key] = p.grad.clone().detach()


    @torch.no_grad()
    def restore_gradients(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = self.state[p][key].clone().detach()

    @torch.no_grad()
    def remove_gradients(self, key):
        for group in self.param_groups:
            for p in group["params"]:
                if key in self.state[p]:
                    del self.state[p][key]




    def grad_align_loss(self, key, key2=None):
        loss = 0
        norm1 = 0
        norm2 = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if key2 is not None:
                    loss += (self.state[p][key] * self.state[p][key2]).sum()
                else:
                    loss += (self.state[p][key]*p.grad).sum()
                norm1 += (self.state[p][key]**2).sum().detach()
                if key2 is not None:
                    norm2 += (self.state[p][key2] ** 2).sum().detach()
                else:
                    norm2 += (p.grad**2).sum().detach()
        loss = loss/(norm1**0.5)/(norm2**0.5)
        return loss

    def grad_align_unnormalized_loss(self, key, key2=None):
        loss = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if key2 is not None:
                    loss += (self.state[p][key] * self.state[p][key2]).sum()
                else:
                    loss += (self.state[p][key] * p.grad).sum()
        return loss

    def grad_norm(self, key=None):
        norm = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if key is None:
                    norm += (p.grad**2).sum()
                else:
                    norm += (self.state[p][key] ** 2).sum()
        return norm**0.5

    @torch.no_grad()
    def clip_grad(self, norm):
        ori_grad_norm = self.grad_norm()
        if ori_grad_norm > norm:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.grad /= (ori_grad_norm /norm)





    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def _grad_norm_last(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        last_grad_norm_list = [
                ((torch.abs(p) if group["adaptive"] else 1.0) * self.state[p]["last_grad"]).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if "last_grad" in self.state[p]
            ]
        if len(last_grad_norm_list) > 0:
            norm = torch.norm(
                torch.stack(last_grad_norm_list),
                p=2
            )
        else:
            norm = None
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class SAM(HelperOptimizer):
    def __init__(self, params, optim, **kwargs):
        super(SAM, self).__init__(params, **kwargs)

        self.base_optimizer = optim(self.param_groups, lr=kwargs['lr'],
                                    momentum=kwargs["momentum"],
                                    weight_decay=kwargs["weight_decay"])

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    def second_step(self, zero_grad=False):
        self.revert_to_saved_parameters()
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()
import random
class PatternSearch(HelperOptimizer):
    def __init__(self, params, **kwargs):
        super(PatternSearch, self).__init__(params, **kwargs)
        self.radius = 1
        self.basis_list = []
        self.best_loss = None
        for group in self.param_groups:
            for p in group["params"]:
                pflatten = p.view(-1)
                for i in range(len(pflatten)):
                    for op in ["+", "-"]:
                        self.basis_list.append((pflatten, i, op))
        random.shuffle(self.basis_list)

    def step(self, closure):
        if self.best_loss == 0:
            return
        if self.best_loss is None:
            self.best_loss = closure().detach().clone()
        # loop over different possible basis:
        random.shuffle(self.basis_list)
        for p, idx, op in self.basis_list:
            if op == "+":
                p.data[idx] += self.radius
            elif op == "-":
                p.data[idx] -= self.radius
            new_loss = closure()
            if new_loss < self.best_loss:
                self.best_loss = new_loss
                self.radius *= 1
                return
            if op == "+":
                p.data[idx] -= self.radius
            elif op == "-":
                p.data[idx] += self.radius
        # if no good perturbations are found shrink the radius
        self.radius /= 2


class NelderMead(HelperOptimizer):
    def __init__(self, params, **kwargs):
        super(NelderMead, self).__init__(params, **kwargs)
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]
        self.rho = kwargs["rho"]
        self.sigma = kwargs["sigma"]
        self.simplex = None
    def initialize_simplex(self, closure, init_step = 4):
        init_loss = closure().detach().cpu().item()
        self.save_parameters("model_0")
        self.simplex = [("model_0", init_loss)]
        # loop through all parameters and adding in solutions
        model_idx = 1
        for group in self.param_groups:
            for p in group["params"]:
                pflatten = p.view(-1)
                for i in range(len(pflatten)):
                    pflatten.data[i] += init_step
                    loss = closure().detach().cpu().item()
                    self.save_parameters(f"model_{model_idx}")
                    self.simplex.append((f"model_{model_idx}", loss))
                    pflatten.data[i] -= init_step
                    model_idx += init_step
        self.simplex.sort(key=lambda x: x[1])


    def replace_worst_simplex(self, new_loss):
        model_name, _ = self.simplex[-1]
        self.save_parameters(model_name)
        self.simplex[-1] = (model_name, new_loss)
        self.simplex.sort(key=lambda x: x[1])
        self.restore_parameters(self.simplex[0][0])

    def compute(self, key_target, a, b, c, coefficient):
        #calculate a+coefficient(b-c), and set it for target
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p][key_target] = self.state[p][a] + coefficient*(self.state[p][b]-self.state[p][c])
    def step(self, closure):
        if self.simplex is None:
            self.initialize_simplex(closure)
            return
        #calculate centroid
        self.remove_parameters("centroid")
        for model_name, acc in self.simplex[:-1]:
            self.accumulate_parameters(key_target="centroid", key_source=model_name)
        self.divide_parameters(key_target="centroid", div=len(self.simplex)-1)

        # calculate reflection
        self.compute(key_target="r", a="centroid", b="centroid", c=self.simplex[-1][0], coefficient=self.gamma)
        self.restore_parameters(key="r")
        loss_r = closure().detach().cpu().item()

        if loss_r < self.simplex[-2][1] and loss_r >= self.simplex[0][1]:
            self.replace_worst_simplex(loss_r)
            return
        elif loss_r < self.simplex[0][1]:
            # calculate expansion
            self.compute(key_target="e", a="centroid", b="r", c="centroid", coefficient=self.gamma)
            self.restore_parameters("e")
            loss_e = closure().detach().cpu().item()
            if loss_e < loss_r:
                self.replace_worst_simplex(loss_e)
                return
            else:
                self.restore_parameters("r")
                self.replace_worst_simplex(loss_r)
                return
        else:
            # calculate contraction
            if loss_r < self.simplex[-1][1]:
                self.compute(key_target="c", a="centroid", b="r", c="centroid", coefficient=self.rho)
                self.restore_parameters("c")
                loss_c = closure().detach().cpu().item()
                if loss_c < loss_r:
                    self.replace_worst_simplex(loss_c)
                    return
                else:
                    pass # continue to shrinking step
            else:
                worst_model_name, loss_worst = self.simplex[-1]
                self.compute(key_target="c", a="centroid", b=worst_model_name, c="centroid", coefficient=self.rho)
                self.restore_parameters("c")
                loss_c = closure().detach().cpu().item()
                if loss_c < loss_worst:
                    self.replace_worst_simplex(loss_c)
                    return
                else:
                    pass # continue to shrinking step

        #shrinking step
        model_name_best, best_loss = self.simplex[0]
        new_simplex = [(model_name_best, best_loss)]
        for model_name, acc in self.simplex[1:]:
            self.compute(key_target=model_name, a=model_name_best, b=model_name, c=model_name_best, coefficient=self.sigma)
            self.restore_parameters(model_name)
            loss_current = closure().detach().cpu().item()
            new_simplex.append((model_name, loss_current))
        self.simplex = new_simplex
        self.simplex.sort(key=lambda x: x[1])
        self.restore_parameters(self.simplex[0][0])








