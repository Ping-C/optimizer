import torch
import os
from tqdm import tqdm
import argparse
from utils import *
from datasets import MNIST, Kink, Cifar10, Slab, SlabLinear, SlabNonlinear4
from fastargs import Section, Param, get_current_config
from fastargs.validation import OneOf
from fastargs.decorators import param, section
from optimizer import NelderMead, PatternSearch
import time
import json
from sql import *

Section("dataset", "Dataset parameters").params(
    name=Param(str, OneOf(("mnist", "kink", "cifar10", "slab", "slab_nonlinear_3", "slab_nonlinear_4", "slab_linear")), default="kink"),
)
Section("dataset.kink", "Dataset parameters for kink").enable_if(
    lambda cfg: cfg['dataset.name'] == 'kink'
).params(
    margin=Param(float),
    noise=Param(float)
)
Section("dataset.mnistcifar", "Dataset parameters for mnist/cifar").params(
    num_classes=Param(int)
)
Section("model", "Model architecture parameters").params(
    arch=Param(str, OneOf(("mlp", "lenet")), default="mlp"),
    model_count_times_batch_size=Param(int, default=20000*16),
    init=Param(str, OneOf(("uniform", "regular", "uniform2", "uniform5", "sphere100", "sphere200")), default="uniform")
)
Section("model.lenet", "Model architecture parameters").params(
    width=Param(float),
    feature_dim=Param(float)
)
Section("model.mlp", "Model architecture parameters").enable_if(lambda cfg: cfg['model.arch'] == 'mlp').params(
    hidden_units=Param(int),
    layers=Param(int)
)
Section("optimizer").params(
    name=Param(str, OneOf(["SGD", "SGDPoison", "Adam", "RMSProp", "guess", "GD"]), default='guess'),
    es_u=Param(float, default=float('inf')),
    es_l=Param(float, default=-float('inf')),
    grad_norm_thres=Param(float, desc='only accept models with gradient norm smaller than specified'),
    lr=Param(float, desc='learning rate'),
    momentum=Param(float, desc='momentum', default=0),
    epochs=Param(int, desc='number of epochs to optimize  for'),
    es_acc=Param(float, desc='stop the training when average training acc reaches this level'),
    batch_size=Param(int, desc='number of epochs ot optimize for', default=3),
    scheduler=Param(int, desc='whether to use a scheduler', default=False),
    poison_factor=Param(float, desc='level of poisoning applied'),
    print_intermediate_test_acc=Param(int, default=0, desc='whether to print intermediate test acc')
)
# TODO: write logic for excluded_cells
Section("distributed").params(
    loss_thres=Param(str, default="0.3,0.4,0.5"),
    num_samples=Param(str, default="2,4,8"),
    excluded_cells=Param(str, default="", desc='ex: 32_(0.3, 0.35)/16_(0.3, 0.35)'),
    target_model_count_subrun=Param(int, default=1),
    training_seed=Param(int, default=None, desc='If there is no training seed, then the training seed increment with every new runs'),
    data_seed=Param(int, default=None, desc='If there is no data seed, then the training seed increment with every new runs, otherwise, it is fix')
)

Section("output", "arguments associated with output").params(
    target_model_count=Param(int, default=1),
    folder=Param(str, default='test_distributed')
)



@section('dataset')
@param('name')
@param('mnistcifar.num_classes')
@param('kink.noise')
@param('kink.margin')
def get_dataset(name, num_samples, seed, num_classes=None, noise=None, margin=0.25):
    if name =="mnist":
        name = MNIST(batch_size=num_samples, threads=1, aug='none', train_count=num_samples, num_classes=num_classes, seed=seed)
        train_data, train_labels = next(iter(name.train))
        test_data, test_labels = next(iter(name.test))
        train_data, train_labels, test_data, test_labels = train_data.cuda(), train_labels.cuda(), test_data.cuda(), test_labels.cuda()
        test_all_data, test_all_labels = name.test_all_data, name.test_all_labels
    elif name == "cifar10":
        name = Cifar10(batch_size=num_samples, threads=1, aug='none', train_count=num_samples, num_classes=num_classes, seed=seed)
        train_data, train_labels = next(iter(name.train))
        test_data, test_labels = next(iter(name.test))
        train_data, train_labels, test_data, test_labels = train_data.cuda(), train_labels.cuda(), test_data.cuda(), test_labels.cuda()
        test_all_data, test_all_labels = name.test_all_data, name.test_all_labels
    elif name == "kink":
        train_data = torch.tensor(
            Kink(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().cuda()
        train_labels = torch.tensor(
            Kink(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().cuda()
        test_data = torch.tensor(
            Kink(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().cuda()
        test_labels = torch.tensor(
            Kink(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().cuda()
        test_all_data, test_all_labels = test_data, test_labels
    elif name == "slab":
        train_data = torch.tensor(
            Slab(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().cuda()
        train_labels = torch.tensor(
            Slab(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().cuda()
        test_data = torch.tensor(
            Slab(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().cuda()
        test_labels = torch.tensor(
            Slab(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().cuda()
        test_all_data, test_all_labels = test_data, test_labels
    elif name == "slab_nonlinear_4":
        dataset = SlabNonlinear4(samples=num_samples)
        train_data = torch.tensor(dataset.data).float().cuda()
        train_labels = torch.tensor(dataset.labels).long().cuda()
        test_data = train_data
        test_labels = train_labels
        test_all_data, test_all_labels = train_data, train_labels
    elif name == "slab_linear":
        dataset = SlabLinear(samples=num_samples)
        train_data = torch.tensor(dataset.data).float().cuda()
        train_labels = torch.tensor(dataset.labels).long().cuda()
        test_data = train_data
        test_labels = train_labels
        test_all_data, test_all_labels = train_data, train_labels
    return train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels


@section('model')
@param('arch')
def get_model(arch, model_count, device):
    if arch == "mlp":
        model = MLPModels(input_dim=2, output_dim=2,
                          layers=config['model.mlp.layers'], hidden_units=config['model.mlp.hidden_units'],
                          model_count=model_count, device=device)
    elif arch == "lenet":
        model = LeNetModels(output_dim=config['dataset.mnistcifar.num_classes'],
                            width_factor=config['model.lenet.width'],
                            model_count=model_count,
                            dataset=config['dataset.name'],
                            feature_dim=config['model.lenet.feature_dim']).to(device)
    elif arch == "linear":
        model = LinearModels(input_dim=(28*28 if config['dataset.name'] == "mnist" else 32*32*3),
                             output_dim=config['dataset.mnistcifar.num_classes'],
                          model_count=model_count, device=device)
    return model


@section('optimizer')
@param('name')
@param('lr')
@param('momentum')
@param('scheduler')
def get_optimizer_and_scheduler(name, model, scheduler=False, lr=None, momentum=0):
    if name in ["SGD", "GD", "RMSProp", "Adam", "SGDPoison"]:
        if name == "RMSProp":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif name == "SGDPoisons":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        if scheduler == False:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999999], gamma=0.2)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2000, 3000], gamma=0.5)
    elif name == "guess":
        optimizer = None
        scheduler = None
    elif name == "NelderMead":
        optimizer = NelderMead(model.parameters(), alpha=1, gamma=2, rho=0.5, sigma=0.5)
        scheduler = None
    elif name == "PatternSearch":
        optimizer = PatternSearch(model.parameters())
        scheduler = None
    else:
        optimizer = None
        scheduler = None
    return optimizer, scheduler

@section('optimizer')
@param('epochs')
@param('batch_size')
@param('es_u')
@param('es_acc')
@param('print_intermediate_test_acc')
def train_sgd(
    train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size, epochs, es_u, es_acc=1, print_intermediate_test_acc=0):

    for epoch in range(epochs):
        idx_list = torch.randperm(len(train_data))
        for st_idx in range(0, len(train_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(train_data))]
            train_loss, train_acc = calculate_loss_acc(train_data[idx], train_labels[idx], model, loss_func)


            if es_u != float('inf'):
                with torch.no_grad():
                    train_loss_all, train_acc_all = calculate_loss_acc(train_data, train_labels,
                                                                       model.forward_normalize,
                                                                       loss_func)
                train_loss = torch.where((train_loss_all > es_u) | (train_acc_all < 1), train_loss,
                                         torch.zeros_like(train_loss))
                
                train_loss = train_loss[~train_loss.isnan()]
            optimizer.zero_grad()
            train_loss.sum().backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
                if len(train_loss[train_acc==1]) > 0:
                    print(f"train loss range: {train_loss[train_acc==1].max().item()} {train_loss[train_acc==1].min().item()}")
                train_loss = train_loss[~train_loss.isnan()]
                test_loss = test_loss[~test_loss.isnan()]
                print(
                    f"epoch {epoch} -  train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}")
                print(
                    f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
                if print_intermediate_test_acc:
                    _, test_acc = calculate_loss_acc(test_all_data.cuda(), test_all_labels.cuda(), model, loss_func, batch_size=batch_size)
                    print("test acc (all):", test_acc)
                if train_acc.mean() >= es_acc:
                    break     
    optimizer.zero_grad()


@section('optimizer')
@param('epochs')
@param('batch_size')
@param('es_u')
@param('es_acc')
@param('poison_factor')
def train_sgd_poison(
    train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size, epochs, es_u, es_acc=1, poison_factor=None,
    test_all_data=None, test_all_labels=None):
    test_all_data, test_all_labels = test_all_data.cuda(), test_all_labels.cuda()
    poison_test_labels = torch.tensor([1,0], device=test_all_labels.device)[test_all_labels]
    repeats = 10
    poison_data = torch.cat([train_data.repeat_interleave(repeats, dim=0), test_all_data], dim=0)
    poison_labels = torch.cat([train_labels.repeat_interleave(repeats), poison_test_labels], dim=0)
    for epoch in range(epochs):
        idx_list = torch.randperm(len(poison_data))
        for st_idx in range(0, len(poison_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(poison_data))]
            train_loss, train_acc = calculate_loss_acc(poison_data[idx], poison_labels[idx], model, loss_func)

            optimizer.zero_grad()
            train_loss.sum().backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_all_data, test_all_labels, model.forward_normalize, loss_func)
                poison_train_loss, poison_train_acc = calculate_loss_acc(poison_data, poison_labels, model.forward_normalize, loss_func)
                if len(train_loss[train_acc==1]) > 0:
                    print(f"train loss range: {train_loss[train_acc==1].max().item()} {train_loss[train_acc==1].min().item()}")
                train_loss = train_loss[~train_loss.isnan()]
                test_loss = test_loss[~test_loss.isnan()]
                print(
                    f"epoch {epoch} -  train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}")
                print(
                    f"epoch {epoch} - test acc all (max, min): {test_acc.max().item(): 0.2f}, {test_acc.min().item(): 0.2f}")
                
                print(
                    f"epoch {epoch} -  poison_train_acc: {poison_train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {poison_train_loss.mean().cpu().detach().item(): 0.4f}")

                if poison_train_acc.mean() >= es_acc:
                    break
    optimizer.zero_grad()

@section('optimizer')
@param('epochs')
@param('es_acc')
@param('es_u')
def train_gd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, epochs, es_u, es_acc=2):
    for epoch in range(epochs):
        train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model, loss_func)
        if es_u != float('inf'):
            train_loss = torch.where((train_loss > es_u) | (train_acc < 1),
                                     train_loss, torch.zeros_like(train_loss))
        optimizer.zero_grad()
        train_loss.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
                print(
                    f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
                print(
                    f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc.mean() >= es_acc:
                break
    optimizer.zero_grad()

@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_nm(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        optimizer.step(lambda: calculate_loss_acc(train_data, train_labels, model, loss_func)[0][0])
        if epoch % (epochs // 100 + 1) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(
                f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(
                f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc >= es_acc:
                break


@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_ps(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        optimizer.step(lambda: calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)[0][0])
        if epoch % (epochs // 100) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(
                f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(
                f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc >= es_acc:
                break

@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_ps_fast(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        model.pattern_search(train_data, train_labels, loss_func)
        if epoch % (epochs // 100) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(
                f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(
                f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc.mean() >= es_acc:
                break
    return model.get_model_subsets([0]).to(train_data.device)


@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_greedy_random(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        model.greedy_random(train_data, train_labels, loss_func)
        if epoch % (epochs // 300) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(
                f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(
                f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc.mean() >= es_acc:
                break
    return model.get_model_subsets([0]).to(train_data.device)


@section('optimizer')
@param('name')
def train(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, name, batch_size=None, es_u=None, test_all_data=None, test_all_labels=None):
    if name in ["SGD",  "RMSProp", "Adam"]:
        train_sgd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size=batch_size, es_u=es_u)
    elif name in ["SGDPoison"]:
        train_sgd_poison(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size=batch_size, es_u=es_u, test_all_data=test_all_data, test_all_labels=test_all_labels)
    elif name == "GD":
        train_gd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, es_u=es_u)
    elif name == "NelderMead":
        train_nm(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    elif name == "PatternSearch":
        train_ps(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    elif name == "PatternSearchFast":
        model = train_ps_fast(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    elif name == "GreedyRandom":
        model = train_greedy_random(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    else:
        pass
    return model

def convert_config_to_dict(config):
    config_dict = {}
    for path in config.entries.keys():
        try:
            value = config[path]
            if value is not None:
                config_dict['.'.join(path)] = config[path]
        except:
            pass
    return config_dict

def build_model_output_path(config, training_seed, data_seed, cur_num_samples):
    output_path = f"{config['output.folder']}/models/"
    output_path += f"{config['dataset.name']}_s{cur_num_samples}_"
    
    if config['model.arch'] == "lenet":
        output_path += f"lenet_w{config['model.lenet.width']}_" 
    elif config['model.arch'] == 'linear':
        output_path += f"linear_"
    elif config['model.arch'] == 'mlp':
        output_path += f"mlp_h{config['model.mlp.hidden_units']}"\
                    f"l{config['model.mlp.layers']}_"

    output_path += f"opt{config['optimizer.name'] }_"
    if config['optimizer.grad_norm_thres']:
        output_path += '_gnorm'
    output_path += f"dseed{data_seed}_tseed{training_seed}"
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the config needs to change to a vector of sample size (potential model counts as well) x loss bins??

    # it will then devote compute to the specific sample size & loss bins combination with the smallest number of trained models

    # always select a random training seed & data seed

    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.summary()
    config_ns = config.get()
 
    
    loss_thres = [float(v) for v in config['distributed.loss_thres'].split(",")]
    loss_bins = [(low, up) for low, up in zip(loss_thres[:-1], loss_thres[1:])]
    num_samples = [int(v) for v in config['distributed.num_samples'].split(",")]


    # create the table with counts
    os.makedirs(config['output.folder'], exist_ok=True)
    db_path = os.path.join(config['output.folder'], "model_stats.db")
    create_model_stats_table(db_path)
    while True:
        next_config = get_next_config(db_path=db_path, loss_bins=loss_bins, num_samples=num_samples)
        model_id, cur_loss_bin, cur_num_samples, data_seed, training_seed, cur_smallest_model_count = next_config

        if cur_smallest_model_count >= config['output.target_model_count']:
            print(f"Found models greater than target model count:{config['output.target_model_count']}, so ending the search")
            break
        if config['optimizer.name'] in ["SGD"]:
            cur_batch_size = min(cur_num_samples//2, config['optimizer.batch_size'])
            cur_model_count = config['model.model_count_times_batch_size']//cur_batch_size
        elif config['optimizer.name'] in ["SGDPoison"]:
            cur_batch_size = config['optimizer.batch_size']
            cur_model_count = config['model.model_count_times_batch_size']//cur_batch_size
        elif config['optimizer.name']=="guess":
            cur_batch_size = None
            cur_model_count = config['model.model_count_times_batch_size']//cur_num_samples
        else:
            cur_batch_size = None
            cur_model_count = config['model.model_count_times_batch_size']//cur_num_samples

        get_model_stats_summary(db_path)
        print("seed:", training_seed, data_seed)
        print("next config:", 
        json.dumps(
            {"num_samples":cur_num_samples, 
            "loss_bin": cur_loss_bin, 
            "model_count": cur_model_count}))

        es_l, es_u  = cur_loss_bin
        train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels = get_dataset(num_samples=cur_num_samples, seed=data_seed)
        torch.manual_seed(training_seed)
        device = torch.device('cuda:0')
        model = get_model(model_count=cur_model_count,device=device)

        perfect_model_count = 0
        perfect_model_weights = []
        loss_func = nn.CrossEntropyLoss(reduction='none')
        target_model_count_subrun = config['distributed.target_model_count_subrun']
        start_time = time.time()
        tested_model_count = 0
        prior_max = 0
        
        while perfect_model_count < target_model_count_subrun:
            # TODO: update this section to take in variable mult + simplifying the way that initializaiton is selected
            if config['model.init'] == "uniform":
                model.reinitialize()
            elif config['model.init'] == "uniform2":
                model.reinitialize(mult=2)
            elif config['model.init'] == "uniform5":
                model.reinitialize(mult=5)
            elif config['model.init'] == "sphere100":
                model.reinitialize_sphere(mult=100)
            elif config['model.init'] == "sphere200":
                model.reinitialize_sphere(mult=200)
            elif config['model.init'] == "regular":
                model.reset_parameters()
            optimizer, scheduler = get_optimizer_and_scheduler(model=model)
            model_result = train(
                train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, 
                batch_size=cur_batch_size, es_u=es_u, test_all_data=test_all_data, test_all_labels=test_all_labels)
            with torch.no_grad():
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model_result.forward_normalize, loss_func, batch_size=cur_batch_size)
                if train_acc.max() > prior_max:
                    print("max train acc:", train_acc.max().detach().cpu().item())
                    prior_max = train_acc.max()
                print("tested_model_count", tested_model_count)
            # filtering models based on loss threshold
            perfect_model_idxs = ((es_l< train_loss) & (train_loss <= es_u) & (train_acc == 1.0))

            perfect_model_count_cur = perfect_model_idxs.sum().detach().cpu().item()
            perfect_model_count += perfect_model_count_cur
            tested_model_count += cur_model_count
            if perfect_model_idxs.sum() > 0:
                if perfect_model_count > target_model_count_subrun:
                    remain_count = perfect_model_count_cur - (perfect_model_count - target_model_count_subrun)
                    tested_model_count -= ((perfect_model_count - target_model_count_subrun)/perfect_model_count_cur)*cur_model_count
                    perfect_model_weights.append(model_result.get_weights_by_idx(perfect_model_idxs.nonzero().squeeze(1)[:remain_count]))
                else:
                    perfect_model_weights.append(model_result.get_weights_by_idx(perfect_model_idxs))
        if len(perfect_model_weights) == 0:
            print(f"Failed to find a good model for set up {cur_num_samples} {cur_loss_bin}")
        else:
            train_time = time.time() - start_time
            print("="*50)

            # test that the model weights can be reloaded
            # concatenating the weights learned into a single model
            
            good_models_state_dict = dict()
            cat_dim = 1 if config['model.arch'] in ["mlp", "linear"] else 0
            for k in perfect_model_weights[0].keys():
                good_models_state_dict[k] = torch.cat(
                    [d[k].cpu() for d in perfect_model_weights], dim=cat_dim
                )

            if config['model.arch'] == "mlp":
                kwargs = {"input_dim": 2,
                        "output_dim": 2,
                        "layers": config['model.mlp.layers'],
                        "hidden_units": config['model.mlp.hidden_units'],
                        "model_count": target_model_count_subrun}
                new_models = MLPModels(**kwargs, device=torch.device('cpu'))

            elif config['model.arch'] == "linear":
                kwargs = {"input_dim": model.input_dim,
                        "output_dim": model.output_dim,
                        "model_count": target_model_count_subrun}
                new_models = LinearModels(**kwargs, device=torch.device('cpu'))

            elif  config['model.arch'] == "lenet":
                kwargs = {"output_dim": config['dataset.mnistcifar.num_classes'],
                        "width_factor": config['model.lenet.width'],
                        "model_count": target_model_count_subrun,
                        "dataset": config['dataset.name'],
                        "feature_dim": config['model.lenet.feature_dim']}
                new_models = LeNetModels(**kwargs)

            new_models.load_state_dict(good_models_state_dict)
            # show norm of the model
            model_l2_norm = 0
            model_linf_norm = 0
            for para in new_models.parameters():
                model_l2_norm += (para**2).sum()
                model_linf_norm = max(para.abs().max(), model_linf_norm)
            model_l2_norm = model_l2_norm ** 0.5
            print(f"model l2 norm: {model_l2_norm}")
            print(f"model linf norm: {model_linf_norm}")

            with torch.no_grad():
                loss, acc = calculate_loss_acc(train_data.cpu(), train_labels.cpu(), new_models, loss_func, batch_size=1)
                test_loss, test_acc = calculate_loss_acc(test_all_data.cpu(), test_all_labels.cpu(), new_models, loss_func, batch_size=1)
                print(f"verify that train acc is 100%: {acc.mean().item()}")
                print(f"test acc: {test_acc.mean().item(): 0.3f} ({test_acc.max().item(): 0.3f} , {test_acc.min().item(): 0.3f} )")

            # saving the models
            os.makedirs(os.path.join(config['output.folder'], "models"), exist_ok=True)
            output_path = build_model_output_path(config, training_seed, data_seed, cur_num_samples)
            print(f"Saving models at: {output_path}")
            # run specific features that are saved only for evaluate_minimas.py,these are used for resumming models
            saveconfig = convert_config_to_dict(config)
            saveconfig['dataset.num_samples'] = cur_num_samples
            saveconfig['training.seed'] = training_seed
            saveconfig['dataset.seed'] = data_seed
            saveconfig['training.es_l'], saveconfig['training.es_u'] = cur_loss_bin

            # save the model
            torch.save({"kwargs": kwargs,
                        "good_models_state_dict": good_models_state_dict,
                        "config": saveconfig},
                    output_path)


            update_model_stats_table(
                db_path, 
                model_id=model_id, data_seed=data_seed, 
                training_seed=training_seed, 
                num_training_samples=cur_num_samples, 
                loss_bin_l=es_l, 
                loss_bin_u=es_u, 
                test_acc=test_acc.mean().item(), 
                train_time=train_time,  
                perfect_model_count=target_model_count_subrun, 
                tested_model_count=tested_model_count, 
                save_path=output_path, status="COMPLETE")
        

