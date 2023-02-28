import torch
import argparse
from utils import *
from train_distributed import get_dataset
import os
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--model_filenames', nargs='+', default=[f"db_SGD_u3.png"], help="model file names")
parser.add_argument('--num_samples', default=16, type=int, help="number of samples")
parser.add_argument('--shorten', type=int, help="shorten the number of models used for evaluation")
parser.add_argument('--normalize', action="store_true", help="normalize matrices")
parser.add_argument('--make_permutation_invariant', action="store_true", help="make the matrices permutation invariant")
parser.add_argument('--bin_count', type=int, default=10, help="number of bins to separate training loss")
parser.add_argument('--output_folder', default='output')
parser.add_argument('--visualize_db', action='store_true', help="visualize decsion boundaries")
parser.add_argument('--testing_seed', type=int, default=100, help="seed for testing dataset")
parser.add_argument('--testing_count', type=int, default=16, help="number of examples for testing")
parser.add_argument('--worst_case', action='store_true', help="worst case test accuracy models")
parser.add_argument('--lower_loss', type=float,  help="lower upper bound for plotting", default=0)
parser.add_argument('--suffix', default='')

args = parser.parse_args()

models_list = []
optimizers = []
valid_model_filenames = []

for model_filename in args.model_filenames:
    try:
        models_dict = torch.load(model_filename)
    except:
        continue
    valid_model_filenames.append(model_filename)
    if models_dict['config']['model.arch'] == "lenet":
        if 'feature_base_dim' in models_dict['kwargs']:
            models_dict['kwargs']['feature_dim'] = models_dict['kwargs']['feature_base_dim']
            del models_dict['kwargs']['feature_base_dim']
        models = LeNetModels(**models_dict["kwargs"])
    elif models_dict['config']['model.arch'] == "linear":
        models = LinearModels(**models_dict["kwargs"], device=torch.device("cpu"))
    else:
        models = MLPModels(**models_dict["kwargs"], device=torch.device("cpu"))
        hidden_units = models_dict["kwargs"]["hidden_units"]
    models.load_state_dict(models_dict["good_models_state_dict"])
    if args.shorten:
        models = models.shorten(args.shorten)
    model_count = models.model_count
    model_configs_dict = models_dict["config"]
    model_configs_dict = defaultdict(lambda : None, model_configs_dict)
    optimizer = model_configs_dict['optimizer.name']
    optimizers.append(optimizer)
    if args.make_permutation_invariant:
        models.make_permutation_invariant()
    if args.normalize:
        models.normalize()
    models_list.append(models)


train_data, train_labels, test_data, test_labels, test_complete_data, test_complete_labels = get_dataset(
    name = model_configs_dict['dataset.name'],
    num_samples = model_configs_dict['dataset.num_samples'],
    seed = model_configs_dict['dataset.seed'],
    num_classes = model_configs_dict['dataset.mnistcifar.num_classes'],
    noise = model_configs_dict['dataset.kink.noise'],
    margin= model_configs_dict['dataset.kink.margin'],
)

_, _, test_data, test_labels, test_complete_data, test_complete_labels = get_dataset(
    name = model_configs_dict['dataset.name'],
    num_samples = 100,
    seed = model_configs_dict['dataset.seed'],
    num_classes = model_configs_dict['dataset.mnistcifar.num_classes'],
    noise = model_configs_dict['dataset.kink.noise'],
    margin= model_configs_dict['dataset.kink.margin'],
)
train_data, train_labels, test_data, test_labels, test_complete_data, test_complete_labels = train_data.cpu(), train_labels.cpu(), test_data.cpu(), test_labels.cpu(), test_complete_data.cpu(), test_complete_labels.cpu()

loss_func = nn.CrossEntropyLoss(reduction='none')
train_accs_list = []
test_accs_list = []
train_losses_list = []
test_losses_list = []
# TODO: separate evaluation with visualization code


for models_i, models in enumerate(models_list):
    if True:
        # get average margins
        model = models.get_model_subsets([0])
        pred = model(train_data).squeeze(1)
        top2logits = pred.topk(k=2, dim=1).values
        avgmargin = (top2logits[:, 0] - top2logits[:, 1]).mean()
        print(f"average margin is : {avgmargin}")

    with torch.no_grad():
        train_losses, train_accs = calculate_loss_acc(train_data, train_labels, models, loss_func, batch_size=1)
        test_losses, test_accs = calculate_loss_acc(test_complete_data, test_complete_labels, models, loss_func, batch_size=1)
    train_accs_list.append(train_accs)
    test_accs_list.append(test_accs)
    train_losses_list.append(train_losses)
    test_losses_list.append(test_losses)
    print(f"model_filename: {valid_model_filenames[models_i]}")
    print(f"models: {models_i}")
    print(f"model_count: {models.model_count}")
    print(f"training acc: {train_accs.mean()}")
    print(f"overall test acc:{test_accs.mean(), test_accs.std()}")
    print("minimum and max train losses", train_losses.min().item(), train_losses.max().item())



print("by train loss", "-"*20)
intervals = torch.linspace(
    min([train_losses.min().item() for train_losses in train_losses_list]),
    max([train_losses.max().item() for train_losses in train_losses_list]),
    args.bin_count+1)

for models_i in range(len(models_list)):
    subset_models_list = []
    train_losses = train_losses_list[models_i]
    test_losses = test_losses_list[models_i]
    test_accs = test_accs_list[models_i]
    print("=" * 20, f"model_filename: {valid_model_filenames[models_i]}")
    for i, (l, u) in enumerate(zip(intervals[:-1], intervals[1:])):
        idx = (train_losses >= l) & (train_losses <= u)
        if idx.sum()> 0:
            print(f"bin{i}-interval:({l.item(): 0.3f},{u.item(): 0.3f}) count:{idx.sum().item(): 4.0f},"
                  f"test accs (mean,min,max): "
                  f"{test_accs[idx].mean().cpu().item(): 0.3f}"
                  f",{test_accs[idx].min().cpu().item(): 0.3f},"
                  f"{test_accs[idx].max().cpu().item(): 0.3f}, "
                  f"test loss: {test_losses[idx].mean().cpu().item(): 0.3f}")

if args.visualize_db:
    subset_models_list = []
    for models_i, models in enumerate(models_list):
        train_losses = train_losses_list[models_i]
        test_losses = test_losses_list[models_i]
        test_accs = test_accs_list[models_i]
        for i, (l,u) in enumerate(zip(intervals[:-1], intervals[1:])):
            idx = (train_losses >= l) & (train_losses <= u - args.lower_loss)
            if (idx == 1).sum() == 0:
                continue
            print(f"interval: {l.item(): 0.3f}, {u.item(): 0.3f}, count:{idx.sum().item()} "
                  f"test accs (mean, min, max): {test_accs[idx].mean().cpu().item(): 0.3f} "
                  f",{test_accs[idx].min().cpu().item(): 0.3f},{test_accs[idx].max().cpu().item(): 0.3f}"
                  f"test loss: {test_losses[idx].mean().cpu().item(): 0.3f}")

            subset_idx = torch.nonzero(idx).squeeze(1)
            if args.worst_case:
                subset_idx = (-test_accs).topk(9).indices
            print(i, subset_idx)
            if len(subset_idx) >= 9:
                subset_idx = subset_idx[:min(9, len(subset_idx))]
                subset_models = models.get_model_subsets(subset_idx)
                subset_models_list.append(subset_models)
            else:
                subset_models_list.append(models.get_model_subsets([0]*9))
    
    os.makedirs(args.output_folder, exist_ok=True)
    filename = f'db_{optimizer}_u{hidden_units}_s{model_configs_dict["dataset.num_samples"]}_seed{model_configs_dict["dataset.seed"]}_noise{model_configs_dict["dataset.kink.noise"]}_{args.suffix}.png'
    filename = os.path.join(args.output_folder, filename)
    print("saving the file at ", filename)
    visualize_decision_boundary(subset_models_list,
                                data=(train_data, train_labels),
                                filename=filename)
