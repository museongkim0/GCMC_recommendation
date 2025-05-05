from comet_ml import Experiment
import torch
import yaml
import argparse

from dataset import MCDataset
from model import GAE
from trainer import Trainer
from utils import calc_rmse, print_pred, ster_uniform, random_init, init_xavier, init_uniform, Config


def main(cfg, comet=False):
    cfg = Config(cfg)
    # comet-ml setting
    if comet:
        experiment = Experiment(
            api_key=cfg.api_key,
            project_name=cfg.project_name,
            workspace=cfg.workspace
        )
        experiment.log_parameters(cfg)
    else:
        experiment = None

    #Reflect line arguments to confing
    ap = argparse.ArgumentParser()
    ap.add_argument("-fd", "--feature_data", type=str, default="None",
                choices=['nn', 'pca', 'vae', 'graph1', 'graph2'],
                help="Feature Dataset.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")
    ap.add_argument("-e", "--epochs", type=int, default=1000,
                help="Number training epochs")
    ap.add_argument("-hi", "--hidden_size", type=int, nargs=2, default=[500, 75],
                help="Number hidden units in 1st and 2nd layer")
    ap.add_argument("-ac", "--accum", type=str, default="add", choices=['add', 'stack'],
                help="Accumulation function: add or stack.")
    ap.add_argument("-do", "--drop_prob", type=float, default=0.7,
                help="Dropout fraction")
    ap.add_argument("-nb", "--num_basis", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")
    ap.add_argument("-nf", "--num_features", type=int, default=1,
                help="""Number of columns in the feature vector. 1 if the feature vector does not exist """)
    #ap.add_argument("-gpu", "--gpu_id", type=int, default=1,help="""cpu:0 gpu:1 """)

    ap.set_defaults(testing=False)
    args = vars(ap.parse_args())

    print('Settings:')
    print(args, '\n')

    cfg.feature = args['feature_data']
    cfg.lr = args['learning_rate']
    cfg.epochs = args['epochs']
    cfg.hidden_size = args['hidden_size']
    cfg.accum = args['accum']
    cfg.drop_prob = args['drop_prob']
    cfg.num_basis = args['num_basis']
    cfg.num_features = args['num_features']
    #cfg.gpu_id = args['gpu_id']

    # device and dataset setting
    device = (torch.device(f'cuda:{cfg.gpu_id}')
        if torch.cuda.is_available() and cfg.gpu_id >= 0
        else torch.device('cpu'))
    dataset = MCDataset(cfg, cfg.root, cfg.dataset_name)
    data = dataset[0].to(device)

    # add some params to config
    cfg.num_nodes = dataset.num_nodes
    cfg.num_relations = dataset.num_relations
    cfg.num_users = int(data.num_users)

    # set and init model
    model = GAE(cfg, random_init).to(device)
    model.apply(init_xavier)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr, weight_decay=cfg.weight_decay,)
    #optimizer = torch.optim.SGD(model.parameters(),lr=cfg.lr, weight_decay=cfg.weight_decay,)
    #optimizer = torch.optim.Adagrad(model.parameters(),lr=cfg.lr, weight_decay=cfg.weight_decay,)
    # train
    trainer = Trainer(cfg,
        model, dataset, data, calc_rmse, print_pred, optimizer, experiment,
    )
    trainer.training(cfg.epochs)


if __name__ == '__main__':
    with open('config.yml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
    # main(cfg, comet=True)
