import argparse
import wandb
import pickle

import torch
import torch.nn as nn

import numpy as np
import random
import pandas as pd
from datautils import load_data

def args_parse():
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument("--model", type=str)

    ##################################################
    ###  model-specific hp ###########################
    ##################################################

    # mlp
    parser.add_argument("--mlp_input_size", type=int, default=None)
    parser.add_argument("--mlp_hidden_size", type=int, default=None)
    parser.add_argument("--mlp_num_blocks", type=int, default=None)

    # fttrans
    parser.add_argument("--ft_num_blocks", type=int, default=3)

    # vime
    parser.add_argument("--vime_input_size", type=int, default=None)
    parser.add_argument("--vime_hidden_size", type=int, default=None)
    parser.add_argument("--vime_num_blocks", type=int, default=None)

    # apar
    parser.add_argument("--repr_layer_name", type=str, default=None) # specify the layer name in order to save the activation of the layer for representation analysis

    ##################################################
    ##################################################
    ##################################################

    # checkpoint
    parser.add_argument("--wnadb_id", type=str)
    parser.add_argument("--wandb_name", type=str)

    # reprodicibility
    parser.add_argument("--device", type=int)
    parser.add_argument("--seed", type=int)
    
    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--numerical_feature_encoding", type=str)
    parser.add_argument("--categorical_feature_encoding", type=str)
    parser.add_argument("--target_encoding", type=str)

    # others
    parser.add_argument("--report_name", type=str, default="report")
    parser.add_argument("--not_write_result_to_wandb", action="store_true")

    args = parser.parse_args()
    return args

def set_random_seed(seed):
    import os
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def test_model(
    model,
    model_type,
    test_loader,
    feat_info,
    device,
    report_log,
    repr_layer_name=None
):
    def mape_criterion(pred, label):
        pred, label = np.array(pred), np.array(label)
        return np.abs((pred-label)/label).mean()
    
    def hitrate_criterion(pred, label, hit):
        pred, label = np.array(pred), np.array(label)
        result = (np.abs((pred-label)/label) < hit).mean()
        return result
    
    mse_criterion = nn.MSELoss().to(device)
    mse_loss = 0
    label_lst = []
    pred_lst = []

    model.eval()

    # for name, module in model.named_modules():
    #     print(name)
    # exit()

    # register a forward hook to save the activation of the layer
    if repr_layer_name is not None:
        layer_activation = {}
        def hook_fn_wrapper(name):
            def hook_fn(module, input, output):
                if name not in layer_activation:
                    layer_activation[name] = [output]
                else:
                    layer_activation[name].append(output)
            return hook_fn
        
        # if there are multiple layers to be analyzed
        if "," in repr_layer_name:
            repr_layer_name = repr_layer_name.split(",")
        else:
            repr_layer_name = [repr_layer_name]
        
        # find specified layer(module)
        for name, module in model.named_modules():
            if name in repr_layer_name:
                _ = module.register_forward_hook(hook_fn_wrapper(name))

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            cat_feat, num_feat, label = data
            cat_feat, num_feat, label = cat_feat.to(device), num_feat.to(torch.float32).to(device), label.to(torch.float32).to(device)

            if model_type == 'apar':
                feat_mask = torch.zeros((256, model.pi_logit.shape[0])).to(torch.float32).to(device)
                if cat_feat.nelement() == 0:
                    cat_feat = None
                elif num_feat.nelement() == 0:
                    num_feat = None
                output, _ = model(num_feat, cat_feat, feat_mask, inference=True)
            elif model_type in ['mlp', 'vime']:
                cat_feat = cat_feat.to(torch.float32)
                feat = torch.concat([cat_feat, num_feat], dim=1)
                output = model(feat)
            elif model_type == "fttrans":
                if cat_feat.nelement() == 0:
                    cat_feat = None
                elif num_feat.nelement() == 0:
                    num_feat = None
                output = model(num_feat, cat_feat)
            else:
                raise Exception

            loss = mse_criterion(output, label)
            mse_loss += loss.item() * label.size(0)
            label_lst.append(label.item())
            pred_lst.append(output.item())
            
            print(f"progress: {(idx+1)}/{len(test_loader)}", end="\r")
    
    # calculate metric
    mse_loss = mse_loss / len(test_loader.dataset)
    label_price, pred_price = [], []
    for l, p in zip(label_lst, pred_lst):
        label_price.append((2 ** l - feat_info['target_log2_delta']).values[0])
        pred_price.append((2 ** p - feat_info['target_log2_delta']).values[0])
    
    # save actual label and prediction
    report_log['label'].extend(label_price)
    report_log['pred'].extend(pred_price)
    
    mape = round(mape_criterion(pred_price, label_price), 4)
    hit10 = round(hitrate_criterion(pred_price, label_price, 0.1), 4)
    hit20 = round(hitrate_criterion(pred_price, label_price, 0.2), 4)

    # save the activation of the layers (entire dict) as a torch tensor
    if repr_layer_name is not None:
        torch.save(layer_activation, f"repr_layer.pt")

    return mse_loss, hit10, hit20, mape


if __name__ == "__main__":

    # get argument
    args = args_parse()

    # cuda device
    torch.cuda.set_device(args.device)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # reprodicibility
    set_random_seed(args.seed)

    # build dataset
    _, _, test_loaders, feat_info = load_data(args)

    if args.model == "apar":
        from apar.rtdl import FTTransformer
        from apar.models import APARModel
        model = APARModel(
            ft_trans=FTTransformer.make_default(
                n_blocks=3,
                n_num_features=feat_info['num_feat_quantity'],
                cat_cardinalities=feat_info['cat_feat_cardinality'],
                last_layer_query_idx=[-1],
                d_out=1
            ),
            input_dim=feat_info['num_feat_quantity']+len(feat_info['cat_feat_cardinality'])
        ).to(args.device)
    
    elif args.model == "fttrans":
        from fttrans.rtdl import FTTransformer
        model = FTTransformer.make_default(
            n_blocks=args.ft_num_blocks,
            n_num_features=feat_info['num_feat_quantity'],
            cat_cardinalities=feat_info['cat_feat_cardinality'],
            last_layer_query_idx=[-1],
            d_out=1,
        )
    
    elif args.model == "mlp":
        from mlp.models import MLPModel

        assert args.mlp_input_size is not None \
        and args.mlp_hidden_size is not None \
        and args.mlp_num_blocks is not None

        model = MLPModel(
            input_size=args.mlp_input_size,
            hidden_size=args.mlp_hidden_size,
            num_blocks=args.mlp_num_blocks,
            output_size=1
        )
    
    elif args.model == "vime":
        from vime.models import PretrainModel
        from vime.models import FinetuneModel

        assert args.vime_input_size is not None \
        and args.vime_hidden_size is not None \
        and args.vime_num_blocks is not None

        pretrained_model = PretrainModel(
            input_size=args.vime_input_size,
            hidden_size=args.vime_hidden_size,
            num_blocks=args.vime_num_blocks,
            output_size=args.vime_input_size,

            num_feat=1,
            cat_feat=[1, ]
        )

        model = FinetuneModel(
            encoder=pretrained_model.encoder,
            hidden_size=args.vime_hidden_size,
        )
    
    abalation_report = {}

    if args.model != "vime":
        # if wandb_id is not None, restore the specific model from wandb
        if args.wnadb_id is not None:
            abalation_report[args.wnadb_id] = {'mse': None, 'rmse': None}
        else:
            # restore all models according to the wandb_name (exp_name) -> used to run ablation study of apar
            assert args.model == "apar"

            api = wandb.Api()
            experiment_name = args.wandb_name.split("/")[-1].split(".")[0]
            all_runs = api.runs(path="johnnyhwu/APAR", filters={"config.exp_name": experiment_name})

            for run in all_runs:
                abalation_report[run.id] = {
                    'wandb_id': run.id,
                    'csty_loss_weight': run.config['csty_loss_weight'],
                    'ftpy_loss_weight': run.config['ftpy_loss_weight'],
                    'learning_rate': run.config['learning_rate'],
                    'mse': None,
                    'rmse': None
                }
    else:
        # VIME does not use wandb
        abalation_report['none'] = {'mse': None, 'rmse': None}
    
    for ablation_idx, wandb_id in enumerate(abalation_report.keys()):
        print(f"ablation_idx: {ablation_idx}/{len(abalation_report)}")

        if args.model != "vime":
            wandb_model = wandb.restore(args.wandb_name, run_path=f"johnnyhwu/APAR/{wandb_id}", replace=True)
            model.load_state_dict(torch.load(wandb_model.name, map_location="cpu"), strict=True)
        else:
            model.load_state_dict(torch.load(f"/home/johnnyhwu/ssd1/apar/method/vime/ckpt/train/{args.dataset}.pt", map_location="cpu"), strict=True)

        model = model.to(args.device)
        print(f"#Params of model: {sum(p.numel() for p in model.parameters())}")

        # evaluate model
        report = {
            'name': [],
            'hit10': [],
            'hit20': [],
            'mape': [],
            'mse': []
        }

        report_log = {
            'label': [],
            'pred': []
        }

        for key, value in test_loaders.items():
            print(f"Evaluation: {key}")
            mse, hit10, hit20, mape = test_model(
                model=model,
                model_type=args.model,
                test_loader=value,
                feat_info=feat_info,
                device=args.device,
                report_log=report_log,
                repr_layer_name=args.repr_layer_name
            )
            report['name'].append(key)
            report['hit10'].append(hit10)
            report['hit20'].append(hit20)
            report['mape'].append(mape)
            report['mse'].append(mse)
        
        # save actual label and prediction
        pd.DataFrame.from_dict(report_log).to_csv(f'{args.report_name}_log.csv')
        
        # average
        hit10_avg = round((sum(report['hit10']) / len(report['hit10'])) * 100, 2)
        hit20_avg = round((sum(report['hit20']) / len(report['hit20'])) * 100, 2)
        mape_avg = round((sum(report['mape']) / len(report['mape'])) * 100, 2)
        mse_avg = round((sum(report['mse']) / len(report['mse'])), 8)

        report['name'].append("avg")
        report['hit10'].append(hit10_avg)
        report['hit20'].append(hit20_avg)
        report['mape'].append(mape_avg)
        report['mse'].append(mse_avg)

        # export report
        pd.DataFrame.from_dict(report).to_csv(f'{args.report_name}.csv')

        # performance
        print("[average performance]")
        print(f"hit10: {hit10_avg}")
        print(f"hit20: {hit20_avg}")
        print(f"mape: {mape_avg}")
        print(f"mse: {mse_avg}")
        abalation_report[wandb_id]['mse'] = mse_avg
        abalation_report[wandb_id]['rmse'] = mse_avg ** 0.5
        print(abalation_report[wandb_id])

        # save mse to wandb
        if not args.not_write_result_to_wandb:
            api = wandb.Api()
            run = api.run(f"johnnyhwu/APAR/{args.wnadb_id}")
            run.notes = f"{mse}"
            run.save()
    
    with open(f"{args.report_name}_ablation.pkl", "wb") as f:
            pickle.dump(abalation_report, f)