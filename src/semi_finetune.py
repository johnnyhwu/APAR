import os
import argparse
import wandb

import torch
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import numpy as np
import random
from rtdl import FTTransformer

from semi_datautils import load_data
from models import ArithmeticsEstimator, PretrainModel, APARModel

def args_parse():
    parser = argparse.ArgumentParser()

    # exp name
    parser.add_argument("--exp_name", type=str)
    
    # training
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--val_freq", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--total_epoch", type=int)
    
    # reprodicibility
    parser.add_argument("--device", type=int)
    parser.add_argument("--seed", type=int)
    
    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--numerical_feature_encoding", type=str)
    parser.add_argument("--categorical_feature_encoding", type=str)
    parser.add_argument("--target_encoding", type=str)

    # pretrain model
    parser.add_argument("--pretrain_wandb_id", type=str)
    parser.add_argument("--pretrain_wandb_name", type=str)

    # special
    parser.add_argument("--pred_loss_weight", type=float, default=1.0)
    parser.add_argument("--csty_loss_weight", type=float, default=0.5)
    parser.add_argument("--ftpy_loss_weight", type=float, default=0.5)


    args = parser.parse_args()
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_model(
    model, 
    train_loader,
    valid_loader,
    test_loader,
    optimizer,
    use_scheduler,
    total_epoch,
    accum_iter,
    val_freq,
    device,

    # special
    pred_loss_weight,
    csty_loss_weight,
    ftpy_loss_weight
):
    mse_criterion = nn.MSELoss().to(device)

    best_val_loss = float("inf")
    val_loss = validate_model(model, valid_loader, device)
    if test_loader is not None:
        test_model(model, test_loader, device)
    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), f"./ckpt/semi_finetune/0-0-{best_val_loss:.4f}.pt")
        torch.save(model.state_dict(), f"./ckpt/semi_finetune/{args.exp_name}.pt")
        # wandb.save(f"./ckpt/semi_finetune/0-0-{best_val_loss:.4f}.pt")
    
    if use_scheduler:
        lr_scheduler = StepLR(
            optimizer=optimizer,
            step_size=len(train_loader),
            gamma=0.98
        )

    for epoch in range(total_epoch):
        
        model.train()

        for iter_idx, data in enumerate(train_loader):
            cat_feat, num_feat, label, feat_mask = data
            cat_feat, num_feat, label, feat_mask = cat_feat.to(device), num_feat.to(torch.float32).to(device), label.to(torch.float32).to(device), feat_mask.to(torch.float32).to(device)

            if cat_feat.nelement() == 0:
                cat_feat = None
            elif num_feat.nelement() == 0:
                num_feat = None
                
            out, out_aug = model(num_feat, cat_feat, feat_mask)
            pred_loss = mse_criterion(out, label) * pred_loss_weight
            consist_loss = mse_criterion(out_aug, label) * csty_loss_weight
            pi_loss = torch.mean(model.pi) * ftpy_loss_weight
            loss = pred_loss + consist_loss + pi_loss

            # gradient accumulation
            loss = loss / accum_iter

            loss.backward()
            wandb.log({"train_step_loss": loss.item()})

            # gradient accumulation
            if ((iter_idx + 1) % accum_iter == 0) or (iter_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            
            # validate model
            if ((iter_idx+1) % val_freq == 0) or (iter_idx + 1 == len(train_loader)):
                val_loss = validate_model(model, valid_loader, device)
                if test_loader is not None:
                    test_model(model, test_loader, device)
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    # torch.save(model.state_dict(), f"./ckpt/semi_finetune/{epoch}-{iter_idx}-{best_val_loss:.4f}.pt")
                    torch.save(model.state_dict(), f"./ckpt/semi_finetune/{args.exp_name}.pt")
                    print(f"model is saved with val_loss = {best_val_loss:.4f}")
                    # wandb.save(f"./ckpt/semi_finetune/{epoch}-{iter_idx}-{best_val_loss:.4f}.pt")
                model.train()
    
    return best_val_loss

def validate_model(
    model, 
    val_loader,
    device
):
    criterion = nn.MSELoss().to(device)
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for data in val_loader:
            cat_feat, num_feat, label, feat_mask = data
            cat_feat, num_feat, label, feat_mask = cat_feat.to(device), num_feat.to(torch.float32).to(device), label.to(torch.float32).to(device), feat_mask.to(torch.float32).to(device)

            if cat_feat.nelement() == 0:
                cat_feat = None
            elif num_feat.nelement() == 0:
                num_feat = None

            out, _ = model(num_feat, cat_feat, feat_mask)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    wandb.log({"val_loss": epoch_loss})
    
    return epoch_loss

def test_model(
    model,
    test_loader,
    device
):
    
    criterion = nn.MSELoss().to(device)
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            cat_feat, num_feat, label = data
            cat_feat, num_feat, label = cat_feat.to(device), num_feat.to(torch.float32).to(device), label.to(torch.float32).to(device)

            feat_mask = torch.zeros((256, model.pi_logit.shape[0])).to(torch.float32).to(device)
            if cat_feat.nelement() == 0:
                cat_feat = None
            elif num_feat.nelement() == 0:
                num_feat = None

            output, _ = model(num_feat, cat_feat, feat_mask, inference=True)
            loss = criterion(output, label)
            running_loss += loss.item() * label.size(0)
                
    mse_loss = running_loss / len(test_loader.dataset)
    rmse_loss = mse_loss ** (1/2)
    print(f"[test loader] rmse: {rmse_loss}")
    


if __name__ == "__main__":

    # get argument
    args = args_parse()

    # cuda device
    torch.cuda.set_device(args.device)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # reprodicibility
    set_random_seed(args.seed)

    # load dataset
    train_loader, valid_loader, test_loaders, feat_info = load_data(args, load_test=False)
    # assert len(test_loaders.keys()) == 1
    # test_loader = test_loaders[list(test_loaders.keys())[0]]

    # build model
    model = FTTransformer.make_default(
        n_blocks=3,
        n_num_features=feat_info['num_feat_quantity'],
        cat_cardinalities=feat_info['cat_feat_cardinality'],
        last_layer_query_idx=[-1],
        d_out=1,
    )

    # load pretrain model
    encoder = FTTransformer.make_default(
        n_blocks=3,
        n_num_features=feat_info['num_feat_quantity'],
        cat_cardinalities=feat_info['cat_feat_cardinality'],
        last_layer_query_idx=[-1],
        d_out=1,
    )
    encoder.transformer.head = nn.Identity()
    decoder = ArithmeticsEstimator(hidden_dim=192)
    pretrain_model = PretrainModel(encoder, decoder).to(args.device)
    if args.pretrain_wandb_name is not None and args.pretrain_wandb_id is not None:
        wandb_model = wandb.restore(args.pretrain_wandb_name, run_path=f"johnnyhwu/APAR/{args.pretrain_wandb_id}")
        pretrain_model.load_state_dict(torch.load(wandb_model.name, map_location=torch.device("cpu")))

    # load encoder of pretrain model into model
    pretrain_encoder = pretrain_model.enc
    revised_state_dict = pretrain_encoder.state_dict()
    model.load_state_dict(revised_state_dict, strict=False)

    # wrap model with into a APAR model
    model = APARModel(ft_trans=model, input_dim=feat_info['num_feat_quantity'] + len(feat_info['cat_feat_cardinality'])).to(args.device)
    print(f"#Params of model: {sum(p.numel() for p in model.parameters())}")

    wandb.init(
        project="APAR",
        name=args.exp_name,
        config=vars(args)
    )
    wandb.define_metric("train_step_loss", summary="min")
    wandb.define_metric("val_loss", summary="min")
    wandb.watch(model, log="all", log_freq=1000, log_graph=True)

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=None,

        optimizer=model.ft_trans.make_default_optimizer(lr=args.learning_rate),
        use_scheduler=True,
        total_epoch=args.total_epoch,
        accum_iter=1,
        val_freq=args.val_freq,
        device=args.device,

        # special
        pred_loss_weight=args.pred_loss_weight,
        csty_loss_weight=args.csty_loss_weight,
        ftpy_loss_weight=args.ftpy_loss_weight
    )

    wandb.save(f"./ckpt/semi_finetune/{args.exp_name}.pt")
    wandb.finish()