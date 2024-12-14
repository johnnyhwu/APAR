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

from self_datautils import load_data

from models import Reconstructor, PretrainModel

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

    # model
    parser.add_argument("--model_pretrain_type", type=str)
    parser.add_argument("--model_corr_prob", type=float)

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

def optimization_param_groups(model):
    no_wd_names = ['feature_tokenizer', 'normalization', '.bias']

    def needs_wd(name):
        return all(x not in name for x in no_wd_names)

    return [
        {'params': [v for k, v in model.named_parameters() if needs_wd(k)]},
        {
            'params': [v for k, v in model.named_parameters() if not needs_wd(k)],
            'weight_decay': 0.0,
        },
    ]

def make_default_optimizer(model, lr):
    return torch.optim.AdamW(
        optimization_param_groups(model),
        lr=lr,
        weight_decay=1e-5,
    )

def train_model(
    model, 
    train_loader,
    test_loader,
    optimizer,
    use_scheduler,
    total_epoch,
    accum_iter,
    val_freq,
    device,
    pretrain_type
):
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()

    best_val_loss = float("inf")
    val_loss = validate_model(model, test_loader, device, pretrain_type)
    if val_loss <= best_val_loss:
        best_val_loss = val_loss
    
    if use_scheduler:
        lr_scheduler = StepLR(
            optimizer=optimizer,
            step_size=len(train_loader),
            gamma=0.98
        )

    for epoch in range(total_epoch):

        model.train()

        for iter_idx, data in enumerate(train_loader):
            cat_feat, num_feat, cat_feat_recon, num_feat_recon, mask_recon = data
            cat_feat, num_feat, cat_feat_recon, num_feat_recon, mask_recon = cat_feat.to(torch.long).to(device), num_feat.to(torch.float32).to(device), cat_feat_recon.to(torch.long).to(device), num_feat_recon.to(torch.float32).to(device), mask_recon.to(torch.float32).to(device)

            if cat_feat.nelement() == 0:
                cat_feat = None
            elif num_feat.nelement() == 0:
                num_feat = None
            
            num_feat_pred, cat_feat_pred, mask_pred = model(num_feat, cat_feat)

            # numerical feature reconstruction loss
            num_feat_recon_loss = mse_criterion(num_feat_pred, num_feat_recon)

            # categorical feature reconstruction loss
            cat_feat_recon_loss = 0
            for idx, pred in enumerate(cat_feat_pred):
                cat_feat_recon_loss += ce_criterion(pred, cat_feat_recon[:, idx])

            # feature reconstruction loss
            if num_feat_recon.shape[1] == 0:
                feat_recon_loss = cat_feat_recon_loss
            else:
                feat_recon_loss = num_feat_recon_loss + cat_feat_recon_loss
            
            # mask reconstruction loss
            mask_recon_loss = bce_criterion(mask_pred, mask_recon)

            # total loss
            if pretrain_type == "both":
                total_loss = 0.5*feat_recon_loss + mask_recon_loss
            elif pretrain_type == "feat":
                total_loss = feat_recon_loss
            elif pretrain_type == "mask":
                total_loss = mask_recon_loss
            else:
                raise Exception("pretrain_type must be one of ['both', 'feat', 'mask']")

            # gradient accumulation
            total_loss = total_loss / accum_iter
            total_loss.backward()
            wandb.log({"train/total_loss": total_loss.item()})

            # gradient accumulation
            if ((iter_idx + 1) % accum_iter == 0) or (iter_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # validate model
            if ((iter_idx+1) % val_freq == 0) or (iter_idx + 1 == len(train_loader)):
                val_loss = validate_model(model, test_loader, device, pretrain_type)
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"./ckpt/self_pretrain/{args.dataset}_{args.model_pretrain_type}_model.pt")
                    print(f"model is saved with val_loss = {best_val_loss:.4f}")
                    wandb.save(f"./ckpt/self_pretrain/{args.dataset}_{args.model_pretrain_type}_model.pt")
                model.train()
    
    return best_val_loss

def validate_model(
    model, 
    val_loader,
    device,
    pretrain_type
):
    mse_criterion = nn.MSELoss().to(device)
    ce_criterion = nn.CrossEntropyLoss().to(device)
    bce_criterion = nn.BCELoss().to(device)

    model.eval()
    running_loss = 0

    with torch.no_grad():
        for data in val_loader:
            cat_feat, num_feat, cat_feat_recon, num_feat_recon, mask_recon = data
            cat_feat, num_feat, cat_feat_recon, num_feat_recon, mask_recon = cat_feat.to(torch.long).to(device), num_feat.to(torch.float32).to(device), cat_feat_recon.to(torch.long).to(device), num_feat_recon.to(torch.float32).to(device), mask_recon.to(torch.float32).to(device)

            if cat_feat.nelement() == 0:
                cat_feat = None
            elif num_feat.nelement() == 0:
                num_feat = None

            num_feat_pred, cat_feat_pred, mask_pred = model(num_feat, cat_feat)

            # numerical feature reconstruction loss
            num_feat_recon_loss = mse_criterion(num_feat_pred, num_feat_recon)

            # categorical feature reconstruction loss
            cat_feat_recon_loss = 0
            for idx, pred in enumerate(cat_feat_pred):
                cat_feat_recon_loss += ce_criterion(pred, cat_feat_recon[:, idx])

            # feature reconstruction loss
            if num_feat_recon.shape[1] == 0:
                feat_recon_loss = cat_feat_recon_loss
            else:
                feat_recon_loss = num_feat_recon_loss + cat_feat_recon_loss
            
            # mask reconstruction loss
            mask_recon_loss = bce_criterion(mask_pred, mask_recon)

            # total loss
            if pretrain_type == "both":
                loss = 0.5*feat_recon_loss + mask_recon_loss
            elif pretrain_type == "feat":
                loss = feat_recon_loss
            elif pretrain_type == "mask":
                loss = mask_recon_loss
            else:
                raise Exception("pretrain_type must be one of ['both', 'feat', 'mask']")

            # statistics
            running_loss += loss.item() * mask_recon.size(0)

    # statistics
    epoch_loss = running_loss / len(val_loader.dataset)


    wandb.log({
        "val/total_loss": epoch_loss
    })
    
    return epoch_loss

if __name__ == "__main__":

    # get argument
    args = args_parse()

    # cuda device
    torch.cuda.set_device(args.device)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # reprodicibility
    set_random_seed(args.seed)

    # load dataset
    train_loader, valid_loader, feat_info = load_data(args, "vime")
    
    # build model
    encoder = FTTransformer.make_default(
        n_blocks=3,
        n_num_features=feat_info['num_feat_quantity'],
        cat_cardinalities=feat_info['cat_feat_cardinality'],
        last_layer_query_idx=[-1],
        d_out=1,
    )
    encoder.transformer.head = nn.Identity()
    decoder = Reconstructor(
        hidden_dim=192,
        num_feat_quantity=feat_info['num_feat_quantity'],
        cat_feat_cardinality=feat_info['cat_feat_cardinality']
    )
    model = PretrainModel(encoder, decoder).to(args.device)
    print(f"#Params of model: {sum(p.numel() for p in model.parameters())}")

    wandb.init(
        project="APAR",
        name=args.exp_name,
        config=vars(args)
    )
    wandb.define_metric("train/total_loss", summary="min")
    wandb.define_metric("val/total_loss", summary="min")

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=valid_loader,
        optimizer=make_default_optimizer(model, lr=args.learning_rate),
        use_scheduler=True,
        total_epoch=args.total_epoch,
        accum_iter=1,
        val_freq=args.val_freq,
        device=args.device,
        pretrain_type=args.model_pretrain_type,
    )

    wandb.finish()