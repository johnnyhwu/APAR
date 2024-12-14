cd ../..

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_subtract\ \(x5kugd5r\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id x5kugd5r \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_subtract.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_subtract\ \(x5kugd5r\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id x5kugd5r \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_subtract.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.1 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_subtract\ \(x5kugd5r\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id x5kugd5r \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_subtract.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.2 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_subtract\ \(x5kugd5r\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id x5kugd5r \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_subtract.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.3 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_subtract\ \(x5kugd5r\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id x5kugd5r \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_subtract.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.4 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_subtract\ \(x5kugd5r\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id x5kugd5r \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_subtract.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.5 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_1\ \(j5ldar2n\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id j5ldar2n \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_1\ \(j5ldar2n\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id j5ldar2n \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.1 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_1\ \(j5ldar2n\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id j5ldar2n \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.2 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_1\ \(j5ldar2n\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id j5ldar2n \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.3 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_1\ \(j5ldar2n\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id j5ldar2n \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.4 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_1\ \(j5ldar2n\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id j5ldar2n \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.5 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_5\ \(5lsnmvxa\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id 5lsnmvxa \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_5.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_5\ \(5lsnmvxa\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id 5lsnmvxa \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_5.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.1 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_5\ \(5lsnmvxa\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id 5lsnmvxa \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_5.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.2 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_5\ \(5lsnmvxa\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id 5lsnmvxa \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_5.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.3 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_5\ \(5lsnmvxa\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id 5lsnmvxa \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_5.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.4 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_5\ \(5lsnmvxa\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id 5lsnmvxa \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_5.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.5 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_1\ \(azo6kzmb\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id azo6kzmb \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_1\ \(azo6kzmb\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id azo6kzmb \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.1 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_1\ \(azo6kzmb\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id azo6kzmb \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.2 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_1\ \(azo6kzmb\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id azo6kzmb \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.3 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_1\ \(azo6kzmb\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id azo6kzmb \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.4 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_1\ \(azo6kzmb\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id azo6kzmb \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_1.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.5 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_01\ \(wsran2rf\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id wsran2rf \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_01.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_01\ \(wsran2rf\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id wsran2rf \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_01.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.1 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_01\ \(wsran2rf\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id wsran2rf \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_01.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.2 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_01\ \(wsran2rf\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id wsran2rf \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_01.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.3 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_01\ \(wsran2rf\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id wsran2rf \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_01.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.4 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_multiply_0_01\ \(wsran2rf\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id wsran2rf \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_multiply_0_01.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.5 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_division\ \(vhsz6c47\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id vhsz6c47 \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_division.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_division\ \(vhsz6c47\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id vhsz6c47 \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_division.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.1 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_division\ \(vhsz6c47\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id vhsz6c47 \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_division.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.2 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_division\ \(vhsz6c47\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id vhsz6c47 \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_division.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.3 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_division\ \(vhsz6c47\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id vhsz6c47 \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_division.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.4 \
--ftpy_loss_weight 0.01

python semi_finetune.py \
--exp_name tw_hp_txg_h-APAR-semi_finetune_division\ \(vhsz6c47\) \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset yp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id vhsz6c47 \
--pretrain_wandb_name ckpt/self_pretrain/tw_hp_txg_h-APAR-self_pretrain_division.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.5 \
--ftpy_loss_weight 0.01

