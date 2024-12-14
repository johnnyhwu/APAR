cd ../..

python semi_finetune.py \
--exp_name kp-APAR-semi_finetune_subtraction \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 0.0005 \
--total_epoch 100 \
--device 0 \
--seed 47 \
--dataset kp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--pretrain_wandb_id kyjhzewt \
--pretrain_wandb_name ckpt/self_pretrain/149-299-0.0057-model.pt \
--pred_loss_weight 1.0 \
--csty_loss_weight 0.01 \
--ftpy_loss_weight 0.01