python test.py \
--model apar \
--wnadb_id r29fbu3o \
--wandb_name ckpt/semi_finetune/25-1399-0.0439.pt \
--device 0 \
--seed 47 \
--dataset yp \
--batch_size 256 \
--val_ratio 0.1 \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta