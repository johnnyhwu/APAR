python test.py \
--model apar \
--wnadb_id n3fgrm9x \
--wandb_name ckpt/semi_finetune/65-54-0.0004.pt \
--device 0 \
--seed 47 \
--dataset bs \
--batch_size 256 \
--val_ratio 0.1 \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta