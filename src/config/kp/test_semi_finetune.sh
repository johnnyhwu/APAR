python test.py \
--model apar \
--wnadb_id 16fc0jsb \
--wandb_name ckpt/semi_finetune/97-99-0.0027.pt \
--device 0 \
--seed 47 \
--dataset kp \
--batch_size 256 \
--val_ratio 0.1 \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta