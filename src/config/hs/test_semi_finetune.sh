python test.py \
--model apar \
--wnadb_id a1c8w7pa \
--wandb_name ckpt/semi_finetune/42-99-0.1088.pt \
--device 0 \
--seed 47 \
--dataset tw_hp_txg_h \
--batch_size 256 \
--val_ratio 0.1 \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta