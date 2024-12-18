python test.py \
--model apar \
--wandb_name ckpt/semi_finetune/tw_hp_ntpc_a-APAR-semi_finetune_division\ \(gyq92ef5\).pt \
--device 0 \
--seed 47 \
--dataset tw_hp_ntpc_a \
--batch_size 256 \
--val_ratio 0.1 \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta