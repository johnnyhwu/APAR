cd ../..

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_subtract \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op sub

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_multiply_1 \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op mul_1

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_multiply_0_5 \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op mul_0_5

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_multiply_0_1 \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op mul_0_1

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_multiply_0_01 \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op mul_0_01

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_division \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op div

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_arithmetic_mean \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op ari_mean

python self_pretrain.py \
--exp_name fp-APAR-self_pretrain_geometric_mean \
--batch_size 256 \
--val_ratio 0.1 \
--val_freq 100 \
--learning_rate 1e-3 \
--total_epoch 150 \
--device 0 \
--seed 47 \
--dataset fp \
--numerical_feature_encoding log2_delta \
--categorical_feature_encoding label \
--target_encoding log2_delta \
--arithm_op geo_mean