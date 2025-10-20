# export CUDA_VISIBLE_DEVICES=0

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"


model_name=FreqMixAttNet
run_date='test'
root_path='./data'

seq_len=96
e_layers=2
learning_rate=0.015
d_model=32
n_heads=128
d_ff=32
train_epochs=20
patience=6
batch_size=8
dropout=0.15
down_sampling_layers=2
down_sampling_window=2
aug_constrast_weight1=0.01
aug_constrast_weight2=0.008
freq_weight=4
alpha=0.4 
l1l2_alpha=0.035
aug_weight=0.04
mix_rate=0.1
jitter_ratio=0.3
devices='0'
decomp_method='wavelet'

python -u run_model.py \
--gpu 0 \
--task_name long_term_forecast \
--is_training 1 \
--devices $devices \
--data_path exchange_rate.csv \
--root_path $root_path \
--model_id $run_date'_exchange_rate' \
--model $model_name \
--data custom \
--features M \
--seq_len $seq_len \
--label_len 0 \
--pred_len 336 \
--e_layers $e_layers \
--decomp_method wavelet \
--enc_in 8 \
--dec_in 8 \
--des 'Exp' \
--itr 1 \
--patch_len 16 \
--d_model $d_model \
--n_heads $n_heads \
--d_ff $d_ff \
--down_sampling_layers $down_sampling_layers \
--down_sampling_window $down_sampling_window \
--levels 3 \
--freq_weight $freq_weight \
--alpha $alpha \
--l1l2_alpha $l1l2_alpha \
--learning_rate $learning_rate \
--dropout $dropout \
--train_epochs $train_epochs \
--patience $patience \
--batch_size $batch_size \
--aug_weight $aug_weight \
--mix_rate $mix_rate \
--aug_constrast_weight1 $aug_constrast_weight1 \
--aug_constrast_weight2 $aug_constrast_weight2 \
--jitter_ratio $jitter_ratio \
