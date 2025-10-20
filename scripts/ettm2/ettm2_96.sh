# export CUDA_VISIBLE_DEVICES=0

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

model_name=FreqMixAttNet
run_date='test'
root_path='./data'

seq_len=96
e_layers=2
learning_rate=0.03
d_model=32
n_heads=8
d_ff=32
train_epochs=6
patience=6
batch_size=128
dropout=0.2
down_sampling_layers=2
down_sampling_window=2
devices='0,1'

python -u run_model.py \
--gpu 0 \
--task_name long_term_forecast \
--is_training 1 \
--devices $devices \
--data_path ETTm2.csv \
--root_path $root_path \
--model_id $run_date'_ETTm2' \
--model $model_name \
--data ETTm2 \
--features M \
--seq_len $seq_len \
--label_len 0 \
--pred_len 96 \
--e_layers $e_layers \
--decomp_method wavelet \
--enc_in 7 \
--c_out 7 \
--des 'Exp' \
--itr 1 \
--patch_len 16 \
--d_model 32 \
--n_heads 4 \
--d_ff $d_ff \
--down_sampling_layers $down_sampling_layers \
--down_sampling_window $down_sampling_window \
--levels 3 \
--freq_weight 8 \
--alpha 0.3 \
--l1l2_alpha 0.035 \
--learning_rate 0.03 \
--dropout 0.2 \
--train_epochs $train_epochs \
--patience $patience \
--batch_size $batch_size \
--aug_weight 0.01 \
--aug_constrast_weight1 0.01 \
--aug_constrast_weight2 0.03 \
--mix_rate 0.1 \
--jitter_ratio 0.3


