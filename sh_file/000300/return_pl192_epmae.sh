python -u create_result_file.py \
--file_name return_pl192_epmae # Create a file_name folder in the RESULTS directory
#命名及路径
result_file=return_pl192_epmae # usually same as the following model_id
root_path=./dataset/raw_data # file path
data_path=000300.SH.return.csv # file name
model_id=return_pl192_epmae # model name
seq_len=96 # input sequence length. Out of cuda if too large. Max: Auto (600), non-linear Transforer (700),Transformer (750)
lab_len=48 # start token length, lab_len < seq_len
pred_len=192 # prediction sequence length, set as the same size as the testing dataset
var_len=20 # variable length. Date not included.
train_val_len=4385 # length of test+val
train_ratio=0.7 # test_len/train_val_len
val_ratio=0.1 # val_len/train_val_len
# Note that we can set train_val_len = whole dataset length (train+val+test), 
# in this case, train_ratio + val_ratio < 1, and test_ratio = 1 - (train_ratio + val_ratio)
target=next_return # prediction target
loss=EPMAE

# informer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_inf \
  --model Informer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# FEDformer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_FED \
  --model FEDformer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# Transformer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_trans \
  --model Transformer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# TimesNet
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_times \
  --model TimesNet \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# Pyraformer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_pyra \
  --model Pyraformer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# Nonstationary_Transformer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_non \
  --model Nonstationary_Transformer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# Crossformer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_cross \
  --model Crossformer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# Autoformer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_auto \
  --model Autoformer \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# LightTS
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_light \
  --model LightTS \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1

# Dlinear
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id\_DL \
  --model DLinear \
  --data custom \
  --features M \
  --target $target \
  --seq_len $seq_len \
  --label_len $lab_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $var_len \
  --dec_in $var_len \
  --c_out $var_len \
  --des 'Exp' \
  --itr 1 \
  --loss $loss \
  --result_file $result_file \
  --train_val_len $train_val_len \
  --train_ratio $train_ratio \
  --val_ratio $val_ratio \
  --is_return 1