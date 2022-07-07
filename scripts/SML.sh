python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name MSL  --data_path dataset/MSL --feature_size 55    --logger_name exp1_MSL
python main.py --anormly_ratio 1  --epoch 20      --batch_size 64     --mode test    --dataset_name MSL   --data_path dataset/MSL  --feature_size 55      --logger_name exp1_MSL


python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name MSL  --data_path dataset/MSL --feature_size 55    --logger_name exp2_MSL
python main.py --anormly_ratio 3  --epoch 20      --batch_size 64     --mode test    --dataset_name MSL   --data_path dataset/MSL  --feature_size 55      --logger_name exp2_MSL

python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name MSL  --data_path dataset/MSL --feature_size 55    --logger_name exp3_MSL
python main.py --anormly_ratio 5  --epoch 20      --batch_size 64     --mode test    --dataset_name MSL   --data_path dataset/MSL  --feature_size 55      --logger_name exp3_MSL

