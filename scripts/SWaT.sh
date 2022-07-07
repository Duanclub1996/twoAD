

###############################anormly_ratio######################################
python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name SWaT  --data_path dataset/SWaT --feature_size 1   --logger_name exp1
python main.py --anormly_ratio 0.5  --epoch 20      --batch_size 64     --mode test    --dataset_name SWaT   --data_path dataset/SWaT  --feature_size 1   --logger_name exp1


python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name SWaT  --data_path dataset/SWaT --feature_size 1   --logger_name exp2
python main.py --anormly_ratio 1  --epoch 20      --batch_size 64     --mode test    --dataset_name SWaT   --data_path dataset/SWaT  --feature_size 1   --logger_name exp2


python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name SWaT  --data_path dataset/SWaT --feature_size 1   --logger_name exp3
python main.py --anormly_ratio 1.5  --epoch 20      --batch_size 64     --mode test    --dataset_name SWaT   --data_path dataset/SWaT  --feature_size 1   --logger_name exp3


python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name SWaT  --data_path dataset/SWaT --feature_size 1   --logger_name exp4
python main.py --anormly_ratio 2  --epoch 20      --batch_size 64     --mode test    --dataset_name SWaT   --data_path dataset/SWaT  --feature_size 1   --logger_name exp4


python main.py  --epoch 20   --batch_size 64  --mode train --dataset_name SWaT  --data_path dataset/SWaT --feature_size 1   --logger_name exp5
python main.py --anormly_ratio 3  --epoch 20      --batch_size 64     --mode test    --dataset_name SWaT   --data_path dataset/SWaT  --feature_size 1   --logger_name exp5


###############################K######################################