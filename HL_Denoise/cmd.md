python HL_Denoise\dn_train_visualize.py --epochs 1

python HL_Denoise\dn_train.py --model_type baseline --epochs 100 --batch_size 32 --num_workers 4
python HL_Denoise\dn_train.py --model_type simple --epochs 100 --batch_size 32 --num_workers 4