@echo off
python HL_Denoise\dn_train.py --model_type baseline --epochs 100 --batch_size 32 --num_workers 8
pause
