python3 -u codes/run.py --cuda --gpu_id 0 --ookb -op --do_train --eval_task TC --do_valid --do_test --data_path data/WN11_OOKB_tail1000 --model RotatE -n 128 -b 1024 -d 300 -g 0.5 -lr 0.001 -a 1.0 -adv --uni_weight -r 0.00001 --max_steps 20000 --warm_up_steps 10000 -save models/InversE_RotatE_WN11OOKB_tail1000_final --test_batch_size 8 --valid_steps 2000 -de