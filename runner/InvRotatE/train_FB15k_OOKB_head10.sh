python3 -u codes/run.py --cuda --gpu_id 2 --ookb -sp --do_train --eval_task LP --do_valid --do_test --data_path data/FB15k_OOKB_head10 --model RotatE -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv -lr 0.001 --max_steps 100000 --warm_up_steps 10000 -save models/InversE_RotatE_FB15kOOKB_head10_final --test_batch_size 16 -de --valid_steps 5000