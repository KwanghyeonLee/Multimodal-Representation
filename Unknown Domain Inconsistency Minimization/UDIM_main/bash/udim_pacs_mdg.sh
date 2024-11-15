cd ..
cuda_id=0
seed=0
data_path="your_directory"
algorithm=IADA_DG_FINAL
batch_size=16
dset=PACS

for syn_lr in 10
do
for worst_weight in 1
do
for sam_warm_up in 5
do
CUDA_VISIBLE_DEVICES=${cuda_id} python3 train_all.py exp_name \
--dataset ${dset} --data_dir ${data_path} --seed ${seed} --algorithm ${algorithm} \
--checkpoint_freq 100 --alpha 0.001 --lr 3e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False\
 --syn_lr ${syn_lr} --worst_weight ${worst_weight} --sam_warm_up_ratio ${sam_warm_up} --batch_size_change True \
 --batch_size ${batch_size} --advstyle False --is_single True
done
done
done