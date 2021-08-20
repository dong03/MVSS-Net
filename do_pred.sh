if [ "$#" -ne 1 ]; then
    echo "Usage: $0 test_collection"
    exit
fi

test_collection=$1
test_file=./data/$test_collection.txt
model_path=./ckpt/mvssnet_casia.pt
save_dir=../save_out/
threshold=0.5

python inference.py --model_path $model_path --test_file $test_file --save_dir $save_dir
