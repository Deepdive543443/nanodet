model=nanodet-plus-m-1.5x_416


find imagenet-sample-images-master/ -type f > imagelist.txt
./ncnn2table \
    ../model/${model}.param \
    ../model/${model}.bin \
    imagelist.txt \
    ${model}.table \
    mean=[103.53,116.28,123.675] norm=[0.017429,0.017507,0.017125] \
    shape=[416,416,3] \
    pixel=BGR thread=8 method=kl

./ncnn2int8 ../model/${model}.param ../model/${model}.bin ${model}_int8.param ${model}_int8.bin ${model}.table