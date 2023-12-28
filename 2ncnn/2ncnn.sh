model=nanodet-plus-m_416

python ../2ncnn.py --config ../../config/${model}.yml --model ../../model/${model}.pth --save_script ../raw_output/${model}
../pnnx ${model}.pt inputshape=[1,3,320,320] \
ncnnparam=../ncnn_model/${model}.param \
ncnnbin=../ncnn_model/${model}.bin