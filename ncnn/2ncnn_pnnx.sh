model=nanodet-plus-m_416

python 2ncnn_pnnx.py \
  --config ../config/${model}.yml \
  --model ../model/${model}.pth \
  --save_script raw_output/${model} \
  --model_name=${model}