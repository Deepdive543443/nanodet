
python ../2ncnn.py --config ../../config/nanodet-plus-m_320.yml --model ../../demo/model/nanodet-plus-m_320.pth --save_script ../script/nanodet-plus-m_320
../script/pnnx ../script/nanodet-plus-m_320.pt inputshape=[1,3,320,320] ncnnparam=../ncnn_model/nanodet-plus-m_320.param ncnnbin=../ncnn_model/nanodet-plus-m_320.bin