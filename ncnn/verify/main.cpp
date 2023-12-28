#include <iostream>
#include <random>

#include "net.h"

static std::default_random_engine generator;
static std::normal_distribution<float> distribution(0.0, 1.0);


void randn_ncnn(ncnn::Mat &mat, int w, int h, int c)
{
    mat.create(w, h, c, (size_t) 4);

    memset(mat.data, 0.f, w * h * c * 4);

    #pragma omp parallel for num_threads(4)
    for (int k = 0; k < c; k++)
    {
        float *c_ptr = mat.channel(k);
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                c_ptr[0] = distribution(generator);
                c_ptr++;
            }
        } 
    }
}


int main()
{
    ncnn::Option opt;
    // opt.lightmode = true;
    opt.num_threads = 4;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;


    ncnn::Net nanodet;
    nanodet.opt = opt;

    // nanodet.load_param("../model/nanodet-plus-m-1.5x_416.param");
    // nanodet.load_model("../model/nanodet-plus-m-1.5x_416.bin");

    nanodet.load_param("../quantize/nanodet-plus-m_416_int8.param");
    nanodet.load_model("../quantize/nanodet-plus-m_416_int8.bin");

    const std::vector<const char*>& input_names = nanodet.input_names();
    const std::vector<const char*>& output_names = nanodet.output_names();

    // for (auto name: input_names)
    // {
    //     std::cout << name << std::endl;
    // }

    // for (auto name: output_names)
    // {
    //     std::cout << name << std::endl;
    // }

    ncnn::Mat input;
    randn_ncnn(input, 416, 416, 3);
    {
        ncnn::Extractor ex = nanodet.create_extractor();
        ex.input("in0", input);
        ncnn::Mat output;
        ex.extract("out1", output);

        std::cout << "c: " << output.c << std::endl;
        std::cout << "h: " << output.h << std::endl;
        std::cout << "w: " << output.w << std::endl;
    }

    return 0; 
}