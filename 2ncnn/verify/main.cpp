#include <iostream>
#include <random>
#include <unistd.h>

#include "net.h" //ncnn

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/videoio.hpp> //opencv_mobile

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

    int target_size = 416;
    nanodet.load_param("../quantize/nanodet-plus-m_320_int8.param");
    nanodet.load_model("../quantize/nanodet-plus-m_320_int8.bin");

    const std::vector<const char*>& input_names = nanodet.input_names();
    const std::vector<const char*>& output_names = nanodet.output_names();


    // Load image
    cv::Mat inputs = cv::imread("../test.jpg");
    int w = inputs.cols;
    int h = inputs.rows;
    float scale = 1.f;

    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(inputs.data, ncnn::Mat::PIXEL_RGB2BGR, inputs.cols, inputs.rows, w, h);
    ncnn::Mat in_pad;
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;


    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    {
        ncnn::Extractor ex = nanodet.create_extractor();
        ex.input("data", in_pad);
        ncnn::Mat output;
        ex.extract("output", output);

        std::cout << "in_pad c: " << in_pad.c << std::endl;
        std::cout << "in_pad h: " << in_pad.h << std::endl;
        std::cout << "in_pad w: " << in_pad.w << std::endl;

        std::cout << "output c: " << output.c << std::endl;
        std::cout << "output h: " << output.h << std::endl;
        std::cout << "output w: " << output.w << std::endl;
    }


    cv::Mat output(target_size,target_size, CV_8UC3);
    in_pad.to_pixels(output.data, ncnn::Mat::PIXEL_BGR2RGB);
    cv::imwrite("out.jpg", output);

    return 0; 
}