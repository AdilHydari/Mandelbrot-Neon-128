#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <arm_neon.h>
#include <complex>

static const int RE_START = -2;
static const int RE_END = 1;
static const int IM_START = -1;
static const int IM_END = 1;
static const int MAX_ITER = 80;

// int mandelbrot(std::complex<double> const &c)
// {
//     std::complex<double> z{0, 0};
//     int n = 0;
//     while (std::norm(z) <= 4 && n < MAX_ITER)
//     {
//         z = z * z + c;
//         n += 1;
//     }
//     return n;
// }

void computeMandelbrot(cv::Mat &mandelbrotImg)
{
    int width = mandelbrotImg.cols;
    int height = mandelbrotImg.rows;
    double wid = (double)width;
    double hei = (double)height;
    double reWidth = RE_END - RE_START;
    double imWidth = IM_END - IM_START;

    float32x4_t reStartVec = vdupq_n_f32(RE_START);
    float32x4_t imStartVec = vdupq_n_f32(IM_START);
    float32x4_t reWidthVec = vdupq_n_f32(reWidth);
    float32x4_t imWidthVec = vdupq_n_f32(imWidth);
    float32x4_t maxIterVec = vdupq_n_f32(MAX_ITER);
    #pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < width; x += 4)
    {
        float32x4_t xVec = vdupq_n_f32((float)x);
        float32x4_t xScale = vdivq_f32(xVec, vdupq_n_f32((float)wid));
        float32x4_t rePart = vaddq_f32(reStartVec, vmulq_f32(xScale, reWidthVec));

        for (int y = 0; y < height; ++y)
        {
            float32x4_t yVec = vdupq_n_f32((float)y);
            float32x4_t yScale = vdivq_f32(yVec, vdupq_n_f32((float)hei));
            float32x4_t imPart = vaddq_f32(imStartVec, vmulq_f32(yScale, imWidthVec));

            float32x4_t real = rePart;
            float32x4_t imag = imPart;

            float32x4_t normX = vmulq_f32(real, real);
            float32x4_t normY = vmulq_f32(imag, imag);

            float32x4_t zNorm = vaddq_f32(normX, normY);
            uint32x4_t mask = vcleq_f32(zNorm, vdupq_n_f32(4.0f));

            uint32_t maskValues;
            vst1q_u32(&maskValues, mask);

            int iterations[4] = {0, 0, 0, 0};

				/*
				 * Secret sauce (simple but took a while) essentially, we set the value mask to be equal to 4 at first (for the number of lanes to process) then we want to take that value and then we need to turn this 4 lane mask  value into a single lane for comparison's sake which is what vst1q_u32 does. 
				 * From here, we create the iterations array in order to keep track of our iterations (since that is required for the output of the mandelbrot set [for color])
				 * Since vst1q_u32 has 4 float32 values stored in a single 128 bit register, we need to segment these values apart, this is what the bitwise AND does from our least significant (0x1) and shifts to correspond with the next element in the register. 
				 * In the end, the product is that we will have 4 subsequent lanes processing concurrently, however, if one lane If a pixel diverges before reaching MAX_ITER, this will result in all other subsequent threads dying. 
				*/

            while ((maskValues & 0x1) && iterations[0] < MAX_ITER &&
                   (maskValues & 0x2) && iterations[1] < MAX_ITER &&
                   (maskValues & 0x4) && iterations[2] < MAX_ITER &&
                   (maskValues & 0x8) && iterations[3] < MAX_ITER)
            {
                float32x4_t tempReal = vaddq_f32(vsubq_f32(vmulq_f32(real, real), vmulq_f32(imag, imag)), rePart);
                float32x4_t tempImag = vaddq_f32(vmulq_f32(vmulq_f32(vdupq_n_f32(2.0f), real), imag), imPart);

                real = tempReal;
                imag = tempImag;

                normX = vmulq_f32(real, real);
                normY = vmulq_f32(imag, imag);

                zNorm = vaddq_f32(normX, normY);
                mask = vcleq_f32(zNorm, vdupq_n_f32(4.0f));

                vst1q_u32(&maskValues, mask);

//The following accounts for divergence, if there is not any divergence, we can add the bit to the amount of iterations (for the creation of our image)
                iterations[0] += (maskValues & 0x1);
//The result of the following operation is either 0 or 2. By right-shifting this result by 1 (>> 1), it's converted to either 0 or 1. Same logic for the rest of the mask values. 
                iterations[1] += (maskValues & 0x2) >> 1;
                iterations[2] += (maskValues & 0x4) >> 2;
                iterations[3] += (maskValues & 0x8) >> 3;
            }

            for (int i = 0; i < 4; ++i)
            {
                int m = iterations[i];

                float hue = 255 * m / (float)MAX_ITER;
                float saturation = 255.0;
                float value = (m < MAX_ITER) ? 255.0 : 0;

                mandelbrotImg.at<cv::Vec3b>(y, x + i) = cv::Vec3b((uchar)hue, (uchar)saturation, (uchar)value);
            }
        }
    }
}

int main()
{
    int width = 7680;
    int height = 4320;

    cv::Mat mandelbrotImg(height, width, CV_8UC3);
    computeMandelbrot(mandelbrotImg);

    cv::cvtColor(mandelbrotImg, mandelbrotImg, cv::COLOR_HSV2BGR);
    cv::imwrite("./mandel.png", mandelbrotImg);

    return 0;
}
