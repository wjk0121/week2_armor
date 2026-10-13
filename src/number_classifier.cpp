#include "number_classifier.hpp"
#include "types.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>

NumberClassifier::NumberClassifier(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "num_clf")
{
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opt);

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_  = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
    std::printf("数字网络: 输入 %s [1,%d,%d,1] (mlp)\n",
                input_name_.c_str(), kNumW, kNumH);
}

size_t NumberClassifier::numLabels() const { return numberLabels().size(); }

std::string NumberClassifier::label(int idx) const
{
    const auto& v = numberLabels();
    if (idx < 0 || idx >= static_cast<int>(v.size())) return "?";
    return v[idx];
}

std::pair<int, float> NumberClassifier::classify(const cv::Mat& warp) const
{
    //  契约检查: 必须是 32x28 的 BGR 矫正图, 尺寸不对说明调用方有误
    if (warp.empty() || warp.cols != kBarW || warp.rows != kBarH)
        return {-1, 0.f};

    //  取中间 20 列即数字区(灯条位于两侧, 数字居中)
    const int x0 = (kBarW - kNumW) / 2;
    const cv::Mat roi = warp(cv::Rect(x0, 0, kNumW, kNumH));

    //  预处理: 灰度 -> Otsu 二值(黑底白字, 正色) -> /255 -> 浮点
    //  该 mlp 训练数据为竖直数字条的二值图, 与 rm_auto_aim 参考实现一致
    cv::Mat gray, f32;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    gray.convertTo(f32, CV_32F, 1.0 / 255.0);
    if (!f32.isContinuous()) f32 = f32.clone();

    //  张量 shape [1,20,28,1]; f32 为 28 行 x 20 列连续内存(行优先 560 个 float)
    //  按内存顺序直接填入 —— 这一顺序经过枚举实验验证, 转置/旋转/横图都会识别错误
    const std::array<int64_t, 4> ishape{1, kNumW, kNumH, 1};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, f32.ptr<float>(), static_cast<size_t>(kNumW) * kNumH,
        ishape.data(), ishape.size());

    const char* in_names[]  = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 in_names, &input, 1, out_names, 1);

    const float* out = outputs[0].GetTensorData<float>();
    const int n = static_cast<int>(numLabels());

    //  softmax 找最大类别; 减去最大值保证数值稳定, 输出概率作为置信度
    int best = 0;
    for (int i = 1; i < n; ++i) if (out[i] > out[best]) best = i;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp(out[i] - out[best]);
    return {best, static_cast<float>(1.0 / sum)};
}
