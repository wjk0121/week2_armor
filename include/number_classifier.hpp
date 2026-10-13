#pragma once

//  数字识别网络(mlp.onnx)封装
//  模型契约: 输入 [1,20,28,1] 灰度 float32; 输出 [1,9] logits 需 softmax
//  标签: 1 2 3 4 5 outpost guard base negative(与 models/label.txt 一致)
//  输入布局(已实测): 20 宽 x 28 高的竖直数字条, Otsu 黑底白字, /255, 内存顺序直塞

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class NumberClassifier
{
public:
    explicit NumberClassifier(const std::string& model_path);

    //  输入契约: 32x28 的装甲板透视矫正图(BGR, 灯条在左右两侧, 数字居中)
    //  输出 (类别索引, 置信度); 尺寸不符或失败返回 {-1, 0}
    std::pair<int, float> classify(const cv::Mat& warp) const;

    std::string label(int idx) const;
    size_t numLabels() const;

private:
    Ort::Env env_;                              //  必须先于 session_ 构造
    std::unique_ptr<Ort::Session> session_;
    //  ORT 的 AllocatedStringPtr 不可默认构造, 成员直接存 std::string(创建后拷贝一次)
    std::string input_name_, output_name_;

    static constexpr int kBarW = 32;            //  warp 契约宽
    static constexpr int kBarH = 28;            //  warp 契约高
    static constexpr int kNumW = 20;            //  数字条宽(取 warp 中间 20 列)
    static constexpr int kNumH = 28;            //  数字条高(即张量 shape 的第 2 维)
};
