#pragma once

// 神经网络方案: Infantry-v8n 直接回归 4 个关键点 + 颜色 + 兵种
// 模型契约(已实测): 输入 images[1,3,480,640] RGB/255
// 输出 output0[1,21,6300]:
// 通道 0~3  : 颜色(0蓝 1红 2白 3紫), 已是概率, 直接使用
// 通道 4~12 : 兵种 9 类, 已是概率, 直接使用
// 通道 13~20: 4 个关键点(x,y), 直接是 640x480 输入坐标系的像素值
// 置信度 = 颜色概率 x 兵种概率

#include "types.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

class NnDetector
{
public:
    explicit NnDetector(const std::string& model_path);

    std::vector<Armor> detect(const cv::Mat& frame) const;

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    //  ORT 的 AllocatedStringPtr 不可默认构造, 成员直接存 std::string(创建后拷贝一次)
    std::string input_name_, output_name_;

    static constexpr int   kInW = 640;             //  模型输入宽(契约固定)
    static constexpr int   kInH = 480;             //  模型输入高
    static constexpr float kConfThresh = 0.5f;     //  颜色x兵种联合置信度阈值
    static constexpr float kNmsThresh  = 0.45f;    //  NMS IoU 阈值
};
