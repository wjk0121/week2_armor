#include "nn_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

NnDetector::NnDetector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "nn_det")
{
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opt);

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_  = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
    std::printf("装甲板网络: 输入 %s [1,3,%d,%d] (Infantry-v8n)\n",
                input_name_.c_str(), kInH, kInW);
}

std::vector<Armor> NnDetector::detect(const cv::Mat& frame) const
{
    if (frame.empty()) return {};

    //  letterbox: 等比缩放到 640x480 居中(本数据 4:3 同比例, pad 恒为 0, 保持通用写法)
    const float scale = std::min(static_cast<float>(kInW) / frame.cols,
                                 static_cast<float>(kInH) / frame.rows);
    const int nw = static_cast<int>(std::round(frame.cols * scale));
    const int nh = static_cast<int>(std::round(frame.rows * scale));
    const int pad_x = (kInW - nw) / 2, pad_y = (kInH - nh) / 2;
    cv::Mat resized, input_img(kInH, kInW, CV_8UC3, {114, 114, 114});
    cv::resize(frame, resized, {nw, nh});
    resized.copyTo(input_img(cv::Rect(pad_x, pad_y, nw, nh)));

    //  预处理: BGR->RGB, /255, 拆通道拼 CHW 连续内存
    cv::Mat rgb;
    cv::cvtColor(input_img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<float> blob(static_cast<size_t>(3) * kInH * kInW);
    std::vector<cv::Mat> planes;
    cv::split(rgb, planes);
    for (int i = 0; i < 3; ++i)
        std::memcpy(blob.data() + static_cast<size_t>(i) * kInH * kInW,
                    planes[i].data, static_cast<size_t>(kInH) * kInW * sizeof(float));

    const std::array<int64_t, 4> ishape{1, 3, kInH, kInW};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(), ishape.data(), ishape.size());

    const char* in_names[]  = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 in_names, &input, 1, out_names, 1);
    const float* data = outputs[0].GetTensorData<float>();

    //  输出形状 [1, 通道数, 锚点数] 由模型决定, 从张量信息读取而不是写死:
    //  写死 6300 在换模型/换输入分辨率时会静默错位, 得到完全错误的框
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[1] < 21)
    {
        std::printf("错误: 装甲板网络输出形状异常(应为 [1,>=21,N])\n");
        return {};
    }
    const int anchors = static_cast<int>(out_shape[2]);
    auto at = [&](int c, int a) -> float { return data[c * anchors + a]; };

    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> all_kps;
    std::vector<float> scores;
    std::vector<int> cls_ids;

    for (int a = 0; a < anchors; ++a)
    {
        //  颜色取 4 通道最大(0蓝1红2白3紫), 输出已是概率
        int color = 0;
        for (int c = 1; c < 4; ++c) if (at(c, a) > at(color, a)) color = c;

        //  兵种取 9 通道最大, 输出已是概率
        int cls = 0;
        for (int c = 1; c < 9; ++c) if (at(4 + c, a) > at(4 + cls, a)) cls = c;

        //  联合置信度直接相乘 —— 模型导出图中已包含 sigmoid, 不要重复套用
        const float conf = at(color, a) * at(4 + cls, a);
        if (conf < kConfThresh) continue;

        //  4 个关键点, 顺序: 左上/左下/右下/右上; 映射到契约顺序 左上、右上、右下、左下
        const int ch_order[4] = {0, 3, 2, 1};   //  契约序 i 对应模型点序
        std::vector<cv::Point2f> kps(4);
        for (int i = 0; i < 4; ++i)
        {
            const int p = ch_order[i];
            kps[i].x = (at(13 + 2 * p,     a) - pad_x) / scale;   //  坐标直接是输入系像素值
            kps[i].y = (at(13 + 2 * p + 1, a) - pad_y) / scale;   //  减去 pad 并还原比例
        }
        all_kps.push_back(kps);
        boxes.push_back(cv::boundingRect(kps));
        scores.push_back(conf);
        cls_ids.push_back(cls);
    }

    //  NMS 去重: 同一块板的多锚点只保留最优
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConfThresh, kNmsThresh, keep);

    std::vector<Armor> armors;
    for (int k : keep)
    {
        Armor armor;
        armor.corners    = all_kps[k];
        armor.number     = cls_ids[k];
        armor.confidence = scores[k];
        armor.label      = armorLabels()[cls_ids[k]];
        //  尺寸启发: 英雄(1)/前哨站(6)/基地(7,8)为大装甲板, 其余按小板解算
        armor.size = (cls_ids[k] == 1 || cls_ids[k] == 6 ||
                      cls_ids[k] == 7 || cls_ids[k] == 8)
                     ? ArmorSize::BIG : ArmorSize::SMALL;
        armors.push_back(armor);
    }
    return armors;
}
