#pragma once

//  传统方案: 颜色阈值找灯条 -> 几何约束配对 -> 数字网络辅助确认

#include "camera.hpp"
#include "number_classifier.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>
#include <vector>

class TraditionalDetector
{
public:
    TraditionalDetector(const CameraParams& params, const NumberClassifier& clf);

    //  返回本帧全部装甲板(可能为空, 调用方按"未识别"处理)
    std::vector<Armor> detect(const cv::Mat& frame) const;

private:
    //  颜色差分 + 阈值 + 轮廓几何筛选, 得到候选灯条
    //  color_diff_thresh: 差分二值化阈值(主召回用 60, 弱召回传 30)
    std::vector<cv::RotatedRect> findLightBars(const cv::Mat& frame,int color_diff_thresh) const;
    //  两两配对, 数字网络确认后生成 Armor
    std::vector<Armor> pairLightBars(const std::vector<cv::RotatedRect>& bars,
                                     const cv::Mat& frame) const;

    const CameraParams& params_;
    const NumberClassifier& clf_;

     //  调参集中处: 含义见 .cpp 注释
    static constexpr int   kColorDiffThresh    = 60;    //  颜色差分二值化阈值(主召回)
    static constexpr int   kColorDiffThreshLow = 30;    //  弱阈值: 半亮灯条二次召回
    static constexpr float kMinBarArea         = 20.f;  //  灯条最小轮廓面积(px)
    static constexpr float kMinBarAspect       = 1.5f;  //  灯条长宽比下限(斜视板投影后可降到 ~1.8)
    static constexpr float kMinPairDistRatio   = 1.0f;  //  约束3中心距下限系数(原为 1.2)
    static constexpr float kPairConfThresh     = 0.5f;  //  数字网络确认配对的置信度下限
    static constexpr float kMaxPairRatioSmall  = 2.7f;  //  小板距比上限: 135/55≈2.45 + 10% 测量余量
};
