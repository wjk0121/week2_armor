#pragma once

//  可视化: 把检测与解算结果画到帧上
//  颜色约定: 绿框 = 装甲板四边形, 黄点/黄字 = 角点序号与信息文字
//  角点序号 0~3 与工程契约一致: 0=左上 1=右上 2=右下 3=左下

#include "types.hpp"

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace Visualizer
{
//  在 frame 上绘制全部结果(直接修改 frame)
void drawResults(cv::Mat& frame,
                 const std::vector<std::pair<Armor, PoseResult>>& results);
}   // namespace Visualizer
