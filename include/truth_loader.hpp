#pragma once

//  真值解析: 读取 truth/imgN.md 中「1.1 相机光学系」小节的数据表
//  表格格式: | armor_1 | 正面 | tx | ty | tz | roll_x | pitch_y | yaw_z | 外法线 |
//  位置单位 m, 角度单位度; 只有「正面」板在图像中可见, 参与误差对比

#include "types.hpp"

#include <string>
#include <vector>

//  解析失败或文件不存在时返回空 vector(调用方按"无真值"处理, 不崩溃)
std::vector<TruthArmor> loadTruth(const std::string& path);
