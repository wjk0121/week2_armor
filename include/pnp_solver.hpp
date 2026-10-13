#pragma once

//  位置解算: 由 4 个角点解算装甲板中心在相机坐标系下的位置

#include "camera.hpp"
#include "types.hpp"

#include <vector>

class PnpSolver
{
public:
    explicit PnpSolver(const CameraParams& params);

    //  输入 armor.corners(像素坐标, 契约序), 输出位置(mm)与重投影自检
    //  畸变不需要手动 undistort, solvePnP 传入畸变系数后内部处理
    PoseResult solve(const Armor& armor) const;

private:
    CameraParams params_;
    //  装甲板坐标系下的 3D 特征点(z=0 平面), 顺序: 左上、右上、右下、左下
    std::vector<cv::Point3f> small_obj_, big_obj_;
};
