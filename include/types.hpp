#pragma once

//  全工程共享的数据结构与类别表

#include <opencv2/core.hpp>
#include <string>
#include <vector>

//  装甲板尺寸类别: 决定 PnP 使用哪组 3D 特征点
enum class ArmorSize { SMALL, BIG };

//  一块装甲板的检测结果
//  corners 为像素坐标 (m,n), 顺序固定: 左上、右上、右下、左下
//  该顺序与 PnpSolver 的 3D 特征点一一对应, 是全工程的契约
struct Armor
{
    std::vector<cv::Point2f> corners;
    int number = -1;                        //  类别索引(trad: 数字标签; nn: 兵种标签)
    float confidence = 0.f;                 //  检测/分类置信度
    ArmorSize size = ArmorSize::SMALL;
    std::string label;                      //  显示用类别名

    //  角点均值作为图像上的板中心(仅用于可视化与配对)
    cv::Point2f center() const
    {
        cv::Point2f c(0, 0);
        for (const auto& p : corners) c += p;
        return c * 0.25f;
    }
};

//  一次 PnP 解算的结果
struct PoseResult
{
    bool ok = false;            //  解算是否成功
    cv::Mat rvec, tvec;         //  tvec 单位 mm, 相机坐标系, 位置参考点为装甲板中心
    double rms_px = -1.0;       //  重投影均方根误差(像素), -1 表示无效
};

//  一条真值记录(truth/imgN.md 1.1 节, 相机光学系, 单位 m)
//  只保留对比真正用到的字段: 位置用于 Δ 误差, 外法线用于姿态角误差。
//  md 中的 roll/pitch/yaw 经穷举 48 种欧拉约定验证, 与其自身给出的外法线矛盾
//  (最小平均角差仍达 77 度), 不能当姿态真值用, 故不解析、不保存。
struct TruthArmor
{
    int id = 0;                 //  armor_1 ~ armor_4
    bool front = false;         //  正面可见(只有正面板参与误差对比)
    cv::Point3d t;              //  (tx, ty, tz), 单位 m
    cv::Vec3d normal;           //  外法线(相机系, 单位向量, 面向相机为正方向)
};

//  数字识别标签(mlp.onnx, 与 models/label.txt 一致, 最后一类为 negative)
inline const std::vector<std::string>& numberLabels()
{
    static const std::vector<std::string> v =
        {"1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"};
    return v;
}

//  装甲板检测标签(Infantry-v8n, 9 类兵种, 与模型通道 4~12 顺序一致)
inline const std::vector<std::string>& armorLabels()
{
    static const std::vector<std::string> v =
        {"guard", "hero", "engineer", "infantry3", "infantry4",
         "infantry5", "outpost", "base_bottom", "base_top"};
    return v;
}
