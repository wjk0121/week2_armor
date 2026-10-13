#include "camera.hpp"

#include <algorithm>
#include <cctype>

//  解析敌方颜色字符串(大小写不敏感)
bool parseEnemyColor(const std::string& text, EnemyColor& out)
{
    std::string s = text;
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "red")  { out = EnemyColor::RED;  return true; }
    if (s == "blue") { out = EnemyColor::BLUE; return true; }
    return false;
}

const char* enemyColorName(EnemyColor c)
{
    return c == EnemyColor::RED ? "red" : "blue";
}

//  用 OpenCV读yaml, 缺字段时保留默认值
bool loadCameraParams(const std::string& path, CameraParams& p)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    fs["camera_matrix"] >> p.camera_matrix;
    fs["dist_coeffs"]   >> p.dist_coeffs;
    if (!fs["image_width"].empty())  fs["image_width"]  >> p.image_width;
    if (!fs["image_height"].empty()) fs["image_height"] >> p.image_height;
    if (!fs["small_width"].empty())  fs["small_width"]  >> p.small_width;
    if (!fs["small_height"].empty()) fs["small_height"] >> p.small_height;
    if (!fs["big_width"].empty())    fs["big_width"]    >> p.big_width;
    if (!fs["big_height"].empty())   fs["big_height"]   >> p.big_height;

    //  yaml 只认 red / blue; 写错时保留默认值, 由调用方日志暴露
    if (!fs["enemy_color"].empty())
    {
        std::string color;
        fs["enemy_color"] >> color;
        parseEnemyColor(color, p.enemy_color);
    }

    return !p.camera_matrix.empty() && !p.dist_coeffs.empty();
}

//分辨率变化的原因:图像等比缩放后, 像素坐标整体按 s 缩放,
//内参矩阵的 fx/fy/cx/cy必须一起乘s才能保持几何一致。
//只有宽高比与标定一致时单一比例才成立(旧实现只用宽度算 s、完全忽略 height,
//遇到非等比分辨率会静默给出错误内参), 因此这里显式校验宽高比。
CameraParams CameraParams::scaledTo(int width, int height, bool* aspect_ok) const
{
    CameraParams out = *this;
    if (aspect_ok) *aspect_ok = true;

    if (image_width <= 0 || image_height <= 0 || width <= 0 || height <= 0) return out;
    if (width == image_width && height == image_height) return out;     //  尺寸相同, 无需缩放

    const double sx = static_cast<double>(width) / image_width;
    const double sy = static_cast<double>(height) / image_height;
    //  允许 1% 的舍入差异(如 1440x1080 -> 960x720 是精确的 2/3)
    if (std::abs(sx - sy) > 0.01 * std::max(sx, sy))
    {
        if (aspect_ok) *aspect_ok = false;
        return out;                     //  宽高比不符: 不做错误的等比缩放
    }

    cv::Mat K = camera_matrix.clone();
    K.at<double>(0, 0) *= sx;           //  fx
    K.at<double>(1, 1) *= sy;           //  fy
    K.at<double>(0, 2) *= sx;           //  cx
    K.at<double>(1, 2) *= sy;           //  cy
    out.camera_matrix = K;
    return out;
}
