#pragma once

//  相机内参、畸变与装甲板尺寸的加载(camera.yaml)

#include <opencv2/core.hpp>

#include <string>

//  敌方灯条颜色。用强类型枚举而非字符串: 字符串一旦写成 "Blue"/"蓝色" 等
//  非预期取值, 与 "red" 的比较会静默落入 else 分支, 直接取错颜色通道
//  (表现为传统方案全丢检, 已实测)。
enum class EnemyColor { RED, BLUE };

struct CameraParams
{
    cv::Mat camera_matrix;                  //3x3 内参矩阵(对应标定图像尺寸)
    cv::Mat dist_coeffs;                    //1x5 畸变系数 [k1 k2 p1 p2 k3]
    int    image_width  = 1440;             //标定图像宽(与内参一一对应)
    int    image_height = 1080;             //标定图像高
    double small_width  = 135.0;            //小装甲板两灯条中心距(mm)
    double small_height = 55.0;             //小装甲板灯条长度(mm)
    double big_width    = 225.0;            //大装甲板(mm)
    double big_height   = 55.0;
    EnemyColor enemy_color = EnemyColor::BLUE;   //默认按仿真图(蓝灯条)

    //  按实际图像尺寸等比缩放内参(畸变系数不变), 返回副本
    //  标定尺寸 1440x1080 而输入图为 960x720 时必须调用, 否则解算结果整体偏移(已实测)
    //  宽高比与标定不一致时单比例缩放不成立, 此时不改内参并把 aspect_ok 置 false
    CameraParams scaledTo(int width, int height, bool* aspect_ok = nullptr) const;
};

//  解析敌方颜色字符串(大小写不敏感): 只接受 red / blue, 其它返回 false
bool parseEnemyColor(const std::string& text, EnemyColor& out);

//  枚举 -> 字符串, 仅用于日志
const char* enemyColorName(EnemyColor c);

//  读取成功返回 true; 文件打不开或矩阵为空返回 false
bool loadCameraParams(const std::string& path, CameraParams& params);
