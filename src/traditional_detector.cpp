#include "traditional_detector.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

TraditionalDetector::TraditionalDetector(const CameraParams& params,
                                         const NumberClassifier& clf)
    : params_(params), clf_(clf) {}

//  取旋转矩形的两个长轴端点(灯条两端)
static std::pair<cv::Point2f, cv::Point2f> barEndpoints(const cv::RotatedRect& r)
{
    cv::Point2f pts[4];
    r.points(pts);
    const cv::Point2f d1 = pts[1] - pts[0], d2 = pts[2] - pts[1];
    cv::Point2f axis = (cv::norm(d1) > cv::norm(d2)) ? d1 : d2;
    axis /= cv::norm(axis);
    const float half = std::max(r.size.width, r.size.height) / 2.f;
    return {r.center - axis * half, r.center + axis * half};
}

//  灯条与水平线的夹角(度), 用长轴方向计算, 范围[0,90]
static float barAngle(const cv::RotatedRect& r)
{
    const auto [p1, p2] = barEndpoints(r);
    return std::abs(std::atan2(p2.y - p1.y, p2.x - p1.x)) * 180.f / CV_PI;
}

std::vector<cv::RotatedRect> TraditionalDetector::findLightBars(
        const cv::Mat& frame,int color_diff_thresh) const
{
    //  颜色差分: 敌红取 R-B, 敌蓝取 B-R; 灯条在差分图上最亮
    //  (实拍视频为橙红灯条 -> red; 仿真测试图为蓝灯条 -> blue)
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);
    cv::Mat diff;
    if (params_.enemy_color == EnemyColor::RED) cv::subtract(ch[2], ch[0], diff);
    else                                        cv::subtract(ch[0], ch[2], diff);
    cv::threshold(diff, diff, color_diff_thresh, 255, cv::THRESH_BINARY);
    //  轻微膨胀黏合发光边缘的碎斑, 使灯条轮廓连续
    cv::dilate(diff, diff, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> bars;
    for (const auto& c : contours)
    {
        if (cv::contourArea(c) < kMinBarArea) continue;       //  滤噪点
        const cv::RotatedRect r = cv::minAreaRect(c);
        const float ls = std::max(r.size.width, r.size.height);
        const float ss = std::min(r.size.width, r.size.height);
        if (ss < 2.f) continue;                               //  排除细线
        if (ls / ss < kMinBarAspect) continue;                //  必须是细长条
        bars.push_back(r);
    }
    return bars;
}

std::vector<Armor> TraditionalDetector::pairLightBars(
        const std::vector<cv::RotatedRect>& bars, const cv::Mat& frame) const
{
    std::vector<Armor> armors;

    for (size_t i = 0; i < bars.size(); ++i)
    {
        for (size_t j = i + 1; j < bars.size(); ++j)
        {
            const auto [a1, b1] = barEndpoints(bars[i]);
            const auto [a2, b2] = barEndpoints(bars[j]);
            const float len1 = cv::norm(b1 - a1), len2 = cv::norm(b2 - a2);

            //  约束1: 长度接近(透视下同板两灯条像素长度近似)
            const float lr = std::min(len1, len2) / std::max(len1, len2);
            if (lr < 0.4f) continue;
            //  约束2: 夹角接近
            if (std::abs(barAngle(bars[i]) - barAngle(bars[j])) > 15.f) continue;
            //  约束3: 中心距约为 1.0~4 倍灯条长(太近是同一根, 太远不相关)
            const cv::Point2f dc = bars[j].center - bars[i].center;
            const float dist = cv::norm(dc), avg_len = (len1 + len2) / 2.f;
            if (dist < kMinPairDistRatio * avg_len || dist > 4.f * avg_len) continue;
            //  约束4: 连线应近似垂直于灯条方向(两灯条左右排布)
            if (std::abs(dc.dot(cv::Point2f(b1.y - a1.y, a1.x - b1.x)))
                    / (dist * avg_len) < 0.3f) continue;

            //  左/右灯条按中心 m 坐标区分; 端点按 n 坐标分上下
            const bool i_left = bars[i].center.x < bars[j].center.x;
            const auto& le = i_left ? std::make_pair(a1, b1) : std::make_pair(a2, b2);
            const auto& re = i_left ? std::make_pair(a2, b2) : std::make_pair(a1, b1);
            const cv::Point2f lt = le.first.y < le.second.y ? le.first  : le.second;
            const cv::Point2f lb = le.first.y < le.second.y ? le.second : le.first;
            const cv::Point2f rt = re.first.y < re.second.y ? re.first  : re.second;
            const cv::Point2f rb = re.first.y < re.second.y ? re.second : re.first;

            //  数字区透视矫正: 契约序角点 -> 32x28 竖直版式
            //  灯条纵向映射到两侧 y∈[7,19] 的短段, 数字居中(已实测验证的几何)
            const std::vector<cv::Point2f> src = {lt, rt, rb, lb};
            const std::vector<cv::Point2f> dst = {{0, 7}, {31, 7}, {31, 19}, {0, 19}};
            const cv::Mat M = cv::getPerspectiveTransform(src, dst);
            cv::Mat warp;
            cv::warpPerspective(frame, warp, M, {32, 28});

            //  数字网络辅助配对: negative 或低置信度 -> 否决这对灯条
            //  TTA: 远距离/大斜视时灯条端点估计有 ±3px 抖动, 且数字区在 warp 后
            //  相对 ROI 存在 ~2px 的系统偏移; mlp 对该偏移敏感(同一张数字, 平移
            //  相位不同判定可完全相反), 故在 [-3,3]^2 的整数格点上遍历全部相位,
            //  取非 negative 且置信度最高者 —— 单点探针(如仅 (3,-2))会漏掉正确判定
            int best_idx = -1;
            float best_conf = 0.f;
            for (int dy = -3; dy <= 3; ++dy)
            {
                for (int dx = -3; dx <= 3; ++dx)
                {
                    cv::Mat probe = warp.clone();
                    if (dx != 0 || dy != 0)
                    {
                        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
                        cv::warpAffine(warp, probe, M, warp.size(),
                                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                    }
                    const auto [idx, conf] = clf_.classify(probe);
                    if (idx < 0) continue;
                    if (idx == static_cast<int>(clf_.numLabels()) - 1) continue;   //  negative
                    if (conf > best_conf)
                        best_idx = idx, best_conf = conf;
                }
            }
            if (best_idx < 0 || best_conf < kPairConfThresh) continue;

            Armor armor;
            armor.corners = {lt, rt, rb, lb};     //  契约: 左上、右上、右下、左下
            armor.number = best_idx;
            armor.confidence = best_conf;
            armor.label = clf_.label(best_idx);
            //  尺寸启发: 1号(英雄)、outpost(前哨站)、base(基地)为大装甲板
            armor.size = (armor.label == "1" || armor.label == "outpost" ||
                          armor.label == "base")
                         ? ArmorSize::BIG : ArmorSize::SMALL;
            //  板型一致的距比上限: 小板(135x55)的 中心距/杆长 最大为正面
            //  135/55≈2.45, 斜视只按 cos 压缩不会更大; 取 10% 像素测量余量 2.7。
            //  超过 2.7 的对物理上不可能是小板 —— 属跨板误配对或背景结构,
            //  其数字判定必为误读; 大板(225/55≈4.09)沿用配对阶段的 4.0 上限
            if (armor.size != ArmorSize::BIG && dist > kMaxPairRatioSmall * avg_len)
                continue;
            armors.push_back(armor);
        }
    }
    return armors;
}

std::vector<Armor> TraditionalDetector::detect(const cv::Mat& frame) const
{
    if (frame.empty()) return {};
    std::vector<cv::RotatedRect> bars = findLightBars(frame, kColorDiffThresh);
    for (const auto& weak : findLightBars(frame, kColorDiffThreshLow))
    {
        //  弱召回的灯条若与已有灯条重叠, 则忽略(避免重复配对)
        bool dup = false;
        for (const auto& b : bars)
        {
            const float r=0.5f*std::max(b.size.width, b.size.height);
            if(std::abs(weak.center.x - b.center.x) < r && std::abs(weak.center.y - b.center.y) < r)
            {
                dup = true;
                break;
            }
            
        }
        if (!dup) bars.push_back(weak);
    }
    return pairLightBars(bars, frame);
}
