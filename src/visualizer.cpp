#include "visualizer.hpp"

#include <algorithm>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace
{
//  画一块装甲板: 四边形 + 四个角点序号
void drawOne(cv::Mat& frame, const Armor& armor)
{
    if (armor.corners.size() != 4) return;

    std::vector<cv::Point> pts;
    pts.reserve(4);
    for (const auto& p : armor.corners)
        pts.emplace_back(cvRound(p.x), cvRound(p.y));

    const cv::Scalar green(0, 255, 0), yellow(0, 255, 255);

    //  闭合四边形, 直观看到四点契约的连线顺序
    cv::polylines(frame, pts, true, green, 2);

    //  角点序号
    for (int i = 0; i < 4; ++i)
    {
        cv::circle(frame, pts[i], 3, yellow, -1);
        char idx[8];
        std::snprintf(idx, sizeof(idx), "%d", i);
        cv::putText(frame, idx, pts[i] + cv::Point(4, -4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1, cv::LINE_AA);
    }

    //  信息文字: 类别 + 置信度(+ 距离 + 重投影误差)
    char text[160];
    if (armor.confidence > 0.f)
        std::snprintf(text, sizeof(text), "%s %.2f", armor.label.c_str(), armor.confidence);
    else
        std::snprintf(text, sizeof(text), "%s", armor.label.c_str());

    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                        0.6, 2, &baseline);
    const cv::Rect box = cv::boundingRect(pts);
    //  文字默认放框上方; 顶到画面上边时改放框内, 避免出界
    cv::Point org(box.x, box.y - 8);
    if (org.y < ts.height + 4) org.y = box.y + ts.height + 6;
    org.x = std::min(org.x, frame.cols - ts.width - 4);
    org.x = std::max(org.x, 4);

    cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);       //  黑描边, 浅色背景也清晰
    cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                yellow, 2, cv::LINE_AA);
}
}   // namespace

namespace Visualizer
{
void drawResults(cv::Mat& frame,
                 const std::vector<std::pair<Armor, PoseResult>>& results)
{
    for (const auto& [armor, pose] : results)
    {
        drawOne(frame, armor);

        //  解算成功时在四边形下方追加一行位姿信息
        if (!pose.ok || armor.corners.size() != 4) continue;

        const cv::Point2f c = armor.center();
        const double dist_m = cv::norm(pose.tvec) / 1000.0;     //  mm -> m
        char line[192];
        //  小数位与"检测与解算"日志保持一致, 避免同一位姿在日志和画面上显示成两个数
        std::snprintf(line, sizeof(line), "t=(%.1f,%.1f,%.1f)mm |t|=%.2fm rms=%.2fpx",
                      pose.tvec.at<double>(0), pose.tvec.at<double>(1),
                      pose.tvec.at<double>(2), dist_m, pose.rms_px);

        //  先量文字宽度再定位: 旧实现用固定的 -150/320 回退, 长文本会越出画面
        int baseline_txt = 0;
        const cv::Size ts2 = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1,
                                             &baseline_txt);
        cv::Point org(cvRound(c.x) - ts2.width / 2, cvRound(c.y) + 26);
        org.x = std::min(std::max(org.x, 4), std::max(4, frame.cols - ts2.width - 4));
        org.y = std::min(std::max(org.y, 20), frame.rows - 8);

        cv::putText(frame, line, org, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame, line, org, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}
}   // namespace Visualizer
