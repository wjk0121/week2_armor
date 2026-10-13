//  主程序: CLI 解析 -> 相机参数 -> 检测(双方案) -> PnP 解算 -> 真值对比 -> 可视化
//  用法速查: ./rm_vision --method trad --source data/test.avi --output results/video.mp4

#include "camera.hpp"
#include "nn_detector.hpp"
#include "number_classifier.hpp"
#include "pnp_solver.hpp"
#include "traditional_detector.hpp"
#include "truth_loader.hpp"
#include "types.hpp"
#include "visualizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

namespace
{
//  ------------------------- 命令行 -------------------------

struct Options
{
    std::string method = "trad";            //  trad | nn
    std::string source;                     //  图片或视频路径(必填)
    std::string truth;                      //  真值 md(图片模式可选)
    std::string camera = "camera.yaml";     //  相机参数文件
    std::string models = "models";          //  模型目录
    std::string output;                     //  输出图片/视频(可选)
    std::string enemy;                      //  red / blue 覆盖 yaml(可选)
    bool   show = false;                    //  窗口显示
    double step = 1.0;                      //  视频抽帧步长(逐帧 = 1)
};

void printUsage()
{
    std::printf(
        "用法: rm_vision <图片|视频> [选项]\n"
        "      (输入路径也可用 --source 指定)\n"
        "\n"
        "  --method trad|nn   检测方案: trad=传统 / nn=神经网络, 默认 trad\n"
        "  --source <路径>    输入图片或视频(必填)\n"
        "  --truth <路径>     真值 md(图片模式可选, 用于误差对比)\n"
        "  --camera <路径>    相机参数 yaml, 默认 camera.yaml\n"
        "  --models <目录>    模型目录, 默认 models\n"
        "  --output <路径>    输出图片/视频路径(可选)\n"
        "  --enemy red|blue   覆盖敌方颜色(可选, 仿真图为 blue)\n"
        "  --show             窗口显示, Esc 退出\n"
        "  --step <n>         视频抽帧步长, 默认 1(逐帧)\n");
}

bool parseArgs(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("参数 ") + name + " 缺少取值");
            return argv[++i];
        };

        if      (a == "--method") o.method = next("--method");
        else if (a == "--source") o.source = next("--source");
        else if (a == "--truth")  o.truth  = next("--truth");
        else if (a == "--camera") o.camera = next("--camera");
        else if (a == "--models") o.models = next("--models");
        else if (a == "--output") o.output = next("--output");
        else if (a == "--enemy")  o.enemy  = next("--enemy");
        else if (a == "--step")   o.step   = std::stod(next("--step"));
        else if (a == "--show")   o.show   = true;
        else if (a == "--help" || a == "-h") { printUsage(); return false; }
        else if (a.rfind("--", 0) == 0) throw std::runtime_error("未知参数: " + a);
        else if (o.source.empty()) o.source = a;    //  位置参数兼容: 输入路径
        else throw std::runtime_error("多余的参数: " + a);
    }
    if (o.source.empty())
    {
        std::printf("错误: 必须指定 --source\n\n");
        printUsage();
        return false;
    }
    return true;
}

//  按扩展名判断输入是图片还是视频
bool isImageFile(const std::string& path)
{
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    for (char& c : ext)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

//  ------------------------- 检测器装配 -------------------------

//  按 --method 只装配其中一套: 传统(clf+trad) 或 神经(nn)
struct DetectorSet
{
    std::unique_ptr<NumberClassifier> clf;
    std::unique_ptr<TraditionalDetector> trad;
    std::unique_ptr<NnDetector> nn;

    std::vector<Armor> detect(const cv::Mat& frame) const
    {
        return trad ? trad->detect(frame) : nn->detect(frame);
    }
};

DetectorSet buildDetectors(const Options& opt, const CameraParams& cam)
{
    DetectorSet det;
    if (opt.method == "trad")
    {
        det.clf  = std::make_unique<NumberClassifier>(opt.models + "/mlp.onnx");
        det.trad = std::make_unique<TraditionalDetector>(cam, *det.clf);
    }
    else if (opt.method == "nn")
    {
        det.nn = std::make_unique<NnDetector>(opt.models + "/Infantry-v8n.onnx");
    }
    else
    {
        throw std::runtime_error("未知方案 --method " + opt.method + " (应为 trad / nn)");
    }
    return det;
}

//  ------------------------- 真值对比 -------------------------

//  PnP 外法线(-R 的第三列)与真值法线的夹角(度), 失败返回 -1
//  符号依据: 板坐标系 +z 轴背离相机, 取反后为"数字面朝外"的法线;
//  已用真值角点闭环实测验证(图1 armor_1 夹角 0.004 度)
double normalAngleError(const cv::Mat& rvec, const cv::Vec3d& gt)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    const cv::Vec3d n(-R.at<double>(0, 2), -R.at<double>(1, 2), -R.at<double>(2, 2));
    const double nn = cv::norm(n), gn = cv::norm(gt);
    if (nn < 1e-9 || gn < 1e-9) return -1.0;
    double c = n.dot(gt) / (nn * gn);
    c = std::min(1.0, std::max(-1.0, c));
    return std::acos(c) * 180.0 / CV_PI;
}

//  打印解算结果与真值的逐板误差(仅正面板; 单位 mm / 度)
void reportAgainstTruth(const std::vector<std::pair<Armor, PoseResult>>& results,
                        const std::vector<TruthArmor>& truths)
{
    std::printf("\n== 与真值对比 (相机光学系) ==\n");

    //  先枚举全部 (真值板, 检测) 候选(3D 距离 < 1 m), 再按距离升序做全局贪心
    //  一对一指派: 两正面板真值相距仅 ~0.30 m, 单个检测常同时落在两块板的
    //  1 m 圈内, 必须"最近者优先 + 每个检测只用一次", 否则产生虚假误差行
    struct Cand { double dist; size_t truth_idx; int det_idx; };
    std::vector<Cand> cands;
    for (size_t ti = 0; ti < truths.size(); ++ti)
    {
        if (!truths[ti].front) continue;        //  背面板图像中不可见, 不参与对比
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& p = results[i].second;
            if (!p.ok) continue;
            const cv::Vec3d est(p.tvec.at<double>(0) / 1000.0,
                                p.tvec.at<double>(1) / 1000.0,
                                p.tvec.at<double>(2) / 1000.0);
            const double d = cv::norm(est - cv::Vec3d(truths[ti].t.x,
                                                      truths[ti].t.y,
                                                      truths[ti].t.z));
            if (d < 1.0) cands.push_back({d, ti, static_cast<int>(i)});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

    std::vector<bool> matched(results.size(), false);   //  是否被匹配(用于虚警统计)
    std::vector<bool> used(results.size(), false);      //  一对一: 每个检测只用一次
    std::vector<int>  assign(truths.size(), -1);        //  每块真值板最终匹配的检测
    for (const auto& c : cands)
    {
        if (assign[c.truth_idx] >= 0 || used[c.det_idx]) continue;
        assign[c.truth_idx] = c.det_idx;
        used[c.det_idx] = true;
        matched[c.det_idx] = true;
    }

    for (size_t ti = 0; ti < truths.size(); ++ti)
    {
        const auto& t = truths[ti];
        if (!t.front) continue;

        const std::string name = "armor_" + std::to_string(t.id);
        const int best = assign[ti];
        if (best < 0)
        {
            std::printf("%-9s 未检出\n", name.c_str());
            continue;
        }

        const auto& p = results[best].second;
        const cv::Vec3d est(p.tvec.at<double>(0) / 1000.0,
                            p.tvec.at<double>(1) / 1000.0,
                            p.tvec.at<double>(2) / 1000.0);
        const cv::Vec3d gt(t.t.x, t.t.y, t.t.z);
        const cv::Vec3d d = (est - gt) * 1000.0;            //  m -> mm

        std::printf("%-9s 匹配检测 #%d (%s)\n", name.c_str(), best,
                    results[best].first.label.c_str());
        std::printf("          dx=%+.1f mm  dy=%+.1f mm  dz=%+.1f mm  |3D|=%.1f mm\n",
                    d[0], d[1], d[2], cv::norm(d));
        std::printf("          法线角误差 %.2f 度 (真值法线 %+.3f,%+.3f,%+.3f)\n",
                    normalAngleError(p.rvec, t.normal),
                    t.normal[0], t.normal[1], t.normal[2]);
    }

    //  统计虚警: 解算成功但没有匹配到任何真值板的检测
    int extra = 0;
    for (size_t i = 0; i < results.size(); ++i)
        if (!matched[i] && results[i].second.ok) ++extra;
    if (extra > 0)
        std::printf("提示: %d 个检测未匹配到真值(可能为虚警或重复检测)\n", extra);
}

//  ------------------------- 图片模式 -------------------------

int processImage(const Options& opt, const cv::Mat& frame,
                 const DetectorSet& det, const PnpSolver& solver)
{
    const std::vector<Armor> armors = det.detect(frame);

    std::vector<std::pair<Armor, PoseResult>> results;
    results.reserve(armors.size());
    for (const auto& a : armors) results.emplace_back(a, solver.solve(a));

    std::printf("\n== 检测与解算: %zu 块装甲板 ==\n", results.size());
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& [a, p] = results[i];
        if (!p.ok)
        {
            std::printf("#%zu %-9s conf=%.2f [解算失败, 已跳过]\n",
                        i, a.label.c_str(), a.confidence);
            continue;
        }
        std::printf("#%zu %-9s conf=%.2f t=(%+.1f, %+.1f, %+.1f) mm |t|=%.3f m rms=%.2f px\n",
                    i, a.label.c_str(), a.confidence,
                    p.tvec.at<double>(0), p.tvec.at<double>(1), p.tvec.at<double>(2),
                    cv::norm(p.tvec) / 1000.0, p.rms_px);
    }

    if (!opt.truth.empty())
    {
        const std::vector<TruthArmor> truths = loadTruth(opt.truth);
        if (truths.empty())
            std::printf("警告: 真值文件未解析到数据(%s)\n", opt.truth.c_str());
        else
            reportAgainstTruth(results, truths);
    }

    cv::Mat canvas = frame.clone();
    Visualizer::drawResults(canvas, results);

    if (!opt.output.empty())
    {
        if (cv::imwrite(opt.output, canvas))
            std::printf("结果图片已保存: %s\n", opt.output.c_str());
        else
            std::printf("警告: 结果图片保存失败 %s(目录不存在?)\n", opt.output.c_str());
    }
    if (opt.show)
    {
        cv::imshow("rm_vision", canvas);
        cv::waitKey(0);                     //  图片模式按任意键关闭
    }
    return 0;
}

//  ------------------------- 视频模式 -------------------------

//  cap 由调用方打开并已读过首帧(用于取分辨率), 这里从当前读指针继续
int processVideo(const Options& opt, const DetectorSet& det, const PnpSolver& solver,
                 cv::VideoCapture& cap)
{
    const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    //  可选输出视频(mp4v 编码; 环境缺编码器时跳过并提示)
    cv::VideoWriter writer;
    if (!opt.output.empty())
    {
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 1.0) fps = 30.0;
        const double out_fps = std::max(fps / std::max(1.0, opt.step), 1.0);
        const cv::Size size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        writer.open(opt.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    out_fps, size);
        if (!writer.isOpened())
            std::printf("警告: 无法创建输出视频 %s, 仅做处理统计\n", opt.output.c_str());
    }

    if (opt.show) cv::namedWindow("rm_vision", cv::WINDOW_NORMAL);

    cv::Mat frame;
    int idx = -1, processed = 0, frames_with_det = 0, total_armors = 0;
    double ms_sum = 0.0;

    while (cap.read(frame))
    {
        ++idx;
        if (idx % static_cast<int>(std::max(1.0, opt.step)) != 0) continue;

        const int64 tick0 = cv::getTickCount();

        //  一帧完整流程: 检测 -> 逐块解算 -> 绘制
        const std::vector<Armor> armors = det.detect(frame);
        std::vector<std::pair<Armor, PoseResult>> results;
        results.reserve(armors.size());
        int solved = 0;
        double min_dist = -1.0;
        for (const auto& a : armors)
        {
            results.emplace_back(a, solver.solve(a));
            const auto& pose = results.back().second;
            if (pose.ok)
            {
                ++solved;
                const double d = cv::norm(pose.tvec);
                if (min_dist < 0 || d < min_dist) min_dist = d;
            }
        }
        Visualizer::drawResults(frame, results);

        ++processed;
        total_armors += static_cast<int>(armors.size());
        if (!armors.empty()) ++frames_with_det;

        const double ms = (cv::getTickCount() - tick0) * 1000.0
                          / cv::getTickFrequency();
        ms_sum += ms;

        if (processed % 30 == 0)
        {
            std::printf("帧 %d/%d: 检出 %zu 块", idx, total, armors.size());
            if (solved > 0)
                std::printf("(解算 %d, 最近 %.2f m) ", solved, min_dist / 1000.0);
            std::printf("| %.1f ms\n", ms);
        }

        if (writer.isOpened()) writer.write(frame);
        if (opt.show)
        {
            cv::imshow("rm_vision", frame);
            if (cv::waitKey(1) == 27) break;        //  Esc 提前退出
        }
    }

    std::printf("\n完成: 处理 %d 帧, 有检出 %d 帧(%.1f%%), 平均检出 %.2f 块/帧\n",
                processed, frames_with_det,
                processed > 0 ? 100.0 * frames_with_det / processed : 0.0,
                processed > 0 ? static_cast<double>(total_armors) / processed : 0.0);
    std::printf("平均耗时 %.1f ms/帧 (%.1f FPS)\n",
                processed > 0 ? ms_sum / processed : 0.0,
                ms_sum > 0.0 ? processed * 1000.0 / ms_sum : 0.0);
    if (writer.isOpened())
        std::printf("输出视频已保存: %s\n", opt.output.c_str());
    return 0;
}
}   // namespace

//  ------------------------- 入口 -------------------------

int main(int argc, char** argv)
{
    try
    {
        Options opt;
        if (!parseArgs(argc, argv, opt)) return 1;

        //  1) 相机参数
        CameraParams cam;
        if (!loadCameraParams(opt.camera, cam))
        {
            std::printf("错误: 无法读取相机参数 %s\n", opt.camera.c_str());
            return 1;
        }
        if (!opt.enemy.empty() && !parseEnemyColor(opt.enemy, cam.enemy_color))
        {
            std::printf("错误: --enemy 只接受 red / blue, 收到 %s\n", opt.enemy.c_str());
            return 1;
        }

        //  2) 判输入类型并读首帧(用于取分辨率); 视频只打开一次, 之后沿用同一句柄
        const bool image_mode = isImageFile(opt.source);
        cv::Mat first;
        cv::VideoCapture cap;
        if (image_mode)
        {
            first = cv::imread(opt.source, cv::IMREAD_COLOR);
        }
        else
        {
            cap.open(opt.source);
            if (cap.isOpened()) cap.read(first);
            //  本次 read 只为取分辨率, 处理阶段要从第 0 帧开始, 否则首帧被丢弃
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
        if (first.empty())
        {
            std::printf("错误: 无法读取输入 %s\n", opt.source.c_str());
            return 1;
        }

        //  3) 内参按实际分辨率适配(视频/仿真各用对应 yaml; 同一相机变分辨率时自动缩放)
        bool aspect_ok = true;
        const CameraParams cam_s = cam.scaledTo(first.cols, first.rows, &aspect_ok);
        if (!aspect_ok)
            std::printf("警告: 输入 %dx%d 与标定 %dx%d 宽高比不一致, 内参不做缩放"
                        "(请为该分辨率单独标定)\n",
                        first.cols, first.rows, cam.image_width, cam.image_height);
        std::printf("输入 %dx%d | 标定 %dx%d | 内参缩放 %.4f | 颜色 %s | 方案 %s\n",
                    first.cols, first.rows, cam.image_width, cam.image_height,
                    static_cast<double>(first.cols) / cam.image_width,
                    enemyColorName(cam_s.enemy_color), opt.method.c_str());

        //  4) 输出目录不存在时自动创建(否则 imwrite/VideoWriter 会静默失败)
        if (!opt.output.empty())
        {
            const auto parent = std::filesystem::path(opt.output).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
        }

        //  5) 装配检测器与解算器
        const DetectorSet det = buildDetectors(opt, cam_s);
        const PnpSolver solver(cam_s);

        return image_mode ? processImage(opt, first, det, solver)
                          : processVideo(opt, det, solver, cap);
    }
    catch (const std::exception& e)
    {
        std::printf("错误: %s\n", e.what());
        return 1;
    }
}
