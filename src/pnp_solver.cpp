#include "pnp_solver.hpp"

#include <cmath>
#include <opencv2/calib3d.hpp>     //  solvePnP / projectPoints 在 calib3d 模块

PnpSolver::PnpSolver(const CameraParams& params) : params_(params)
{
    //  以板中心为原点、z=0 平面构造 3D 角点, 顺序与 Armor::corners 契约一致
    //  先转 float 再构造 Point3f: 避免 double 到 float 的 narrowing conversion 编译报错
    const float sw = static_cast<float>(params.small_width  / 2.0);
    const float sh = static_cast<float>(params.small_height / 2.0);
    small_obj_ = {cv::Point3f(-sw, -sh, 0), cv::Point3f(sw, -sh, 0),
                  cv::Point3f(sw, sh, 0),   cv::Point3f(-sw, sh, 0)};

    const float bw = static_cast<float>(params.big_width  / 2.0);
    const float bh = static_cast<float>(params.big_height / 2.0);
    big_obj_   = {cv::Point3f(-bw, -bh, 0), cv::Point3f(bw, -bh, 0),
                  cv::Point3f(bw, bh, 0),   cv::Point3f(-bw, bh, 0)};
}

PoseResult PnpSolver::solve(const Armor& armor) const
{
    PoseResult res;
    if (armor.corners.size() != 4) return res;      //  异常: 角点不全

    const auto& obj = (armor.size == ArmorSize::BIG) ? big_obj_ : small_obj_;

    try
    {
        //  IPPE 专为平面四点目标设计, 比 ITERATIVE 更稳且无需初值
        const bool ok = cv::solvePnP(obj, armor.corners,
                                     params_.camera_matrix, params_.dist_coeffs,
                                     res.rvec, res.tvec, false, cv::SOLVEPNP_IPPE);
        if (!ok) return res;

        //  平面目标存在镜像歧义, z<=0 说明解到相机背后, 直接判失败
        if (res.tvec.at<double>(2) <= 0) return res;

        //  用同一内外参把 3D 点投回像素平面, 算重投影 RMS 作为质量门
        std::vector<cv::Point2f> reproj;
        cv::projectPoints(obj, res.rvec, res.tvec,
                          params_.camera_matrix, params_.dist_coeffs, reproj);
        double se = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const cv::Point2f d = reproj[i] - armor.corners[i];
            se += d.x * d.x + d.y * d.y;
        }
        res.rms_px = std::sqrt(se / 4.0);
        res.ok = true;
    }
    catch (const cv::Exception&)
    {
        res.ok = false;     //  异常: solvePnP 数值失败, 不让程序崩
    }
    return res;
}
