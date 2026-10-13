#include "truth_loader.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace
{
//  去掉首尾空白, 供表格单元格使用
std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

//  解析 "(x, y, z)" 形式的法线字段: 先把括号与逗号换成空格再流式读取
bool parseVec3(const std::string& s, cv::Vec3d& out)
{
    std::string t = s;
    for (char& c : t)
        if (c == '(' || c == ')' || c == ',') c = ' ';
    std::istringstream ss(t);
    return static_cast<bool>(ss >> out[0] >> out[1] >> out[2]);
}
}   // namespace

std::vector<TruthArmor> loadTruth(const std::string& path)
{
    std::vector<TruthArmor> truths;

    std::ifstream in(path);
    if (!in.is_open()) return truths;

    std::string line;
    while (std::getline(in, line))
    {
        //  只处理 1.1 节的数据行: 行以 "| armor_" 开头
        //  (1.1 节 9 列, 竖线切分后含首空段为 10 段(行尾带空白时 11 段);
        //   1.2 节世界系表格 7 列只有 8~9 段, 不足下方要求的 10 段, 天然跳过)
        if (line.rfind("| armor_", 0) != 0) continue;

        //  按竖线切分: cells = ["", "armor_1", "正面", tx ... normal] (尾随空段可有可无)
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '|')) cells.push_back(trim(cell));
        if (cells.size() < 10) continue;

        try
        {
            TruthArmor ta;
            ta.id    = std::stoi(cells[1].substr(6));         //  "armor_1" -> 1
            ta.front = (cells[2] == "正面");
            ta.t     = {std::stod(cells[3]), std::stod(cells[4]), std::stod(cells[5])};
            //  cells[6..8] 是 roll/pitch/yaw: 与真值外法线矛盾, 不解析(理由见 types.hpp)
            if (!parseVec3(cells[9], ta.normal)) continue;
            truths.push_back(ta);
        }
        catch (const std::exception&)
        {
            continue;       //  个别行格式异常时跳过该行, 不中断整体解析
        }
    }
    return truths;
}
