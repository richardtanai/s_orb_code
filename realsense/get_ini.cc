#include "System.h"
#include <librealsense2/rs_advanced_mode.hpp>
#include "cv-helpers.hpp"
#include <librealsense2/rs.hpp>

using namespace std;

int main()
{
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    rs2::pipeline_profile selection = pipe.start(cfg);
    auto depth_stream = selection.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto resolution = std::make_pair(depth_stream.width(), depth_stream.height());
    auto i = depth_stream.get_intrinsics();
    auto principal_point = std::make_pair(i.ppx, i.ppy);
    auto focal_length = std::make_pair(i.fx, i.fy);
    rs2_distortion model = i.model;
    cout<<"FX "<<focal_length.first<<"FY "<<focal_length.second<<"CX"<<i.ppx<<"CY"<<i.ppy<<endl;

    return 0;
}
