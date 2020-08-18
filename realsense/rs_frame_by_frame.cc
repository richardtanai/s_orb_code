
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <cstdlib>

#include<opencv2/core/core.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

#include <System.h>
#include "cv-helpers.hpp"
#include <librealsense2/rs.hpp>


using namespace std;
//using namespace rs2;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);


//Intel RealSense D435I
int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./rs_live path_to_vocabulary path_to_settings bagfile last_frame_number" << endl;
        return 1;

        
    }
    /*
    const auto window_name = "Display Image";
    cv::namedWindow(window_name,);
    rs2::context ctx;

    auto device = ctx.query_devices();
	auto dev = device[0];

		if (strcmp(dev.get_info(RS2_CAMERA_INFO_NAME), "Intel RealSense D435I") == 0) // Check for compatibility, must have if executed on a computer
		{
			config cfg;
			string serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			string json_file_name = "/home/richard/orbslam_thesis/realsense/preset.json";
			cout << "Configuring camera : " << serial << endl;

			auto advanced_mode_dev = dev.as<rs400::advanced_mode>();

			// Check if advanced-mode is enabled to pass the custom config
			if (!advanced_mode_dev.is_enabled())
			{
				// If not, enable advanced-mode
				advanced_mode_dev.toggle_advanced_mode(true);
				cout << "Advanced mode enabled. " << endl;
			}

			// Select the custom configuration file
			std::ifstream t(json_file_name);
			std::string preset_json((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
			advanced_mode_dev.load_json(preset_json);
			cfg.enable_device(serial);
		}
		else
		{
			cout << "Selected device is not an Intel RealSense D415, check the devices list. " << endl;
		}

        */

    rs2::pipeline pipe;
    rs2::config cfg;
    //cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    //cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    string bag_path = argv[3];

    //int last_frame_num = atoi(argv[4]);

    cfg.enable_device_from_file(bag_path,false);


    auto profile = pipe.start(cfg);

    auto playback = profile.get_device().as<rs2::playback>();
    playback.set_real_time(false);

    rs2::frameset frames_un;

    rs2::align align_to_color(RS2_STREAM_COLOR);

    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true,true);

    //std::vector<double> vTimestamps;
    //std::vector<cv::Mat> vRGB;
    //std::vector<cv::Mat> vDepth;

    size_t frame_index = 0;
    float sum_track = 0;
    int n_frames = 0;

    bool stop = false;
	bool playing = false;


	while( !stop )
	{
		if( pipe.poll_for_frames( &frames_un) )
		{
			playing = true;
            if(n_frames < 30)
            {
                ++frame_index;
                n_frames ++;
                continue;
            }

			// Add some frames processing here...
            auto frames = align_to_color.process(frames_un);

			//std::cout << "\rsuccessfully retrieved frame #" << ++frame_index << " (" << playback.current_status() << ")" << std::endl;

            rs2::frame color_frame = frames.get_color_frame();
            rs2::frame depth_frame = frames.get_depth_frame();

            cv::Mat imRGB(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat imD(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

            //imRGB.convertTo(imRGB,cv::CV_RGB2BGR);
            
            double timestamp = frames.get_timestamp();

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,timestamp);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        //vTimesTrack[ni]=ttrack;

        //std::cout << "System Processing Frame time: " << ttrack << endl;

        sum_track += ttrack;
        n_frames ++ ;

        //if(last_frame_num != 0)
        //{
        //    if (frame_index >= last_frame_num)
        //    {
        //        //playback.pause();
        //        playback.stop();
        //        //usleep(100000);
        //        //playback.
        //        break;
        //    }
        //    
        //}
		}

        
		else
		{
			usleep(1000);
		}

		stop = playing && ( playback.current_status() == RS2_PLAYBACK_STATUS_STOPPED );
	}

	std::cout << std::endl << "successfully ended file playback" << std::endl;

	pipe.stop();
    

    

    

   
    
    
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    //sort(vTimesTrack.begin(),vTimesTrack.end());
    //float totaltime = 0;
    //for(int ni=0; ni<nImages; ni++)
    //{
    //    totaltime+=vTimesTrack[ni];
    //}
    //cout << "-------" << endl << endl;
    //cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    //cout << "mean tracking time: " << totaltime/nImages << endl;


    cout << "mean tracking time: " << sum_track/n_frames << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("results/CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("results/KeyFrameTrajectory.txt");   

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
/*
void read_from_bag(unsigned int num_frames, std::string const fname )
{
    rs2::pipeline pipeline;
    rs2::config cfg;

    std::cout << fname << std::endl;
    cfg.enable_device_from_file(fname);
    cfg.enable_all_streams();

    rs2::pipeline_profile pipe_profile;
    rs2::frameset frameset;
    uint32_t counter = 0;
    rs2::device dev;

    try
    {
    pipe_profile = pipeline.start(cfg);
    dev = pipeline.get_active_profile().get_device();
    rs2::playback playback = dev.as<rs2::playback>();

    if (playback)
    {
    ;
    //std::cout << "Playback from file\n";
    }
    else
    {
    std::cerr << "playback error" << "\n";
    }
    playback.set_real_time(false);
    std::chrono::nanoseconds duration = playback.get_duration();
    std::cout <<"File duration: " << duration.count() << std::endl;
    float frame_count = duration.count() / (1e9);
    std::cout << "frames: " << frame_count * 30 << std::endl;
    uint64_t curPos;
    uint64_t lastPos = 0;
    while (pipeline.try_wait_for_frames(&frameset))
    {
    curPos = playback.get_position();
    if (curPos < lastPos)
        break;
    else
    {

    counter++;
    lastPos = curPos;
    depth_frame = frameset.get_depth_frame();
    colour_frame = frameset.get_color_frame();
    }
    }
    //std::cout << dev.get_info(RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION) << "\n";
    }
    catch (rs2::error &e)
    {
    std::cerr << "Problem in reading file. Failed in function:" <<
    e.get_failed_function() << std::endl;
    std::cerr << e.what() << std::endl; // What is the error
    return;
    }


    std::cout << counter << " frames read." << std::endl;
    pipeline.stop();

    return;
}
*/