
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

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
        cerr << endl << "Usage: ./rs_live path_to_vocabulary path_to_settings bagfile" << endl;
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
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    string bag_path = argv[3];

    cfg.enable_device_from_file(bag_path,false);
    auto profile = pipe.start(cfg);
    auto playback = profile.get_device().as<rs2::playback>();
    //auto profile = mconfig.get_stream(RS2_STREAM_COLOR)
    //                     .as<video_stream_profile>();
    rs2::align align_to_color(RS2_STREAM_COLOR);

    // Retrieve paths to images
    //vector<string> vstrImageFilenamesRGB;
    //vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    //string strAssociationFilename = string(argv[4]);
    //LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    //int nImages = vstrImageFilenamesRGB.size();
    //if(vstrImageFilenamesRGB.empty())
    //{
    //    cerr << endl << "No images found in provided path." << endl;
    //    return 1;
    //}
    //else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    //{
    //    cerr << endl << "Different number of images for rgb and depth." << endl;
    //    return 1;
    //}

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true,true);

    // Vector for tracking time statistics
    //vector<float> vTimesTrack;
    //vTimesTrack.resize(nImages);

    //cout << endl << "-------" << endl;
    //cout << "Start processing sequence ..." << endl;
    //cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    //cv::Mat imRGB, imD;
    //for(int ni=0; ni<nImages; ni++)
    //{
        // Read image and depthmap from file
        //imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        //imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);

    float sum_track = 0;
    int n_frames = 0;

    for(int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        auto frames = pipe.wait_for_frames();
    }

    std::cout << "Camera Initialize Complete" << std::endl;

    while(1)
    {
        if (playback.current_status() == RS2_PLAYBACK_STATUS_STOPPED)
        {
            break;
        }

        auto frames_un = pipe.wait_for_frames();

        auto frames = align_to_color.process(frames_un);


        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        cv::Mat imRGB(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat imD(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        double timestamp = frames.get_timestamp(); 
        
        //double tframe = vTimestamps[ni];

        //double tframe = 0.0;

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at" << std::endl;
        }

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

        // Wait to load the next frame
        //double T=0;
        //if(ni<nImages-1)
        //    T = vTimestamps[ni+1]-tframe;
        //else if(ni>0)
        //    T = tframe-vTimestamps[ni-1];
//
        //if(ttrack<T)
        //    usleep((T-ttrack)*1e6);
    }
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
