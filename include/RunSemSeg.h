/**
* This file is part of ORB-SLAM2.
* Class that runs target detection
*/
#pragma once

#ifndef RUNSEMSEG_H
#define RUNSEMSEG_H

//#include "System.h"
#include <opencv2/core/core.hpp>
#include <condition_variable>// Multithreaded lock state variable
#include "KeyFrame.h"
#include "SemSeg.h"        // 2d target detection result===
#include <boost/make_shared.hpp>
#include <thread>
#include "System.h"


using namespace ORB_SLAM2;
class ORB_SLAM2::KeyFrame;

class SemSeg; // Declare target detection class

class RunSemSeg
{

public:
    void insertKFColorImg(KeyFrame* kf, cv::Mat color);// Insert a color image of a keyframe，Can be deleted after detection===
    void Run(void);// Thread run function====
    RunSemSeg();
    ~RunSemSeg();

protected:
    std::shared_ptr<thread>  mRunThread; // Execution thread==

    condition_variable  colorImgUpdated; 
// Keyframe update <condition_variable> The header file mainly contains classes and functions related to condition variables.
// Global condition variable. Used for mutual waiting between multiple threads！！！！！！！
    // condition_variable Class reference https://msdn.microsoft.com/zh-cn/magazine/hh874752(v=vs.120)
    mutex               colorImgMutex;// Keyframe update  Mutex lock
    std::vector<cv::Mat>     colorImgs;   // Grayscale    Array
    std::vector<KeyFrame*> mvKeyframes;  // Key frame pointer array vector
    mutex mvKeyframesMutex;

    //std::vector<std::vector<Object>> mvvObjects;// Keep each keyframe image 2d test results==== 
    //mutex  mvvObjectsMutex;


    uint16_t          lastKeyframeSize =0;
    SemSeg* mSemSeg;// Target detection object====
};
#endif
