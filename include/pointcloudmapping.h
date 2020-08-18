/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#pragma once

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

//#define PCL_NO_PRECOMPILE
#include "System.h"
#include <condition_variable>
#include <thread>
#include <SemSeg.h>
#include <vector>
#include <set>


//#include <pcl/pcl_macros.h>
//#include <pcl/impl/instantiate.hpp>

#include <pcl/point_types.h>


#include <pcl/common/transforms.h>


#include <pcl/filters/voxel_grid.h>



#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>

#include <pcl/features/normal_3d.h>


#include <pcl/search/kdtree.h>



#include <pcl/segmentation/conditional_euclidean_clustering.h>



//#include "extra_func.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/projection_matrix.h>

#include "Objects.h"
#include "Converter.h"
#include <omp.h>

#include <boost/make_shared.hpp>




typedef pcl::PointXYZL PointT;
typedef pcl::PointCloud<PointT> PointCloud;

typedef pcl::PointXYZLNormal PointTN;
typedef pcl::PointCloud<PointTN> PointCloudN;

typedef pcl::PointXYZL PointTypeIO;
typedef pcl::PointXYZLNormal PointTypeFull;

using namespace std;

using namespace ORB_SLAM2;

class LabelStat{
    public:
        uint32_t label;
        int count;

};


class FrameData{
    public:
        KeyFrame* mpKF;

        cv::Mat mRGB;
        cv::Mat mDepth;
        cv::Mat mSegLabel;
        cv::Mat mSegLabelRemap;
        cv::Mat mSegColor;
        cv::Mat mSegProb;
        cv::Mat mDynMask;
        
        
};

class Object3D{
    public:
        Object3D();
        void ComputeAll(float radius_factor);
        void Append(Object3D* fusion, float radius_factor);

        PointCloud::Ptr pCloud;
        Eigen::Vector3f maxPt;
        Eigen::Vector3f minPt;
        Eigen::Vector3f centroid;
        Eigen::Vector3f sizePt;
        Eigen::Vector3f boxCenter;

        int object_id;
        int label;
        int n_points;
        float searchRadius;
        
        std::vector<LabelStat> mvLabels;
        

};


class ObjectDataBase{
    public:
        ObjectDataBase(std::string settingsPath);
        void ClusterPCL(PointCloud::Ptr cloud_in, pcl::IndicesPtr pIndices, FrameData* pFD, std::vector<Object3D*>& mvpObject3D);
        std::vector<Object3D*> mvpObjects;
        void Merge(std::vector<Object3D*>& mvpNew);
        void AddObject(Object3D* newObject);
        void AddObjectByCentroid(Object3D* newObject);
        void ReMerge();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr RenderLatest();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr RenderLatest(int _label);

        void RenderAndSave();
        void RenderAndSaveFilter(int _label, string pcd_name);

        std::mutex dataMutex;

        float resolution;

        int lastReMerge =0;
        int useRemerge = 0;

        int useAddByCentroid;
        float radius_factor = 5;
        
        int DBSize =0;
        int colormap[151][3] = {{0, 0, 0},
      {120, 120, 120},
      {180, 120, 120},
      {6, 230, 230},
      {80, 50, 50},
      {4, 200, 3},
      {120, 120, 80},
      {140, 140, 140},
      {204, 5, 255},
      {230, 230, 230},
      {4, 250, 7},
      {224, 5, 255},
      {235, 255, 7},
      {150, 5, 61},
      {120, 120, 70},
      {8, 255, 51},
      {255, 6, 82},
      {143, 255, 140},
      {204, 255, 4},
      {255, 51, 7},
      {204, 70, 3},
      {0, 102, 200},
      {61, 230, 250},
      {255, 6, 51},
      {11, 102, 255},
      {255, 7, 71},
      {255, 9, 224},
      {9, 7, 230},
      {220, 220, 220},
      {255, 9, 92},
      {112, 9, 255},
      {8, 255, 214},
      {7, 255, 224},
      {255, 184, 6},
      {10, 255, 71},
      {255, 41, 10},
      {7, 255, 255},
      {224, 255, 8},
      {102, 8, 255},
      {255, 61, 6},
      {255, 194, 7},
      {255, 122, 8},
      {0, 255, 20},
      {255, 8, 41},
      {255, 5, 153},
      {6, 51, 255},
      {235, 12, 255},
      {160, 150, 20},
      {0, 163, 255},
      {140, 140, 140},
      {250, 10, 15},
      {20, 255, 0},
      {31, 255, 0},
      {255, 31, 0},
      {255, 224, 0},
      {153, 255, 0},
      {0, 0, 255},
      {255, 71, 0},
      {0, 235, 255},
      {0, 173, 255},
      {31, 0, 255},
      {11, 200, 200},
      {255, 82, 0},
      {0, 255, 245},
      {0, 61, 255},
      {0, 255, 112},
      {0, 255, 133},
      {255, 0, 0},
      {255, 163, 0},
      {255, 102, 0},
      {194, 255, 0},
      {0, 143, 255},
      {51, 255, 0},
      {0, 82, 255},
      {0, 255, 41},
      {0, 255, 173},
      {10, 0, 255},
      {173, 255, 0},
      {0, 255, 153},
      {255, 92, 0},
      {255, 0, 255},
      {255, 0, 245},
      {255, 0, 102},
      {255, 173, 0},
      {255, 0, 20},
      {255, 184, 184},
      {0, 31, 255},
      {0, 255, 61},
      {0, 71, 255},
      {255, 0, 204},
      {0, 255, 194},
      {0, 255, 82},
      {0, 10, 255},
      {0, 112, 255},
      {51, 0, 255},
      {0, 194, 255},
      {0, 122, 255},
      {0, 255, 163},
      {255, 153, 0},
      {0, 255, 10},
      {255, 112, 0},
      {143, 255, 0},
      {82, 0, 255},
      {163, 255, 0},
      {255, 235, 0},
      {8, 184, 170},
      {133, 0, 255},
      {0, 255, 92},
      {184, 0, 255},
      {255, 0, 31},
      {0, 184, 255},
      {0, 214, 255},
      {255, 0, 112},
      {92, 255, 0},
      {0, 224, 255},
      {112, 224, 255},
      {70, 184, 160},
      {163, 0, 255},
      {153, 0, 255},
      {71, 255, 0},
      {255, 0, 163},
      {255, 204, 0},
      {255, 0, 143},
      {0, 255, 235},
      {133, 255, 0},
      {255, 0, 235},
      {245, 0, 255},
      {255, 0, 122},
      {255, 245, 0},
      {10, 190, 212},
      {214, 255, 0},
      {0, 204, 255},
      {20, 0, 255},
      {255, 255, 0},
      {0, 153, 255},
      {0, 41, 255},
      {0, 255, 204},
      {41, 0, 255},
      {41, 255, 0},
      {173, 0, 255},
      {0, 245, 255},
      {71, 0, 255},
      {122, 0, 255},
      {0, 255, 184},
      {0, 92, 255},
      {184, 255, 0},
      {0, 133, 255},
      {255, 214, 0},
      {25, 194, 194},
      {102, 255, 0},
      {92, 0, 255}};
};



bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance);



class PointCloudMapping
{
public:
    


    //for testing purposes only
    PointCloudMapping(const string strSettingPath, SemSeg* pSemSeg);

    PointCloudMapping(double resolution_, const string strSettingPath, SemSeg* pSemSeg);
    

    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( FrameData* pFD );
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth );
    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& segColor, cv::Mat& segLabel, cv::Mat& segProb, cv::Mat& dynMask);
    void shutdown();
    void viewer();

    void cloud_viewer();
    void my_what_Mat(cv::Mat& inMat);

    void SaveColorCloud();
    void SaveLabelCloud();
    //shared_ptr< SemSeg > mpSemSeg;
    SemSeg* mpSemSeg;
    ObjectDataBase* mpOD;

    int useCloudViewer;
    
    float pclMinDepth;
    float pclMaxDepth;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  view_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  objectCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  color_cloud;
    PointCloud::Ptr  label_cloud;
    //camera intrinsiscs
    float cx, cy, fx, fy;
    float classProb_thresh;

    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void generatePointCloud(FrameData* pFD);
    void generateLabelPointCloud(FrameData* pFD);
    bool isBackground(int _label);

    int usePCLMapping;
    
    int countLabelCloud = 0;
    int labelCloudInterval = 5;
    int useSimple;
    int removeBG = 0;
    
    
    

protected:
    

    PointCloud::Ptr globalMap;
    shared_ptr<thread>  viewerThread;
    shared_ptr<thread> cloudThread;

    bool    shutDownFlag    =false;
    mutex   shutDownMutex;

    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;

    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;
    vector<cv::Mat>    mvSegColor;
    vector<cv::Mat>    mvSegLabel;
    vector<cv::Mat>    mvSegProb;
    vector<cv::Mat>    mvDynMask;
    vector<FrameData*> mvpFrameData; 
    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;
    int count_defective;
    double resolution = 0.04;
    pcl::VoxelGrid<PointT>  voxel;
    
};

#endif // POINTCLOUDMAPPING_H
