#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <time.h>
#include <thread>


#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pangolin/pangolin.h>

#include "System.h"
#include "mypointcloud.h"

#define TARGET_PCL down_seg_pcl_sor
//#include "SemSeg.h"

/*
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"



#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
*/

using namespace std;
using namespace cv;

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.

typedef pcl::PointXYZRGBL PointT;
typedef pcl::PointCloud<PointT> PointCloud;


PointCloud::Ptr seg_pcl;

PointCloud::Ptr down_seg_pcl_sor;

unsigned int text_id = 0;


void pickPointEvent (const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
         
  if (event.getPointIndex () == -1)
  {
    return;
  }
  //std::cout << (event.getPointIndex()) << std::endl;
  //std::cout <<  << std::endl
  //PointT p = seg_pcl[(int)(event.getPointIndex())];
  //PointCloud::iterator pP = seg_pcl->begin;
  //PointT p =  seg_pcl->points[event.getPointIndex()];
  PointT p =  down_seg_pcl_sor->points[event.getPointIndex()];
  std::cout <<"X:" <<p.x <<" Y:" << p.y << " Z:" << p.z << " L:" << p.label << " - " <<  labeltoText[p.label] << std::endl;
  //std::cout << pP->x << std::endl;

  
 
}

pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGBL>::ConstPtr cloud, std::string window_name)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (window_name));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBL> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGBL> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  //viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)viewer.get ());
  //viewer->registerMouseCallback (mouseEventOccurred, (void*)viewer.get ());
  viewer->registerPointPickingCallback(pickPointEvent, (void*)viewer.get ());

  return (viewer);
}

int main(int argc, char* argv[]) {

  

  std::string image_path_rgb;
  std::string image_path_depth;
  std::string strSettingPath =  "/home/richard/orbslam_thesis/Examples/RGB-D/TUM3.yaml";

  std::set<int> listLabel;
  std::cout << "argc = " << argc << std::endl;
  if(argc == 3)
  {
    image_path_rgb = argv[1];
    image_path_depth = argv[1];
  }
  else
  {
    image_path_rgb = "/home/richard/Data/DATASETS/TUM/fr3_walking_rpy/rgb/1341846647.734820.png";
    image_path_depth = "/home/richard/Data/DATASETS/TUM/fr3_walking_rpy/depth/1341846647.802269.png";
  }


  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    float resolution_ = fSettings["PointCloudMapping.Resolution"];

  //PointCloudMapping* mpPCM;
  //mpPCM = new PointCloudMapping(settings_path);

  SemSeg* mpSemSeg;
  mpSemSeg = new SemSeg();


  std::cout << "Reading Image RGB " << image_path_rgb << std::endl;
  std::cout << "Reading Image Depth " << image_path_depth << std::endl;

  

  clock_t total;
  total = clock();

  cv::Mat color;
  cv::Mat depth;

  color = cv::imread(image_path_rgb,CV_LOAD_IMAGE_UNCHANGED);
  depth = cv::imread(image_path_depth,CV_LOAD_IMAGE_UNCHANGED);

  if(color.empty())
  {
    std::cout <<"Read Color Failed" << std::endl;

  }

  if(depth.empty())
  {
    std::cout <<"Read Depth Failed" << std::endl;
    
  }

  depth.convertTo(depth,CV_32F,1.0f/5000);


  //PointCloud::Ptr seg_pcl( new PointCloud());
  seg_pcl = boost::make_shared< PointCloud >( );
  down_seg_pcl_sor = boost::make_shared<PointCloud>();
  PointCloud::Ptr seg_pcl_masked( new PointCloud());
  PointCloud::Ptr pcl_color( new PointCloud());
  PointCloud::Ptr down_seg_pcl(new (PointCloud));
  PointCloud::Ptr down_seg_pcl_masked(new (PointCloud));
  //PointCloud::Ptr down_seg_pcl_sor(new (PointCloud));

  
  cv::Mat SegLabel(color.size().height,color.size().width,CV_8U);
  cv::Mat SegColor(color.size().height,color.size().width,CV_8UC3);
  cv::Mat DynMask(color.size().height,color.size().width,CV_8U);
  //std::cout << "3.1 segcolor: " << kf->mnId << std::endl;
  std::cout << "Sent color Rows: " << color.rows << " Cols: " << color.cols << std::endl;
  //cv::Mat dMask(color.size().height,color.size().width,CV_8U);
  mpSemSeg->RunColorAndMask(color,SegLabel,SegColor,DynMask);
  //mpSemSeg->RunColorAndMask(color,SegColor,DynMask);
  //mpSemSeg->RunColorAndMask(color,SegColor,dMask);
  
  // point cloud is null ptr
  //std::cout << "3.2 Compute PCL from Depth KF mnId: " << kf->mnId << std::endl;
  std::cout << "Depth Rows: " << depth.rows << " Cols: " << depth.cols << std::endl;
  int count_defective = 0;

  listLabel.clear();
  /*
  if (depth.empty())
  {
      std::cout << "Depth is Empty" << std::endl;
      count_defective++;
      std::cout << "Defective Depth Frames: " << count_defective << std::endl;
      return seg_pcl;
  }
  */
  
  if (depth.rows > 480 || depth.rows < 0  || depth.cols > 640 || depth.cols < 0)
  {
      std::cout <<"Depth map has abnormal dimensions" << std::endl;
      count_defective++;
      std::cout << "Defective Depth Frames: " << count_defective << std::endl;
      return 1;
  }
  for ( int m=0; m<depth.rows; m+=1 )
  {
      for ( int n=0; n<depth.cols; n+=1 )
      {
          //std::cout << "Cols index = " << n << std::endl;
          float d = depth.ptr<float>(m)[n];
          //std::cout << "Depth = " << d << std::endl; 
          //M.at<double>(0,0)
          //float d = depth.at<float>(m,n);
          if (d < 0.01 || d>5)
              continue;
          
          PointT p;
          p.z = d;
          p.x = ( n - cx) * p.z / fx;
          p.y = ( m - cy) * p.z / fy;
          //p.b = 200;
          //p.g = 200;
          //p.r = 200;
          //p.b = SegColor.ptr<uchar>(m)[n*3];
          //p.g = SegColor.ptr<uchar>(m)[n*3+1];
          //p.r = SegColor.ptr<uchar>(m)[n*3+2];
          p.label = SegLabel.ptr<uchar>(m)[n];

          listLabel.insert(p.label);
          
          if (DynMask.ptr<uchar>(m)[n]==1)
          {
            seg_pcl->points.push_back(p);
          }
          else
          {
            seg_pcl->points.push_back(p);
            seg_pcl_masked->points.push_back(p);
          }
      }
  }

  //std::cout << "Point CLoud Obtained from Depth" << std::endl;
  //std::cout << "Befor SE3 to quat" << std::endl;
  //Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
  //PointCloud::Ptr cloud(new PointCloud);
  //pcl::transformPointCloud( *seg_pcl, *cloud, T.inverse().matrix());
  //cloud->is_dense = false;
  std::cout << "Seg Color PCL size = " << seg_pcl->points.size() << std::endl;
  std::cout << "Masked Seg Color PCL size = " << seg_pcl_masked->points.size() << std::endl;

  pcl::VoxelGrid<PointT>  voxel;
  voxel.setLeafSize( resolution_, resolution_, resolution_);
  voxel.setInputCloud(seg_pcl);
  voxel.filter(*down_seg_pcl);
  std::cout << "DownSampled Seg Color size= " << down_seg_pcl->points.size() << std::endl;


  voxel.setInputCloud(seg_pcl_masked);
  voxel.filter(*down_seg_pcl_masked);
  std::cout << "DownSampled Seg Color size= " << down_seg_pcl_masked->points.size() << std::endl;
  


  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud (down_seg_pcl_masked);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*down_seg_pcl_sor);

  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = rgbVis(down_seg_pcl_sor,"SegPCL");

  for (auto it=listLabel.begin(); it != listLabel.end(); ++it) 
        std::cout << *it << "- "<< labeltoText[*it] << std::endl; 



  //pcl::visualization::PCLVisualizer::Ptr viewer_masked;
  //viewer_masked = rgbVis(seg_pcl_masked,"SegPCL Masked");
  //pcl::visualization::PCLVisualizer::Ptr view_downsampled;
  //view_downsampled = rgbVis(down_seg_pcl,"SegPCL DownSampled");
//
  //pcl::visualization::PCLVisualizer::Ptr view_downsampled_masked;
  //view_downsampled_masked = rgbVis(down_seg_pcl_masked,"SegPCL DownSampled Masked");


  pcl::io::savePCDFile ("seg_pcl.pcd", *seg_pcl);
  pcl::io::savePCDFile ("seg_pcl_masked.pcd", *seg_pcl_masked);
  pcl::io::savePCDFile ("down_seg_pcl_sor.pcd.", *down_seg_pcl_sor);
  pcl::io::savePCDFile ("down_seg_pcl_masked.pcd", *down_seg_pcl_masked);



  while (!viewer->wasStopped ())
  {
      viewer->spinOnce (100);
      //view_downsampled->spinOnce (100);
      //viewer_masked->spinOnce(100);
      //view_downsampled_masked->spinOnce(100);
  }


  return 0;
}

///Pangolic Cloud Viewer
  //mycloud = mpPCM->generatePointCloud(color,depth);
  
  //pcl::visualization::PCLVisualizer::Ptr viewer;
  //pcl::visualization::CloudViewer viewer("viewer");

  /*
  pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer",1024,768);
  glEnable(GL_DEPTH_TEST);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  
  while(1)
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    pangolin::FinishFrame();
    //mpMapDrawer->DrawMapPoints();
    glPointSize(2); //2 mPointSize
    glBegin(GL_POINTS); 
    glColor3f(0.0,0.0,0.0); //set color R,G,B in float

    for (pcl::PointCloud<PointT>::iterator iter = mycloud->begin(); iter != mycloud->end(); iter++)
    {
      glVertex3f(iter->x,iter->y,iter->z);
    }

    glEnd();

    getchar();

  }
  */