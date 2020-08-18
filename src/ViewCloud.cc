#include "System.h"
#include "SegCloudViewer.h"
//#include "SemSeg.h"



#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


typedef pcl::PointXYZRGBL PointT;
typedef pcl::PointCloud<PointT> PointCloud;


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


  PointCloud::Ptr seg_pcl ( new PointCloud());
  PointCloud::Ptr seg_pcl_masked( new PointCloud());
  PointCloud::Ptr pcl_color( new PointCloud());


  
  cv::Mat SegLabel(color.size().height,color.size().width,CV_8U);
  cv::Mat SegColor(color.size().height,color.size().width,CV_8UC3);
  cv::Mat DynMask(color.size().height,color.size().width,CV_8U);

  std::cout << "Sent color Rows: " << color.rows << " Cols: " << color.cols << std::endl;
  //cv::Mat dMask(color.size().height,color.size().width,CV_8U);
  mpSemSeg->RunColorAndMask(color,SegLabel,SegColor,DynMask);
  //mpSemSeg->RunColorAndMask(color,SegColor,DynMask);
  //mpSemSeg->RunColorAndMask(color,SegColor,dMask);

  std::cout << "Depth Rows: " << depth.rows << " Cols: " << depth.cols << std::endl;
  int count_defective = 0;

  listLabel.clear();

  //check if depth image valid
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
          float d = depth.ptr<float>(m)[n];

          if (d < 0.01 || d>5)
              continue;
          
          PointT p;
          p.z = d;
          p.x = ( n - cx) * p.z / fx;
          p.y = ( m - cy) * p.z / fy;
          p.b = SegColor.ptr<uchar>(m)[n*3];
          p.g = SegColor.ptr<uchar>(m)[n*3+1];
          p.r = SegColor.ptr<uchar>(m)[n*3+2];
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

  //Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
  //PointCloud::Ptr cloud(new PointCloud);
  //pcl::transformPointCloud( *seg_pcl, *cloud, T.inverse().matrix());
  //cloud->is_dense = false;

  std::cout << "Seg Color PCL size = " << seg_pcl->points.size() << std::endl;
  std::cout << "Masked Seg Color PCL size = " << seg_pcl_masked->points.size() << std::endl;
  PointCloud::Ptr down_seg_pcl(new (PointCloud));
  PointCloud::Ptr down_seg_pcl_masked(new (PointCloud));
  pcl::VoxelGrid<PointT>  voxel;
  voxel.setLeafSize( resolution_, resolution_, resolution_);
  voxel.setInputCloud(seg_pcl);
  voxel.filter(*down_seg_pcl);
  std::cout << "DownSampled Seg Color size= " << down_seg_pcl->points.size() << std::endl;


  voxel.setInputCloud(seg_pcl_masked);
  voxel.filter(*down_seg_pcl_masked);
  std::cout << "DownSampled Seg Color size= " << down_seg_pcl_masked->points.size() << std::endl;
  


  //pcl::visualization::PCLVisualizer::Ptr viewer;(seg_pcl,"SegPCL");

  SegCloudViewer seg_viewer = SegCloudViewer(seg_pcl,"SegPCL");

  for (auto it=listLabel.begin(); it != listLabel.end(); ++it) 
        cout << *it << endl; 



  //pcl::visualization::PCLVisualizer::Ptr viewer_masked;
  //viewer_masked = rgbVis(seg_pcl_masked,"SegPCL Masked");
  //pcl::visualization::PCLVisualizer::Ptr view_downsampled;
  //view_downsampled = rgbVis(down_seg_pcl,"SegPCL DownSampled");
//
  //pcl::visualization::PCLVisualizer::Ptr view_downsampled_masked;
  //view_downsampled_masked = rgbVis(down_seg_pcl_masked,"SegPCL DownSampled Masked");


  while(true)
  {
      seg_viewer.spinOnce (100);
      //std::cout << "Spinning Once" << std::endl;
       //view_downsampled->spinOnce (100);
      //viewer_masked->spinOnce(100);
      //view_downsampled_masked->spinOnce(100);
  }


  return 0;
}