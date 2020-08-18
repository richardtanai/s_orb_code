#pragma once


#ifndef SEM_SEG_H
#define SEM_SEG_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <time.h>
#include <sstream>
#include <mutex>

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


//Test result class ====
typedef struct Segmented2D //results
{
    cv::Mat Prediction;  //per pixel label probabilities
    cv::Mat Prob; //probabilities
    cv::Mat DynamicMask;  // binary mask for determining the pixel
    std::vector<int> classes;//list of classes in image
} Segmented2D;

typedef struct MaskLabel{
  int label;
  std::vector<int> indices;

}MaskLabel;

class SemSeg
{
public:
  /** Default constructor */
  SemSeg(const std::string _strSettingsFile);//Network initialization

  SemSeg();

  /** Default destructor */
  ~SemSeg();//Destructor
  
  bool isWarmedUp();
  void Run(const cv::Mat& bgr_img, Segmented2D& results);
  void RunColor(const cv::Mat& cvImg, cv::Mat& resultImg);
  void RunColorAndMask(const cv::Mat& cvImg, cv::Mat& resultImg, cv::Mat& resultMask);
  void RunColorAndMask(const cv::Mat& cvImg, cv::Mat& resultClasses, cv::Mat& resultImg, cv::Mat& resultMask);
  void RunSegLabel(const cv::Mat& cvImg);
  void ComputeAllLatest(const cv::Mat& cvImg, const cv::Mat& cvDepth);

  void SetSegMaxDepth(float val);

  void SetSegMinDepth(float val);

  void SetSegMinProb(float val);

  
  cv::Mat resized_output;
  cv::Mat resized_output_remap;
  cv::Mat resized_output_prob;
  cv::Mat resized_color_out;
  cv::Mat resized_mask_out;

  cv::Mat LastSegLabel;
  cv::Mat LastSegLabelRemap;
  cv::Mat LastSegColor;
  cv::Mat LastSegProb;
  cv::Mat LastDynMask;
  cv::Mat LastRemapped;
  


  int isRGB;

  float SegMinDepth;
  float SegMaxDepth;
  float SegMinProb;

  int useSegDepthTh;
  int useSegProbTh;

    //nn parameters
  cv::String path_graph;
  std::string output_node_label;
  std::string output_node_prob;
  std::mutex settingsMutex;

  int mDilation_size;
  


  cv::Mat Show(const cv::Mat& bgr_img, Segmented2D& results);
private:
  // ncnn::Net * det_net_mobile;  
  //ncnn::Net * det_net_ptr;//Detect network pointer
  //ncnn::Mat * net_in_ptr; //Detect network pointer
  
  //cv::FileStorage fsSettings;
  //std::string strSettingsFile;
  int remapLabel(int _label);
  bool isDynamic(int _label);
  

  int tensor_image_width = 513;
  int tensor_image_height = 385;
  bool warmedUp;
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Session* sess;
  
  int image_number=0;
  std::vector<int> mcompression_params;
  
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

#endif
