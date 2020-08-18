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

#include "pointcloudmapping.h"
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>

//struct cluster_label_stats{
//    int label;
//    float sum_prob;
//    int N;
//    float average;
//}cluster_label_stats;

using namespace std;

ofstream PCL_Log("results/PLC_Log.csv");

Object3D::Object3D()
{
    mvLabels.clear();
    this->pCloud = boost::make_shared< PointCloud >( );
}
void Object3D::ComputeAll(float radius_factor)
{   
    for(auto it = pCloud->begin(); it != pCloud->end(); ++it)
    {
        auto itLabel = it->label;
        LabelStat _labelStat;
        if(mvLabels.empty())
        {
            _labelStat.count=1;
            _labelStat.label = itLabel;
            mvLabels.push_back(_labelStat);
        }
        else
        {
            auto it2 = find_if(mvLabels.begin(),mvLabels.end(),[&itLabel](const LabelStat& v ){ return v.label == itLabel; });
            if(it2 != mvLabels.end())
            {
              it2->count++;
            }
            else
            {
              _labelStat.count=1;
              _labelStat.label = itLabel;
              mvLabels.push_back(_labelStat);
            }
        }
        
    }
    std::sort(mvLabels.begin(),mvLabels.end(), [](const LabelStat& lhs, const LabelStat& rhs) {return lhs.count > rhs.count;}); //sort highest to lowest
    this->label = mvLabels.front().label; 


    auto cmp_x = [](PointT const& l, PointT const& r) { return l.x < r.x; };
    auto minmax_x = std::minmax_element(pCloud->begin(), pCloud->end(), cmp_x);// Point cloud x max

    auto cmp_y = [](PointT const& l, PointT const& r) { return l.y < r.y; };
    auto minmax_y = std::minmax_element(pCloud->begin(), pCloud->end(), cmp_y);// Point cloud x max

    auto cmp_z = [](PointT const& l, PointT const& r) { return l.z < r.z; };
    auto minmax_z = std::minmax_element(pCloud->begin(), pCloud->end(), cmp_z);// Point cloud x max  
         
         // Border center point =====
    auto sum_x = [](double sum_x, PointT const& l){return sum_x + l.x;};
	  auto sumx = std::accumulate(pCloud->begin(), pCloud->end(), 0.0, sum_x);
	  double mean_x =  sumx / pCloud->size(); //Mean

    auto sum_y = [](double sum_y, PointT const& l){return sum_y + l.y;};
	  auto sumy = std::accumulate(pCloud->begin(), pCloud->end(), 0.0, sum_y);
	  double mean_y =  sumy / pCloud->size(); //Mean

    auto sum_z = [](double sum_z, PointT const& l){return sum_z + l.z;};
	  auto sumz = std::accumulate(pCloud->begin(), pCloud->end(), 0.0, sum_z);
	  double mean_z =  sumz / pCloud->size(); //Mean

    this->minPt    = Eigen::Vector3f(minmax_x.first->x, minmax_y.first->y, minmax_z.first->z);
    this->maxPt    = Eigen::Vector3f(minmax_x.second->x,minmax_y.second->y,minmax_z.second->z);
    this->centroid = Eigen::Vector3f(mean_x, mean_y, mean_z); // Mean center
    // 3d border
    this->sizePt   = Eigen::Vector3f( this->maxPt[0]- this->minPt[0],
                                    this->maxPt[1]- this->minPt[1],
                                    this->maxPt[2]- this->minPt[2]);
    // 3d border center ===
    this->boxCenter= Eigen::Vector3f( this->minPt[0]+ this->sizePt[0]/2.0,
                                              this->minPt[1]+ this->sizePt[1]/2.0,
                                              this->minPt[2]+ this->sizePt[2]/2.0);


    this->searchRadius = (pow(sizePt[0],2) + pow(sizePt[1],2) + pow(sizePt[2],2))/(4*radius_factor);
    this->n_points = this->pCloud->size();
}

void Object3D::Append(Object3D* fusion, float radius_factor)
{
  //cout << "APPEND Object" << fusion->object_id << " to Object "<< this->object_id << endl;

  //*pCloudRep = *this->
  //Combine label statistics
  for(auto it = fusion->pCloud->begin(); it != fusion->pCloud->end(); ++it)
    {
        auto itLabel = it->label;
        LabelStat _labelStat;
        if(this->mvLabels.empty())
        {
            _labelStat.count=1;
            _labelStat.label = itLabel;
            this->mvLabels.push_back(_labelStat);
        }
        else
        {
            auto it2 = find_if(this->mvLabels.begin(),this->mvLabels.end(),[&itLabel](const LabelStat& v ){ return v.label == itLabel; });
            if(it2 != mvLabels.end())
            {
              it2->count++;
            }
            else
            {
              _labelStat.count=1;
              _labelStat.label = itLabel;
              mvLabels.push_back(_labelStat);
            }
        }
        
    }
    //sort labels //highest is first
    std::sort(this->mvLabels.begin(),this->mvLabels.end(), [](const LabelStat& lhs, const LabelStat& rhs) {return lhs.count > rhs.count;});
    this->label = mvLabels.front().label; //strongest label

    int new_n_points = this->n_points + fusion->n_points;

    *(this->pCloud) += *(fusion->pCloud);
    pcl::VoxelGrid<PointT> vgn;
    vgn.setInputCloud(this->pCloud);
    vgn.setDownsampleAllData(false);
    vgn.setLeafSize(0.01,0.01,0.01);
    vgn.filter(*(this->pCloud));

    auto cmp_x = [](PointT const& l, PointT const& r) { return l.x < r.x; };
    auto minmax_x = std::minmax_element(pCloud->begin(), pCloud->end(), cmp_x);// Point cloud x max

    auto cmp_y = [](PointT const& l, PointT const& r) { return l.y < r.y; };
    auto minmax_y = std::minmax_element(pCloud->begin(), pCloud->end(), cmp_y);// Point cloud x max

    auto cmp_z = [](PointT const& l, PointT const& r) { return l.z < r.z; };
    auto minmax_z = std::minmax_element(pCloud->begin(), pCloud->end(), cmp_z);// Point cloud x max  
         
         // Border center point =====
    //auto sum_x = [](double sum_x, PointTN const& l){return sum_x + l.x;};
	  //auto sumx = std::accumulate(pCloud->begin(), pCloud->end(), 0.0, sum_x);
	  //double mean_x =  sumx / pCloud->size(); //Mean
//
    //auto sum_y = [](double sum_y, PointTN const& l){return sum_y + l.y;};
	  //auto sumy = std::accumulate(pCloud->begin(), pCloud->end(), 0.0, sum_y);
	  //double mean_y =  sumy / pCloud->size(); //Mean
//
    //auto sum_z = [](double sum_z, PointTN const& l){return sum_z + l.z;};
	  //auto sumz = std::accumulate(pCloud->begin(), pCloud->end(), 0.0, sum_z);
	  //double mean_z =  sumz / pCloud->size(); //Mean
//
    this->minPt    = Eigen::Vector3f(minmax_x.first->x, minmax_y.first->y, minmax_z.first->z);
    this->maxPt    = Eigen::Vector3f(minmax_x.second->x,minmax_y.second->y,minmax_z.second->z);

    //Weighted mean of centroids
    this->centroid = Eigen::Vector3f(((this->centroid[0])*this->n_points + fusion->centroid[0]*fusion->n_points)/new_n_points, 
                                      ((this->centroid[1])*this->n_points + fusion->centroid[1]*fusion->n_points)/new_n_points, 
                                      ((this->centroid[2])*this->n_points + fusion->centroid[2]*fusion->n_points)/new_n_points); // Mean center
    // 3d border
    this->sizePt   = Eigen::Vector3f( this->maxPt[0]- this->minPt[0],
                                    this->maxPt[1]- this->minPt[1],
                                    this->maxPt[2]- this->minPt[2]);
    // 3d border center ===
    this->boxCenter= Eigen::Vector3f( this->minPt[0]+ this->sizePt[0]/2.0,
                                              this->minPt[1]+ this->sizePt[1]/2.0,
                                              this->minPt[2]+ this->sizePt[2]/2.0);


    this->searchRadius = (pow(sizePt[0],2) + pow(sizePt[1],2) + pow(sizePt[2],2))/(4*radius_factor);
    this->n_points = new_n_points;

    





  //cout << "MergeComplete New Size" << this->n_points << " " << this->pCloud->size() <<   endl;
  delete fusion;
  fusion = static_cast<Object3D*>(NULL);
  return;
}

ObjectDataBase::ObjectDataBase(std::string settingsPath)
{
  cv::FileStorage fSettings(settingsPath, cv::FileStorage::READ);
  this->useAddByCentroid = 0;
  this->useAddByCentroid = fSettings["Merge.Use.Centroid"];
  this->radius_factor = fSettings["Merge.RadiusFactor"];
  this->useRemerge= fSettings["OD.Use.Remerge"];
  this->resolution = fSettings["PCL.Resolution"];
  mvpObjects.clear();
}

void ObjectDataBase::RenderAndSave()
{
  auto Latest = RenderLatest();
  if(Latest->size() > 0)
  {
    pcl::io::savePCDFileASCII("results/cloud.pcd", *Latest);
  }
  
}

void ObjectDataBase::RenderAndSaveFilter(int label, string pcd_name)
{
  auto cloud = RenderLatest(label);
  if (cloud->size() > 0)
  {
    pcl::io::savePCDFileASCII(pcd_name, *cloud);
  }
  else
  {
    std::cout << "Cloud with label " << label << " is empty" <<endl;
  }
  
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ObjectDataBase::RenderLatest()
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZRGB>());
  for (auto it = mvpObjects.begin(); it!=mvpObjects.end(); ++it)
  {
    uint32_t label = (*it)->label;
    //int color_r = floor(rand()*255/100);
    //int color_g = floor(rand()*255/100);
    //int color_b = floor(rand()*255/100);
    if (label == 13)
    {
      continue;
    }

    for(auto it_cloud = (*it)->pCloud->begin(); it_cloud != (*it)->pCloud->end(); ++it_cloud)
    {
      pcl::PointXYZRGB p;
      p.x = it_cloud->x;
      p.y = it_cloud->y;
      p.z= it_cloud->z;
      //p.r = colormap[label][color_r];
      //p.g = colormap[label][color_g];
      //p.b = colormap[label][color_b];

      p.b = colormap[label][0];
      p.g = colormap[label][1];
      p.r = colormap[label][2];
      cloud_final->push_back(p);
    }
  }
  return cloud_final;
  
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ObjectDataBase::RenderLatest(int _label)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZRGB>());
  for (auto it = mvpObjects.begin(); it!=mvpObjects.end(); ++it)
  {
    uint32_t label = (*it)->label;
    //int color_r = floor(rand()*255/100);
    //int color_g = floor(rand()*255/100);
    //int color_b = floor(rand()*255/100);
    if (label != _label)
    {
      continue;
    }

    for(auto it_cloud = (*it)->pCloud->begin(); it_cloud != (*it)->pCloud->end(); ++it_cloud)
    {
      pcl::PointXYZRGB p;
      p.x = it_cloud->x;
      p.y = it_cloud->y;
      p.z= it_cloud->z;
      //p.r = colormap[label][color_r];
      //p.g = colormap[label][color_g];
      //p.b = colormap[label][color_b];

      p.b = colormap[label][0];
      p.g = colormap[label][1];
      p.r = colormap[label][2];
      cloud_final->push_back(p);
    }
  }
  return cloud_final;
}


void ObjectDataBase::AddObject(Object3D* newObject)
{
  std::unique_lock<mutex> lock(dataMutex);

  DBSize ++;
  //if database empty
  newObject->object_id = DBSize;
  if(mvpObjects.empty())
  {
    
    mvpObjects.push_back(newObject);
    return;
  }
  //database has previous entries
  else
  {
    //find by class
    std::vector<std::vector<Object3D*>::iterator> similar_label;
    similar_label.clear();
    for( auto it_label =  mvpObjects.begin(); it_label != mvpObjects.end(); ++it_label)
    {
      //if label is similar collect
      //get dominant only, more than 25%
      std::set<int> setLabel;
      setLabel.clear();
      for(int i = 0; i < (*it_label)->mvLabels.size(); ++i)
      {
        setLabel.insert((*it_label)->mvLabels[i].label);
      }
      for(int j = 0; j < newObject->mvLabels.size(); ++j)
      {
        setLabel.insert(newObject->mvLabels[j].label);
      }
      //if there is a common label
      if(setLabel.size() < ((*it_label)->mvLabels.size() + newObject->mvLabels.size()))
      {
        similar_label.push_back(it_label);
      }
      //if(newObject->mvLabels.front().label == (*it_label)->mvLabels.front().label)
      //{
      //  similar_label.push_back(it_label);
      //}
    }
      //if label is unique
    if(similar_label.empty())
    {
      mvpObjects.push_back(newObject);
      return;
    }
    else
    {
      //get closest object
      vector<pair<float,int>> bestDistances;
      bestDistances.clear();

      Eigen::Vector3f centNew = newObject->centroid;
      for(auto k = 0; k<similar_label.size(); ++k)
      {
        pair<float,int> bestEntry;
        Eigen::Vector3f centCand = (*similar_label[k])->centroid;
        float dist = pow((centCand[0]-centNew[0]),2) + pow((centCand[1]-centNew[1]),2) + pow((centCand[2]-centNew[2]),2);
        bestEntry.first = dist;
        bestEntry.second = k;
        bestDistances.push_back(bestEntry);
      }
      sort(bestDistances.begin(), bestDistances.end());

      for (auto l = 0; l < bestDistances.size(); ++l)
      {
        float candR = (*similar_label[bestDistances[l].second])->searchRadius;
        if (bestDistances[l].first < candR && bestDistances[l].first < newObject->searchRadius)
        {
          (*similar_label[bestDistances[l].second])->Append(newObject,this->radius_factor);

          return;
        }
      }

      
      mvpObjects.push_back(newObject);
      return;

    }
    }
}

//by centroid
void ObjectDataBase::AddObjectByCentroid(Object3D* newObject)
{
  std::unique_lock<mutex> lock(dataMutex);

  DBSize ++;
  //if database empty
  newObject->object_id = DBSize;
  if(mvpObjects.empty())
  {
    
    mvpObjects.push_back(newObject);
    return;
  }
  //database has previous entries
  else
  {
    //find by class
    std::vector<std::vector<Object3D*>::iterator> similar_label;
    vector<pair<float,int>> bestDistances;
    bestDistances.clear();
    similar_label.clear();
    Eigen::Vector3f centNew = newObject->centroid;
    for( auto it_d =  mvpObjects.begin(); it_d != mvpObjects.end(); ++it_d)
    {
      //if label is similar collect
      //get dominant only, more than 25%
      //if label is unique
      //get closest object
      
      
        pair<float,int> bestEntry;
        Eigen::Vector3f centCand = (*it_d)->centroid;
        float dist = pow((centCand[0]-centNew[0]),2) + pow((centCand[1]-centNew[1]),2) + pow((centCand[2]-centNew[2]),2);
        bestEntry.first = dist;
        bestEntry.second = it_d - mvpObjects.begin();
        bestDistances.push_back(bestEntry);
      }
      sort(bestDistances.begin(), bestDistances.end());

      for (auto l = 0; l < bestDistances.size(); ++l)
      {
        float candR = (mvpObjects[bestDistances[l].second])->searchRadius;
        if (bestDistances[l].first < candR && bestDistances[l].first < newObject->searchRadius)
        {
          (mvpObjects[bestDistances[l].second])->Append(newObject,this->radius_factor);

          return;
        }
      }

      
      mvpObjects.push_back(newObject);
      return;

    
    }
}
  


void ObjectDataBase::Merge(std::vector<Object3D*>& mvpNew)
{
  this->lastReMerge++;
  if(mvpNew.empty())
  {
    return;
  }
  for (auto it=mvpNew.begin(); it!= mvpNew.end(); ++it)
  {
    if(this->useAddByCentroid == 1)
    {
      AddObjectByCentroid(*it);
    }
    else
    {
      AddObject(*it);
    }
    
  }

  PCL_Log << "Active Objects " << this->mvpObjects.size() << endl;
  if(useRemerge == 1)
  {
    if(lastReMerge > 5)
  {
    ReMerge();
  }
  }
  

  return;
}

void ObjectDataBase::ReMerge()
{
  auto mvpTemp = this->mvpObjects;
  this->mvpObjects.clear();
  for (auto it=mvpTemp.begin(); it!= mvpTemp.end(); ++it)
  {
    if(this->useAddByCentroid == 1)
    {
      AddObjectByCentroid(*it);
    }
    else
    {
      AddObject(*it);
    }
    
  }

  PCL_Log << "ReMerge Active Objects " << this->mvpObjects.size() << endl;
  this->lastReMerge = 0;

  return;
}

PointCloudMapping::PointCloudMapping(const string strSettingPath, SemSeg* pSemSeg): mpSemSeg(pSemSeg)
{
    PCL_Log << "Initialize" << endl;
    count_defective =0;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    this->fx = fSettings["Camera.fx"];
    this->fy = fSettings["Camera.fy"];
    this->cx = fSettings["Camera.cx"];
    this->cy = fSettings["Camera.cy"];
    this->classProb_thresh = fSettings["Class.Prob.Thresh"];
    
    this->resolution = fSettings["PCL.Resolution"];

    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );

    this->view_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>( );    
    this->color_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>( );
    this->label_cloud = boost::make_shared< PointCloud >( );

    this->useCloudViewer = fSettings["Use.CloudViewer"];
    this->usePCLMapping = fSettings["PCL.MappingEnabled"];
    this->pclMinDepth = fSettings["PCL.Min.Depth"];
    this->pclMaxDepth = fSettings["PCL.Max.Depth"];
    this->labelCloudInterval = fSettings["PCL.Label.Interval"];
    this->useSimple = fSettings["PCL.Use.Simple"];
    this->removeBG = fSettings["PCL.Remove.BG"];
    this->mvpFrameData.clear();

    mpOD = new ObjectDataBase(strSettingPath);

    if(usePCLMapping)
    {
      viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
    }
    
    if(useCloudViewer)
    {
      cloudThread = make_shared<thread>( bind(&PointCloudMapping::cloud_viewer, this ) );
    }
}

bool PointCloudMapping::isBackground(int _label)
{
  //wall , floor, window
  if (_label == 1 || _label == 4 || _label == 9)
  {
    return true;
  }

  else
  {
    return false;
  } 
}

void PointCloudMapping::SaveColorCloud()
{
  if(color_cloud->size() > 0)
  {
    pcl::io::savePCDFileASCII("results/color_cloud.pcd",*color_cloud);
  }
  
  return;
}

void PointCloudMapping::SaveLabelCloud()
{
  voxel.setInputCloud(this->label_cloud);
  voxel.setDownsampleAllData(true);
  voxel.filter(*(this->label_cloud));
  if(label_cloud->size() > 0)
  {
    pcl::io::savePCDFileASCII("results/label_cloud.pcd",*label_cloud);
  }
  PointCloud::Ptr label_cloud_no_person(new (PointCloud));

  for(auto it = label_cloud->begin(); it != label_cloud->end(); it ++)
  {
    if(it->label != 13)
    {
      label_cloud_no_person->push_back(*it);
    }
  }
  if(label_cloud_no_person->size() > 0)
  {
    pcl::io::savePCDFileASCII("results/label_cloud_filtered.pcd",*label_cloud_no_person);
  }


  PointCloud::Ptr label_cloud_chair(new (PointCloud));

  for(auto it = label_cloud->begin(); it != label_cloud->end(); it ++)
  {
    if(it->label == 20)
    {
      label_cloud_chair->push_back(*it);
    }
  }
  if(label_cloud_chair->size() > 0)
  {
    pcl::io::savePCDFileASCII("results/label_cloud_filtered_chair.pcd",*label_cloud_chair);
  }
  return;


}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    SaveColorCloud();
    if (useSimple ==1)
    {
      SaveLabelCloud();
    }

    mpOD->RenderAndSave();
    mpOD->RenderAndSaveFilter(20, "results/chair.pcd");
    mpOD->RenderAndSaveFilter(16, "results/table.pcd");
    mpOD->RenderAndSaveFilter(75, "results/computer.pcd");
    mpOD->RenderAndSaveFilter(116, "results/bag.pcd");
    mpOD->RenderAndSaveFilter(5, "results/tree.pcd");
    mpOD->RenderAndSaveFilter(18, "results/plant.pcd");


    if(useCloudViewer)
    {
      viewerThread->join();
    }

    if(usePCLMapping)
    {
      cloudThread->join();
    }
    
}

void PointCloudMapping::cloud_viewer(){
  
  pcl::visualization::PCLVisualizer::Ptr viewCloud (new pcl::visualization::PCLVisualizer ("Point_Cloud_Viewer"));
  viewCloud->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(this->view_cloud);
  viewCloud->addPointCloud<pcl::PointXYZRGB> (this->view_cloud, rgb, "sample cloud");
  //pcl::visualization::PointCloudColorHandlerGenericField<PointT> point_cloud_color_handler(this->view_cloud, "label");
  //viewCloud->addPointCloud<PointT> (this->view_cloud, point_cloud_color_handler, "sample cloud");
  
  viewCloud->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewCloud->addCoordinateSystem (1.0);
  viewCloud->initCameraParameters ();
  
  //auto viewCloud = rgbVis(this->view_cloud,"VisCloud");
  while(!shutDownFlag)
  {
    viewCloud->removeAllPointClouds();
    //pcl::visualization::PointCloudColorHandlerGenericField<PointT> point_cloud_color_handler(this->view_cloud, "label");
    //viewCloud->addPointCloud<PointT> (this->view_cloud, point_cloud_color_handler, "sample cloud");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(this->view_cloud);
    viewCloud->addPointCloud<pcl::PointXYZRGB> (this->view_cloud, rgb, "sample cloud");

    viewCloud->spinOnce (100);

  }
  
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& segColor, cv::Mat& segLabel, cv::Mat& segProb, cv::Mat& dynMask)
{
    {
        unique_lock<mutex> lck(keyframeMutex);
        keyframes.push_back(kf);
        colorImgs.push_back(color);
        depthImgs.push_back(depth);
        mvSegColor.push_back(segColor);
        mvSegLabel.push_back(segLabel);
        mvSegProb.push_back(segProb);
        mvDynMask.push_back(dynMask); 
        keyFrameUpdated.notify_one();
    }

}

void PointCloudMapping::insertKeyFrame( FrameData* pFD )
{
  
  unique_lock<mutex> lck(keyframeMutex);
  this->mvpFrameData.push_back(pFD);
  keyFrameUpdated.notify_one();
}


//Currently inuse
PointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& SegLabel, cv::Mat& depth)
{   
    
    //PointCloud::Ptr tmp( new PointCloud() );
    
    if (depth.rows > 480 || depth.rows < 0  || depth.cols > 640 || depth.cols < 0 || depth.empty())
    {
        std::cout <<"Depth map has abnormal dimensions" << std::endl;
        count_defective++;
        std::cout << "Defective Depth Frames: " << count_defective << std::endl;
        exit(-1);
    }
   
   PointCloud::Ptr cloud(new (PointCloud));
   //pcl::PointIndicesPtr cloud_ind(new (pcl::PointIndices));
   //pcl::PointIndices::Ptr cloud_ind (new pcl::PointIndices());
  std::vector<int> cloud_ind;
  cloud_ind.clear();
   

      //std::cout << "Cols index = " << n << std::endl;
    cloud->resize(depth.rows * depth.cols);
    cloud->width    =  depth.cols;  
    cloud->height   =  depth.rows;// Ordered point cloud
    cloud->is_dense =  false;// Non-dense point clouds, there will be bad points, may contain values such as inf/NaN   
    #pragma omp parallel for
    for ( int m=0; m<(depth.rows); m+=1 )// Each line /+3
    {
      for ( int n=0; n<(depth.cols); n+=1 )//Each column
      {
          if (SegLabel.ptr<uchar>(m)[n] == 13)
          {
              continue;
          }
          float d = depth.ptr<float>(m)[n];//Depth m is the unit. Keep points within 0~2m.
          //if (d < 0.01 || d>2.0) //Camera measurement range 0.5~6m
          //std::cout << "Depth is " << d << std::endl;
          if (d < 0.1 || d>4.0) // Camera measurement range 0.5~6m
             continue;
          //float z = d;
          float y = ( m - cy) * d / fy;
          //if(y<-3.0 || y>3.0) 
          //{
          //    continue;// Retain the vertical direction - points in the range of 3 to 3 m
          //}
           
          int ind = m * depth.cols + n;// Total index

          {
            cloud->points[ind].z = d;
            cloud->points[ind].x = ( n - cx) * d / fx;
            cloud->points[ind].y = y;

            int label = SegLabel.ptr<uchar>(m)[n];
            cloud->points[ind].label = label;
            cloud_ind.push_back(ind);

          }
          

          //std::cout << "X is " << cloud->points[ind].x << std::endl;
          //std::cout << "Y is " << cloud->points[ind].y << std::endl;
          //std::cout << "Z is " << cloud->points[ind].z << std::endl;
          //cloud->points[ind].b = color.ptr<uchar>(m)[n*3+0];// Point color=====
          //cloud->points[ind].g = color.ptr<uchar>(m)[n*3+1];
          //cloud->points[ind].r = color.ptr<uchar>(m)[n*3+2];

          //cloud->points[ind].b = 200;// Point color=====
          //cloud->points[ind].g = 200;
          //cloud->points[ind].r = 200;
          //int label = SegLabel.ptr<uchar>(m)[n];
          //cloud->points[ind].label = label;
          //listLabel.insert(cloud->points[ind].label);
          //if (SegLabel.ptr<uchar>(m)[n]!=13)
          //{
          //  seg_pcl_masked->points.push_back(cloud->points[ind]);
          //}
      }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr tmp(new PointCloud);
    pcl::transformPointCloud( *cloud, *tmp, T.inverse().matrix());
    tmp->is_dense = false;

    PCL_Log << "Generate point cloud for KF mnId " << kf->mnId << ", size= " << cloud->points.size() << std::endl;
    //std::ostringstream ss;
    //ss<< "/home/richard/orbslam_thesis/pointcloud/PointCloud" << kf->mnId << ".pcd";
    //ClusterPCL(tmp);


    //pcl::io::savePCDFileASCII(ss.str(),*tmp);
    return tmp;
}

void PointCloudMapping::generatePointCloud(FrameData* pFD)
{   
    
    //PointCloud::Ptr tmp( new PointCloud() );

    cv::Mat & depth = pFD->mDepth;
    cv::Mat & color = pFD->mRGB;
    cv::Mat & SegLabel = pFD->mSegLabelRemap;
    cv::Mat & SegProb = pFD->mSegProb;
    cv::Mat & DynMask = pFD->mDynMask;
    auto kf = pFD->mpKF;
    
    if (depth.rows > 480 || depth.rows < 0  || depth.cols > 640 || depth.cols < 0 || depth.empty())
    {
        std::cout <<"Depth map has abnormal dimensions" << std::endl;
        count_defective++;
        std::cout << "Defective Depth Frames: " << count_defective << std::endl;
        exit(-1);
    }

   PointCloud::Ptr cloud(new (PointCloud));

   //pcl::PointIndicesPtr cloud_ind(new (pcl::PointIndices));
   //pcl::PointIndices::Ptr cloud_ind (new pcl::PointIndices());
  
   

      //std::cout << "Cols index = " << n << std::endl;
    cloud->resize(depth.rows * depth.cols);
    cloud->width    =  depth.cols;  
    cloud->height   =  depth.rows;// Ordered point cloud
    cloud->is_dense =  false;// Non-dense point clouds, there will be bad points, may contain values such as inf/NaN   
//#pragma omp parallel for
    for ( int m=0; m<(depth.rows); m+=1 )// Each line /+3
    {
      for ( int n=0; n<(depth.cols); n+=1 )//Each column
      {
          
          int label = SegLabel.ptr<uchar>(m)[n];
          float prob = SegProb.ptr<uchar>(m)[n];
          int dyn = DynMask.ptr<uchar>(m)[n];

          if (dyn == 1)
          {
              label = 13;
              continue;

          }
          else if(prob<classProb_thresh)
          {
            //label = 0;
            continue;
          }

          if(this->removeBG == 1)
          {
            if(isBackground(label))
            {
              continue;
            }
          }
          
          //if (label == 1)
          //{
          //  continue;
          //}

          float d = depth.ptr<float>(m)[n];//Depth m is the unit. Keep points within 0~2m.
          //if (d < 0.01 || d>2.0) //Camera measurement range 0.5~6m
          //std::cout << "Depth is " << d << std::endl;
          if (d < this->pclMinDepth || d > this->pclMaxDepth) // Camera measurement range 0.5~6m
             continue;
          //float z = d;
          float y = ( m - cy) * d / fy;
          //if(y<-3.0 || y>3.0) 
          //{
          //    continue;// Retain the vertical direction - points in the range of 3 to 3 m
          //}
           
          int ind = m * depth.cols + n;// Total index

          {
            cloud->points[ind].z = d;
            cloud->points[ind].x = ( n - cx) * d / fx;
            cloud->points[ind].y = y;
            cloud->points[ind].label = label;
            //cloud_ind.push_back(ind);

          }
          
          
      }
    }

    boost::shared_ptr<vector<int> > cloud_indices (new vector<int> ()); 
    
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.5, 3.0);
    //pass.setFilterLimitsNegative (true);

    pass.filter (*cloud_indices);
    //cloud_inlier = pass.getIndices();
    //pcl::io::savePCDFileASCII("cloud_filtered.pcd",*cloud_filtered);


    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud_world(new PointCloud);
    

    pcl::transformPointCloud( *cloud, *cloud_world, T.inverse().matrix());
    
    cloud_world->is_dense = false;





    //std::cout << "Generate point cloud for KF mnId " << kf->mnId << ", size= " << cloud->points.size() << std::endl;
    //std::cout << "Cloud Filterd "  << cloud_filtered->size();
    //std::cout << "Cloud Indexed "  << cloud_indices->size() << std::endl;
    //std::cout << "Inliers "  << cloud_inlier->size();

    //pcl::ExtractIndices<PointT> extract;
    //extract.setInputCloud(cloud);
    //extract.setIndices(cloud_ind1);
    //extract.filter(*cloud_filter_index);
    //std::cout << "Cloud Filtered Index "  << cloud_filtered_index->size() std::endl;

    
    //std::ostringstream ss;
    //ss<< "/home/richard/orbslam_thesis/pointcloud/PointCloud" << kf->mnId << ".pcd";
    //3d objects for this scene
    std::vector<Object3D*> mvpObject3D;
    mvpObject3D.clear();
    mpOD->ClusterPCL(cloud_world,cloud_indices,pFD,mvpObject3D);

    mpOD->Merge(mvpObject3D);


    //pcl::io::savePCDFileASCII(ss.str(),*tmp);s
    return;
}

void PointCloudMapping::generateLabelPointCloud(FrameData* pFD)
{   
    
    //PointCloud::Ptr tmp( new PointCloud() );

    cv::Mat & depth = pFD->mDepth;
    cv::Mat & color = pFD->mRGB;
    cv::Mat & SegLabel = pFD->mSegLabelRemap;
    cv::Mat & SegProb = pFD->mSegProb;
    cv::Mat & DynMask = pFD->mDynMask;
    auto kf = pFD->mpKF;
    
    if (depth.rows > 480 || depth.rows < 0  || depth.cols > 640 || depth.cols < 0 || depth.empty())
    {
        std::cout <<"Depth map has abnormal dimensions" << std::endl;
        count_defective++;
        std::cout << "Defective Depth Frames: " << count_defective << std::endl;
        exit(-1);
    }


   PointCloud::Ptr label_cloud_new(new (PointCloud));
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_color_cloud(new (pcl::PointCloud<pcl::PointXYZRGB>));

    for ( int m=0; m<(depth.rows); m+=1 )// 
    {
      for ( int n=0; n<(depth.cols); n+=1 )//Each column
      {
          
          int label = SegLabel.ptr<uchar>(m)[n];
          float prob = SegProb.ptr<uchar>(m)[n];
          int dyn = DynMask.ptr<uchar>(m)[n];

          //if (dyn == 1)
          //{
          //    label = 13;
          //    continue;
          //}
          ////else if(prob<classProb_thresh)
          //{
          //  //label = 0;
          //  continue;
          //}
          //if (label == 1)
          //{
          //  continue;
          //}

          float d = depth.ptr<float>(m)[n];//Depth m is the unit. Keep points within 0~2m.
          //if (d < 0.01 || d>2.0) //Camera measurement range 0.5~6m
          //std::cout << "Depth is " << d << std::endl;
          if (d < this->pclMinDepth || d > this->pclMaxDepth) // Camera measurement range 0.5~6m
             continue;
          //float z = d;
          float y = ( m - cy) * d / fy;
          //if(y<-3.0 || y>3.0) 
          //{
          //    continue;// Retain the vertical direction - points in the range of 3 to 3 m
          //}
           
          int ind = m * depth.cols + n;// Total index

          
          PointT p;
          p.x = ( n - cx) * d / fx;;
          p.y = y;
          p.z = d;
          p.label = label;

          //label_cloud_new->push_back(p);


          pcl::PointXYZRGB p_color;
          p_color.x = ( n - cx) * d / fx;;
          p_color.y = y;
          p_color.z = d;
          p_color.r = color.ptr<uchar>(m)[n*3+0];
          p_color.g = color.ptr<uchar>(m)[n*3+1];
          p_color.b = color.ptr<uchar>(m)[n*3+2];
          //temp_color_cloud->push_back(p_color);

          if (dyn == 1)
          {
              label = 13;
              continue;
          }
          else
          {
            label_cloud_new->push_back(p);
            temp_color_cloud->push_back(p_color);
          }
      }
    }

    
    countLabelCloud++;

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr label_cloud_world(new PointCloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_color_cloud_world(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::transformPointCloud( *label_cloud_new, *label_cloud_world, T.inverse().matrix());
    pcl::transformPointCloud( *temp_color_cloud, *temp_color_cloud_world, T.inverse().matrix());

    label_cloud_world->is_dense = false;
    temp_color_cloud_world->is_dense = false;

    *(this->label_cloud) += *label_cloud_world;
    *(this->color_cloud) += *temp_color_cloud_world;
    pcl::VoxelGrid<pcl::PointXYZRGB> vg_color;
    vg_color.setInputCloud (color_cloud);
    vg_color.setLeafSize (this->resolution, this->resolution, this->resolution);
    vg_color.filter(*color_cloud);



    if ((countLabelCloud % labelCloudInterval) == 0)
    {
        voxel.setInputCloud(this->label_cloud);
        voxel.setDownsampleAllData(true);
        voxel.filter(*(this->label_cloud));
    }
    return;
}





void PointCloudMapping::viewer()
{
    //cv::namedWindow("DepthWindow");
    

    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = mvpFrameData.size();
        }
        
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            if(mvpFrameData[i]->mpKF->isBad() && i >3)
            {
                continue;
            }
            //cv::imshow("DepthWindow",mvSegColor[i]);
            //PointCloud::Ptr p = generatePointCloud( keyframes[i], mvSegLabel[i], depthImgs[i] );
            generatePointCloud(mvpFrameData[i]);
            if (useSimple ==1)
            {
              //generateLabelPointCloud(mvpFrameData[i]);
            }
            generateLabelPointCloud(mvpFrameData[i]);
            
            //std::cout << "P size" << p->size() << std::endl;
            //*globalMap = *globalMap +  *p;
            //this->view_cloud.swap(p);
            if(useCloudViewer)
            {
              auto objectCloud = mpOD->RenderLatest();
              this->view_cloud.swap(objectCloud);
            }
            
            
            
        }

        
        
        //PointCloud::Ptr tmp2(new PointCloud());
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZRGB>)
        //PointCloud::Ptr tmp2;

        //cout << "show global map, size=" << globalMap->points.size() << endl;
        
        //voxel.setInputCloud( globalMap );
        //voxel.filter( *globalMap );
        
        
        
        //globalMap->swap( *tmp2 );
        
        //viewer.showCloud( globalMap );
        //cout << "show global map, size=" << globalMap->points.size() << endl;
        //pcl::io::savePCDFileASCII ("global_map.pcd", *globalMap);
        
        PCL_Log << "finished processing until Keyframe index: " << N << endl;
        lastKeyframeSize = N;
        
    }
}


bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  //if (point_a.label == 20 && point_b.label != 20)
  //{
  //  return false;
  //}
  //if (point_b.label == 20 && point_a.label != 20)
  //{
  //  return false;
  //}
  //if (point_a.label == 1 || point_b.label ==1)
  //{
  //  return false;
  //}
  //{
  //  return false;
  //}
  //if (point_b.label == 20 && point_a.label != 20)
  //{
  //  return false;
  ////}
  //if (squared_distance < 0.025)
  //{
  //  return true;
  //}
  if (squared_distance < 0.03)
  {
    if (point_a.label == point_b.label)
      return (true);

    float normal_sim= std::abs(point_a_normal.dot (point_b_normal));
    if ( normal_sim > 0.20 && normal_sim < 0.8)
    {
        //if (point_a.label == 13)
        //    point_b.label = point_a.label;
        return (true);
    }
      
  }
  else
  {
    //if (point_a.label == point_b.label)
     // return (true);
     return (false);
  }
  return (false);
}


void ObjectDataBase::ClusterPCL(PointCloud::Ptr cloud_in, pcl::IndicesPtr pIndices, FrameData* pFD, std::vector<Object3D*>& mvpObject3D)
{
  // Data containers used

  pcl::IndicesPtr indices_cloud_voxel(new std::vector<int>());
  pcl::PointIndicesPtr indices_cloud_voxel_outliers(new pcl::PointIndices);
  PointCloud::Ptr vg_cloud_out(new PointCloud);
  PointCloudN::Ptr cloud_with_normals (new PointCloudN);
  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
  pcl::search::KdTree<PointT>::Ptr search_tree (new pcl::search::KdTree<PointT>);
  pcl::console::TicToc tt;


  // Downsample the cloud using a Voxel Grid class
  //std::cerr << "Downsampling...\n", tt.tic ();
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud (cloud_in);
  vg.setIndices(pIndices);
  vg.setLeafSize (0.01, 0.01, 0.01);
  vg.setDownsampleAllData (true);
  vg.filter(*vg_cloud_out);
  vg.getRemovedIndices(*indices_cloud_voxel_outliers);
  
  //indices_cloud_voxel = vg.getIndices();

  //std::cerr << ">> Done: " << tt.toc () << " ms, " << indices_cloud_voxel->size () << " points\n" << std::endl;
  //std::cout << "Voxel Filter Inliers " << indices_cloud_voxel->size() << std::endl;
  //std::cout << "Removed Indices " << indices_cloud_voxel_outliers->indices.size() << std::endl;

  // Set up a Normal Estimation class and merge data in cloud_with_normals
  //std::cerr << "Computing normals...\n", tt.tic ();
  pcl::copyPointCloud (*vg_cloud_out, *cloud_with_normals);
  pcl::NormalEstimation<PointT, PointTN> ne;
  //ne.setIndices(indices_cloud_voxel);
  ne.setInputCloud (vg_cloud_out);
  ne.setSearchMethod (search_tree);
  ne.setRadiusSearch (0.03);
  ne.compute (*cloud_with_normals);

  //std::cerr << ">> Done: " << tt.toc () << " ms\n";
  //std::cout << "Cloud with Normals: " << cloud_with_normals->size() <<std::endl;

  // Set up a Conditional Euclidean Clustering class
  //std::cerr << "Segmenting to clusters...\n", tt.tic ();
  pcl::ConditionalEuclideanClustering<PointTN> cec;
  cec.setInputCloud (cloud_with_normals);
  //cec.setIndices(indices_cloud_voxel);
  cec.setConditionFunction (&customRegionGrowing);
  cec.setClusterTolerance (0.02);
  cec.setMinClusterSize (100);
  cec.setMaxClusterSize (50000);
  cec.segment (*clusters);
  //cec.getRemovedClusters (small_clusters, large_clusters);
  //std::cerr << ">> Done: " << tt.toc () << " ms\n";

  // Using the intensity channel for lazy visualization of the output
  
  //for (int i = 0; i < small_clusters->size (); ++i)
  //  for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
  //    cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
  //for (int i = 0; i < large_clusters->size (); ++i)
  //  for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
  //    cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;
  //for (int i = 0; i < clusters->size (); ++i)
  //{
  //  int label = rand () % 8;
  //  for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
  //    cloud_out->points[(*clusters)[i].indices[j]].intensity = label;
  //}
  PCL_Log << "The Cloud Has N Clusters" << clusters->size () <<std::endl;


  //std::vector<Object3D*> mvpObject3D;
  //std::vector<cluster_label_stats> mvClusterStats;
  
  //for each cluster
  for (int i = 0; i < clusters->size(); ++i)
  {
      //points[(*clusters)[i].indices[j]].intensity = label;
      //std::cout << "Class " << vg_cloud_out->points[(*clusters)[i].indices[0]].label << " with size " << (*clusters)[i].indices.size() << std::endl;

      Object3D* pTempObject(new Object3D());
    
    //for each index in cluster
    for (int j = 0; j < (*clusters)[i].indices.size(); j++)
    {
      //int _index = (*clusters)[i].indices[j];
     pTempObject->pCloud->push_back(vg_cloud_out->points[(*clusters)[i].indices[j]]);
      
      //pcl::ExtractIndices<PointTN> ext;
      //ext.setInputCloud(cloud_with_normals);
      //ext.setIndices(clusters[i]);
      //ext.filter(*(pTempObject->pCloud));
    }
  
    pTempObject->ComputeAll(this->radius_factor);
    //log cluster
    //std::cout <<pTempObject->label << " With " << pTempObject->n_points << std::endl;
    mvpObject3D.push_back(pTempObject);
  }
      

      

      
      

  // Save the output point cloud
  //std::cerr << "Saving...\n", tt.tic ();
  //pcl::io::savePCDFile ("output_seg.pcd", *cloud_out);
  //std::cerr << ">> Done: " << tt.toc () << " ms\n";

  return;
}

