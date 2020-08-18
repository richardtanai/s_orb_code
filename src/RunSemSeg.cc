/* This file is part of ORB-SLAM2-SSD-Semantic.
* 2d target detection
*/
#include "RunSemSeg.h"

RunSemSeg::RunSemSeg()
{
   mSemSeg = new(SemSeg);
   mvKeyframes.clear();// Keyframe array clear ===
   colorImgs.clear();  // Color image====
   mRunThread = make_shared<thread>( bind(&RunSemSeg::Run, this ) );// Visualization thread shared pointer binding RunDetect::Run() function
}
RunSemSeg::~RunSemSeg()
{
   delete mSemSeg;
}
// Insert a keyframe in the map =======tracker's build keyframe function execution===================
void RunSemSeg::insertKFColorImg(KeyFrame* kf, cv::Mat color)
{  
   /*
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(colorImgMutex); // Lock keyframes
    colorImgs.push_back( color.clone() );  // Image array add an image deep copy
    mvKeyframes.push_back(kf);

    colorImgUpdated.notify_one();         
    */
   std::cout << "RunSemSeg insertKFCalled" <<std::endl;

    // Data update thread Unblocks the target image for color image ===
}

void RunSemSeg::Run(void)
{
 while(1)
 {
    {
       unique_lock<mutex> lck_colorImgUpdated( colorImgMutex); // Key frame update lock
       colorImgUpdated.wait( lck_colorImgUpdated );// Block keyframe update lock
       // Need to wait for the insertKeyFrame() function to complete after adding the keyframe, execute the following!!
       std::cout << "RunSemRun Thread Notify one" <<std::endl;

    }
    // Color image ===
    size_t N=0;
    {
       unique_lock<mutex> lck( colorImgMutex );// Key frame lock
       N = colorImgs.size();                   // The number of key frames currently saved
    }  
    for ( size_t i=lastKeyframeSize; i<N ; i++ )// Then the place where the last processing started
    {
        /*
        std::vector<Object> vobject;
        mSemSeg->Run(colorImgs[i], vobject);
        if(vobject.size()>0)
        {  
           std::cout << "detect : " << vobject.size() << " uums obj" << std::endl;
           for(unsigned int j =0; j<vobject.size(); j++)
           {
               unique_lock<mutex> lckObj(mvKeyframesMutex); // 2d test results locked
               mvKeyframes[i]->mvObject.push_back(vobject[j]);// Deposit into the target detection result library====
           }
       }*/
       //Segmented2D seg2d;
       std::cout << "Performing Inferecne" << std::endl;
       //mSemSeg->Run(colorImgs[i], mvKeyframes[i]->mSeg2D);
       
    }

    lastKeyframeSize = N;
 }
    
}
