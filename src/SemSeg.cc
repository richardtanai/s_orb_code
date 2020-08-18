#include "SemSeg.h"
//#include "extra_func.h"

//class constructor
using namespace cv;
using namespace std;

SemSeg::SemSeg():warmedUp(false)
{

    std::cout << "Starting SemSeg" << std::endl;
    std::string strSettingsFile = "/home/richard/orbslam_thesis/Examples/RGB-D/TUM3.yaml";
    cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
       exit(-1);
    }

    //pointcloudmapping parameter 
    //float resolution = fsSettings["PointCloudMapping.Resolution"];
    
    //path_graph = "/home/richard/tensorflow_testing/model_zoo/xception/frozen_inference_graph.pb";

    path_graph = (std::string)fsSettings["FrozenGraphPath"];
    output_node_label = (std::string)fsSettings["LabelOutputNode"];
    output_node_prob = (std::string)fsSettings["ProbOutputNode"];
    float gpu_fraction = fsSettings["TF.GPU.Fraction"];
    this->isRGB = fsSettings["Camera.RGB"];

    this->useSegDepthTh = fsSettings["Use.SegDepthTh"];
    this->useSegProbTh = fsSettings["Use.SegProbTh"];

    this->SegMinDepth = fsSettings["Seg.Min.Depth"];
    this->SegMaxDepth = fsSettings["Seg.Max.Depth"];
    this->SegMinProb =  fsSettings["Seg.Min.Prob"];

    this->mDilation_size = fsSettings["DynMask.Dilation"];

    std::cout << "Loading Graph from " << path_graph << std::endl;
    std::cout << "output_node_label set to  " << output_node_prob << std::endl;
    std::cout << "output_node_labe set to" << output_node_label << std::endl;





    tensorflow::SessionOptions options;
    tensorflow::Status status;
    //options.config.mutable_gpu_options()->set_allow_growth(true);
    //options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(gpu_fraction);
    //status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    tensorflow::NewSession(options, &sess); 	
    if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
    	//return 1;
    }   
    //define graph
    tensorflow::GraphDef graph_def; 	
    //load frozen graph into the graph  	
    status = ReadBinaryProto(tensorflow::Env::Default(),path_graph, &graph_def); 	//check if load successful  	
    if (!status.ok()) {
      	std::cout << status.ToString() << "\n";
       	//return 1;
    }
    // Add the graph to the session
    status = sess->Create(graph_def);
    if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
       //return 1;
    }

    clock_t t_warmup = clock();


    std::cout << "Warming Up Semantic Segmentation" << std::endl;


    tensorflow::Tensor emptyTensor(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});


    //if MobileNet ResizeBilinear_2:0, if Xception ResizeBilinear_3:0


    status = sess->Run({{"ImageTensor:0", emptyTensor}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    std::cout << "Warm Up Completed in " << ((float)(clock()-t_warmup))/CLOCKS_PER_SEC<< std::endl;
    
    warmedUp = true;
}


SemSeg::SemSeg(const std::string strSettingsFile):warmedUp(false)
{
    cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
       exit(-1);
    }
    //pointcloudmapping parameter 
    //float resolution = fsSettings["PointCloudMapping.Resolution"];
    
    //path_graph = "/home/richard/tensorflow_testing/model_zoo/xception/frozen_inference_graph.pb";

    path_graph = (std::string)fsSettings["FrozenGraphPath"];
    output_node_label = (std::string)fsSettings["LabelOutputNode"];
    output_node_prob = (std::string)fsSettings["ProbOutputNode"];
    float gpu_fraction = fsSettings["TF.GPU.Fraction"];
    this->isRGB = fsSettings["Camera.RGB"];

    this->useSegDepthTh = fsSettings["Use.SegDepthTh"];
    this->useSegProbTh = fsSettings["Use.SegProbTh"];

    this->SegMinDepth = fsSettings["Seg.Min.Depth"];
    this->SegMaxDepth = fsSettings["Seg.Max.Depth"];
    this->SegMinProb =  fsSettings["Seg.Min.Prob"];

    this->mDilation_size = fsSettings["DynMask.Dilation"];

    std::cout << "Loading Graph from " << path_graph << std::endl;
    std::cout << "output_node_label set to  " << output_node_prob << std::endl;
    std::cout << "output_node_labe set to" << output_node_label << std::endl;



    tensorflow::SessionOptions options;
    tensorflow::Status status;
    //options.config.mutable_gpu_options()->set_allow_growth(true);
    //options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(gpu_fraction);
    //status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    tensorflow::NewSession(options, &sess); 	
    if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
    	//return 1;
    }   
    //define graph
    tensorflow::GraphDef graph_def; 	
    //load frozen graph into the graph  	
    status = ReadBinaryProto(tensorflow::Env::Default(),path_graph, &graph_def); 	//check if load successful  	
    if (!status.ok()) {
      	std::cout << status.ToString() << "\n";
       	//return 1;
    }
    // Add the graph to the session
    status = sess->Create(graph_def);
    if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
       //return 1;
    }

    clock_t t_warmup = clock();


    std::cout << "Warming Up Semantic Segmentation" << std::endl;


    tensorflow::Tensor emptyTensor(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});


    //if MobileNet ResizeBilinear_2:0, if Xception ResizeBilinear_3:0


    status = sess->Run({{"ImageTensor:0", emptyTensor}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    std::cout << "Warm Up Completed in " << ((float)(t_warmup-clock()))/CLOCKS_PER_SEC<< std::endl;
    
    warmedUp = true;
}

SemSeg::~SemSeg()
{
    sess->Close();
}

bool SemSeg::isWarmedUp(){
    return warmedUp;
}

void SemSeg::Run(const cv::Mat& cvImg, Segmented2D& results)// bgr image, output will be written to results class
{
    tensorflow::Status status;
    //template
    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    cv::Mat cvImg_rgb;
    //cv::cvtColor(cvImg, cvImg, COLOR_BGR2RGB); 
    //convert image to tensor (513,385)
    
    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});
    cv::Mat cvImg_resized(tensor_image_height, tensor_image_width, CV_8UC3, imgTensorWithSharedData.flat<uint8_t>().data());
    cv::resize(cvImg, cvImg_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);
    
    if(!isRGB)
    {
    cv::cvtColor(cvImg_resized, cvImg_resized, COLOR_BGR2RGB); 
    }
	//setup input tensor

	// Output
	//std::vector<tensorflow::Tensor> outputs;
    //std::cout << "Start Inference" << std::endl;
    clock_t t_inference;
    t_inference = clock();
    status = sess->Run({{"ImageTensor:0", imgTensorWithSharedData}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    if (!status.ok()) {
  	    std::cout << status.ToString() << "\n";
        return;
    }
    //std::cout << ((float)(t_inference))/CLOCKS_PER_SEC << std::endl;


    //process output
    //std::cout << "Make Output Matrices" << std::endl;
    
    cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1);
    cv::Mat out_prob(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_32F);
    cv::Mat dynamic_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);


    
    //out_image.release();
    //out_prob.release();

    int value;
    float val_prob;
    //std::cout << "Get Tensor Values" << std::endl;
    //std::cout << "Out[0] size: "<< outputs[0].shape() << std::endl;
    //std::cout << "Out[1] size: "<< outputs[1].shape() << std::endl;
    int output_img_n0 = outputs[0].shape().dim_size(0);
    int output_img_h0 = outputs[0].shape().dim_size(1);
    int output_img_w0 = outputs[0].shape().dim_size(2);

    int output1_img_n0 = outputs[1].shape().dim_size(0);
    int output1_img_h0 = outputs[1].shape().dim_size(1);
    int output1_img_w0 = outputs[1].shape().dim_size(2);
    int output1_img_c0 = outputs[1].shape().dim_size(3);
    //std::cout << "Check1" << std::endl;

    for (unsigned int ni = 0; ni < output_img_n0; ni++)
    {
        #pragma omp parallel for
        for (unsigned int hi = 0; hi < output_img_h0; hi++)
        {
            for (unsigned int wi = 0; wi < output_img_w0; wi++)
            {
                {
                    // Get vaule through .flat()
                    //std::cout << "Check2" << std::endl;
                    unsigned int offset = ni * output_img_h0 * output_img_w0  +
                                hi * output_img_w0 +
                                wi;
                    //std::cout << "Check3" << std::endl;
                    value = outputs[0].flat<long long int>()(offset);
                    //std::cout << "Check4" << std::endl;
                    unsigned int offset_prob = ni * output_img_h0 * output_img_w0 * output1_img_c0 +
                                hi * output_img_w0 * output1_img_c0 +
                                wi * output1_img_c0 +
                                value;
                    //std::cout << "Check5" << std::endl;
                    val_prob = outputs[1].flat<float>()(offset_prob);
                    //std::cout << "Check6" << std::endl;
                    out_image.at<uchar>(int(hi),int(wi)) = value;
                    //std::cout << "Check7" << std::endl;
                    out_prob.at<float>(int(hi),int(wi)) = (val_prob+15)/40; //normalized
                    //std::cout << "Check8" << std::endl;

                }
            }
        }
    }

    //results.Prediction = out_image;
    //results.Prob = out_prob;
    //std::cout << "Make Share Matrix" << std::endl;
    resized_output.release();
    resized_output_prob.release();

    resized_output.create(inputHeight,inputWidth,CV_8U);
    resized_output_prob.create(inputHeight,inputWidth,CV_8U);


    //cv::Mat resized_output_prob(inputHeight,inputWidth,CV_32F);
    //cv::imwrite("/home/richard/tensorflow_testing/images/resize_before", color_image, compression_params);
    //cv::resize(resized_output, out_image, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(cvImage_rgb, cvImg, cv::Size(target_width,target_height), 0, 0, INTER_LINEAR);
    //std::cout << "Resize" << std::endl;
    cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(out_prob, resized_output_prob, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);

    results.Prediction = resized_output;
    results.Prob = resized_output_prob;

    
    //results.DynamicMask = dynamic_mask
    //cv::imwrite("/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/one.png",results.Prediction,mcompression_params);






    // Create a window for display.
    //namedWindow( "Semantic", WINDOW_AUTOSIZE );
    //imshow( "Semantic", resized_output);
    //results.classes;
    //std::cout << "Writing to" << ss.str() << std::endl;
    image_number++;
}

void SemSeg::RunColor(const cv::Mat& cvImg, cv::Mat& resultImg)// bgr image, output will be written to results class
{
    //std::cout << "Start RunColor" << std::endl;
    tensorflow::Status status;

    
    //template
    cv::Mat cv_copy = cvImg;
    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    //std::string ty =  type2str( cvImg.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), cvImg.cols, cvImg.rows );
    cv::Mat cvImg_rgb;
    //cv::cvtColor(cvImg, cvImg, COLOR_BGR2RGB); 
    int tensor_image_width = 513;
    int tensor_image_height = 385;
    //convert image to tensor (513,385)
    
    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});
    cv::Mat cvImg_resized(tensor_image_height, tensor_image_width, CV_8UC3, imgTensorWithSharedData.flat<uint8_t>().data());
    //std::cout << "Before Color Resize" << std::endl;
    cv::resize(cvImg, cvImg_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);
    //std::cout << "After Resize" << std::endl;

    if(!isRGB)
    {
    cv::cvtColor(cvImg_resized, cvImg_resized, COLOR_BGR2RGB); 
    }
    //std::cout << "after color convert" << std::endl;

	//setup input tensor

	// Output
	
    //std::cout << "Start Inference" << std::endl;
    //clock_t t_inference;
    //t_inference = clock();
    status = sess->Run({{"ImageTensor:0", imgTensorWithSharedData}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    if (!status.ok()) {
  	    std::cout << status.ToString() << "\n";
        return;
    }
    //std::cout << ((float)(t_inference))/CLOCKS_PER_SEC << std::endl;


    //process output
    //std::cout << "Make Output Matrices" << std::endl;
    
    //cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1);
    cv::Mat color_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC3);
    //cv::Mat out_prob(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_32F);
    //cv::Mat dynamic_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);


    
    //out_image.release();
    //out_prob.release();

    int value;
    float val_prob;
    //std::cout << "Get Tensor Values" << std::endl;
    //std::cout << "Out[0] size: "<< outputs[0].shape() << std::endl;
    //std::cout << "Out[1] size: "<< outputs[1].shape() << std::endl;
    int output_img_n0 = outputs[0].shape().dim_size(0);
    int output_img_h0 = outputs[0].shape().dim_size(1);
    int output_img_w0 = outputs[0].shape().dim_size(2);

    int output1_img_n0 = outputs[1].shape().dim_size(0);
    int output1_img_h0 = outputs[1].shape().dim_size(1);
    int output1_img_w0 = outputs[1].shape().dim_size(2);
    int output1_img_c0 = outputs[1].shape().dim_size(3);
    //std::cout << "Check1" << std::endl;

    for (unsigned int ni = 0; ni < output_img_n0; ni++)
    {
        for (unsigned int hi = 0; hi < output_img_h0; hi++)
        {
            for (unsigned int wi = 0; wi < output_img_w0; wi++)
            {
                {
                    // Get vaule through .flat()
                    //std::cout << "Check2" << std::endl;
                    unsigned int offset = ni * output_img_h0 * output_img_w0  +
                                hi * output_img_w0 +
                                wi;
                    //std::cout << "Check3" << std::endl;
                    value = outputs[0].flat<long long int>()(offset);
                    //std::cout << "Check4" << std::endl;
                    //unsigned int offset_prob = ni * output_img_h0 * output_img_w0 * output1_img_c0 +
                    //            hi * output_img_w0 * output1_img_c0 +
                    //            wi * output1_img_c0 +
                    //            value;
                    //std::cout << "Check5" << std::endl;
                    //val_prob = outputs[1].flat<float>()(offset_prob);
                    //std::cout << "Check6" << std::endl;
                    //out_image.at<uchar>(int(hi),int(wi)) = value;
                    //std::cout << "Check7" << std::endl;
                    //out_prob.at<float>(int(hi),int(wi)) = (val_prob+15)/40; //normalized
                    //std::cout << "Check8" << std::endl;

                    color_mask.at<Vec3b>(Point(wi, hi))[0] = colormap[value][0];
                    color_mask.at<Vec3b>(Point(wi, hi))[1] = colormap[value][1];
                    color_mask.at<Vec3b>(Point(wi, hi))[2] = colormap[value][2];

                }
            }
        }
    }

    //results.Prediction = out_image;
    //results.Prob = out_prob;
    //std::cout << "Make Share Matrix" << std::endl;
    //resized_output.release();
    //resized_output_prob.release();
    resized_color_out.release();

    //resized_output.create(inputHeight,inputWidth,CV_8U);
    //resized_output_prob.create(inputHeight,inputWidth,CV_8U);
    resized_color_out.create(inputHeight,inputWidth,CV_8UC3);

    //cv::Mat resized_output_prob(inputHeight,inputWidth,CV_32F);
    //cv::imwrite("/home/richard/tensorflow_testing/images/resize_before", color_image, compression_params);
    //cv::resize(resized_output, out_image, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(cvImage_rgb, cvImg, cv::Size(target_width,target_height), 0, 0, INTER_LINEAR);
    //std::cout << "Resize" << std::endl;
    //cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(out_prob, resized_output_prob, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(color_mask, resized_color_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);

    resultImg = resized_color_out;
    //std::cout << " Inference Complete" << std::endl;
    //results.Prob = resized_output_prob;

    
    std::ostringstream ss;
    ss << "/home/richard/orbslam_thesis/images/seg/image" << image_number << ".png";
    cv::imwrite(ss.str(),resultImg,mcompression_params);

    std::ostringstream aa;
    aa << "/home/richard/orbslam_thesis/images/raw/image" << image_number << ".png";
    cv::imwrite(aa.str(),cv_copy,mcompression_params);

    //problem with this
    //std::ostringstream ss_seg;
    //ss_seg << "/home/richard/orbslam_thesis/images/seg/image" << image_number << ".png";
    //cv::imwrite(ss_seg.str(),cvImg,mcompression_params);

    //std::cout << " Write Complete" << std::endl;
    //results.DynamicMask = dynamic_mask
    //std::ostringstream ss;
    //ss << "/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/image" << image_number << ".png";
    //cv::imwrite("/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/one.png",results.Prediction,mcompression_params);
    //cv::imwrite(ss.str(),results.Prediction,mcompression_params);

    // Create a window for display.
    //namedWindow( "Semantic", WINDOW_AUTOSIZE );
    //imshow( "Semantic", resized_output);
    //results.classes;
    //std::cout << "Writing to" << ss.str() << std::endl;
    image_number++;
}

cv::Mat SemSeg::Show(const cv::Mat& cvImg, Segmented2D& results)//cvImg is bgr
{
        //label to color conversion
    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    cv::Mat color_mask(inputHeight,inputWidth,CV_8UC3);
    for (int iy = 0; iy < inputHeight; iy++ )
    {
            for (int ix = 0; ix < inputWidth; ix++ )
        {
            color_mask.at<Vec3b>(Point(ix, iy))[0] = colormap[results.Prediction.at<uchar>(iy,ix)][0];
            color_mask.at<Vec3b>(Point(ix, iy))[1] = colormap[results.Prediction.at<uchar>(iy,ix)][1];
            color_mask.at<Vec3b>(Point(ix, iy))[2] = colormap[results.Prediction.at<uchar>(iy,ix)][2];
        }
    }
    namedWindow( "Semantic", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Semantic", color_mask);

}

void SemSeg::RunColorAndMask(const cv::Mat& cvImg, cv::Mat& resultImg, cv::Mat& resultMask)// bgr image, output will be written to results class
{
   
    tensorflow::Status status;


    //template
    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    //std::string ty =  type2str( cvImg.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), cvImg.cols, cvImg.rows );
    cv::Mat cvImg_rgb;
    //cv::cvtColor(cvImg, cvImg, COLOR_BGR2RGB); 
    int tensor_image_width = 513;
    int tensor_image_height = 385;
    //convert image to tensor (513,385)
    
    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});
    cv::Mat cvImg_resized(tensor_image_height, tensor_image_width, CV_8UC3, imgTensorWithSharedData.flat<uint8_t>().data());
    //std::cout << "Before Color Resize" << std::endl;
    cv::resize(cvImg, cvImg_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);
    //std::cout << "After Resize" << std::endl;
    
    if(!isRGB)
    {
    cv::cvtColor(cvImg_resized, cvImg_resized, COLOR_BGR2RGB); 
    }//std::cout << "after color convert" << std::endl;

	//setup input tensor

	// Output
	//std::vector<tensorflow::Tensor> outputs;
    //std::cout << "Start Inference" << std::endl;
    //clock_t t_inference;
    //t_inference = clock();
    status = sess->Run({{"ImageTensor:0", imgTensorWithSharedData}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    if (!status.ok()) {
  	    std::cout << status.ToString() << "\n";
        return;
    }
    //std::cout << ((float)(t_inference))/CLOCKS_PER_SEC << std::endl;


    //process output
    //std::cout << "Make Output Matrices" << std::endl;
    
    //cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1);
    cv::Mat color_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC3);
    //cv::Mat out_prob(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_32F);
    //cv::Mat dynamic_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);
    cv::Mat mask_out = cv::Mat::zeros(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);



    
    //out_image.release();
    //out_prob.release();

    int value;
    float val_prob;
    //std::cout << "Get Tensor Values" << std::endl;
    //std::cout << "Out[0] size: "<< outputs[0].shape() << std::endl;
    //std::cout << "Out[1] size: "<< outputs[1].shape() << std::endl;
    int output_img_n0 = outputs[0].shape().dim_size(0);
    int output_img_h0 = outputs[0].shape().dim_size(1);
    int output_img_w0 = outputs[0].shape().dim_size(2);

    int output1_img_n0 = outputs[1].shape().dim_size(0);
    int output1_img_h0 = outputs[1].shape().dim_size(1);
    int output1_img_w0 = outputs[1].shape().dim_size(2);
    int output1_img_c0 = outputs[1].shape().dim_size(3);
    //std::cout << "Check1" << std::endl;

    for (unsigned int ni = 0; ni < output_img_n0; ni++)
    {
        for (unsigned int hi = 0; hi < output_img_h0; hi++)
        {
            for (unsigned int wi = 0; wi < output_img_w0; wi++)
            {
                {
                    // Get vaule through .flat()
                    //std::cout << "Check2" << std::endl;
                    unsigned int offset = ni * output_img_h0 * output_img_w0  +
                                hi * output_img_w0 +
                                wi;
                    //std::cout << "Check3" << std::endl;
                    value = outputs[0].flat<long long int>()(offset);
                    //std::cout << "Check4" << std::endl;
                    //unsigned int offset_prob = ni * output_img_h0 * output_img_w0 * output1_img_c0 +
                    //            hi * output_img_w0 * output1_img_c0 +
                    //            wi * output1_img_c0 +
                    //            value;
                    //std::cout << "Check5" << std::endl;
                    //val_prob = outputs[1].flat<float>()(offset_prob);
                    //std::cout << "Check6" << std::endl;
                    //out_image.at<uchar>(int(hi),int(wi)) = value;
                    //std::cout << "Check7" << std::endl;
                    //out_prob.at<float>(int(hi),int(wi)) = (val_prob+15)/40; //normalized
                    //std::cout << "Check8" << std::endl;
                    if (value == 13)
                    {
                        mask_out.ptr<uchar>(hi)[wi] = 1;
                    }
                    
                    color_mask.at<Vec3b>(Point(wi, hi))[0] = colormap[value][0];
                    color_mask.at<Vec3b>(Point(wi, hi))[1] = colormap[value][1];
                    color_mask.at<Vec3b>(Point(wi, hi))[2] = colormap[value][2];

                }
            }
        }
    }

    //results.Prediction = out_image;
    //results.Prob = out_prob;
    //std::cout << "Make Share Matrix" << std::endl;
    //resized_output.release();
    //resized_output_prob.release();
    resized_color_out.release();
    resized_mask_out.release();

    //resized_output.create(inputHeight,inputWidth,CV_8U);
    //resized_output_prob.create(inputHeight,inputWidth,CV_8U);
    resized_color_out.create(inputHeight,inputWidth,CV_8UC3);
    resized_mask_out.create(inputHeight,inputWidth,CV_8U);

    //cv::Mat resized_output_prob(inputHeight,inputWidth,CV_32F);
    //cv::imwrite("/home/richard/tensorflow_testing/images/resize_before", color_image, compression_params);
    //cv::resize(resized_output, out_image, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(cvImage_rgb, cvImg, cv::Size(target_width,target_height), 0, 0, INTER_LINEAR);
    //std::cout << "Resize" << std::endl;
    //cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(out_prob, resized_output_prob, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(color_mask, resized_color_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(mask_out, resized_mask_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    

    resultImg = resized_color_out;
    resultMask = resized_mask_out;
    //std::cout << " Inference Complete" << std::endl;
    //results.Prob = resized_output_prob;

    
    //results.DynamicMask = dynamic_mask
    //std::ostringstream ss;
    //ss << "/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/image" << image_number << ".png";
    //cv::imwrite("/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/one.png",results.Prediction,mcompression_params);
    //cv::imwrite(ss.str(),results.Prediction,mcompression_params);

    // Create a window for display.
    //namedWindow( "Semantic", WINDOW_AUTOSIZE );
    //imshow( "Semantic", resized_output);
    //results.classes;
    //std::cout << "Writing to" << ss.str() << std::endl;
    //image_number++;
}

void SemSeg::RunColorAndMask(const cv::Mat& cvImg, cv::Mat& resultClasses, cv::Mat& resultImg, cv::Mat& resultMask)// bgr image, output will be written to results class
{
    //std::cout << "Start RunColor" << std::endl;
    tensorflow::Status status;


    //template
    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    //std::string ty =  type2str( cvImg.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), cvImg.cols, cvImg.rows );
    cv::Mat cvImg_rgb;
    //cv::cvtColor(cvImg, cvImg, COLOR_BGR2RGB); 
    int tensor_image_width = 513;
    int tensor_image_height = 385;
    //convert image to tensor (513,385)
    
    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});
    cv::Mat cvImg_resized(tensor_image_height, tensor_image_width, CV_8UC3, imgTensorWithSharedData.flat<uint8_t>().data());
    //std::cout << "Before Color Resize" << std::endl;
    cv::resize(cvImg, cvImg_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);
    //std::cout << "After Resize" << std::endl;
    
    if(!isRGB)
    {
    cv::cvtColor(cvImg_resized, cvImg_resized, COLOR_BGR2RGB); 
    }//std::cout << "after color convert" << std::endl;

	//setup input tensor

	// Output
	//std::vector<tensorflow::Tensor> outputs;
    //std::cout << "Start Inference" << std::endl;
    //clock_t t_inference;
    //t_inference = clock();
    status = sess->Run({{"ImageTensor:0", imgTensorWithSharedData}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    if (!status.ok()) {
  	    std::cout << status.ToString() << "\n";
        return;
    }
   // std::cout << ((float)(t_inference))/CLOCKS_PER_SEC << std::endl;


    //process output
    //std::cout << "Make Output Matrices" << std::endl;
    
    cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1);
    cv::Mat color_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC3);
    //cv::Mat out_prob(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_32F);
    //cv::Mat dynamic_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);
    cv::Mat mask_out = cv::Mat::zeros(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);



    
    //out_image.release();
    //out_prob.release();

    int value;
    float val_prob;
    //std::cout << "Get Tensor Values" << std::endl;
    //std::cout << "Out[0] size: "<< outputs[0].shape() << std::endl;
    //std::cout << "Out[1] size: "<< outputs[1].shape() << std::endl;
    int output_img_n0 = outputs[0].shape().dim_size(0);
    int output_img_h0 = outputs[0].shape().dim_size(1);
    int output_img_w0 = outputs[0].shape().dim_size(2);

    int output1_img_n0 = outputs[1].shape().dim_size(0);
    int output1_img_h0 = outputs[1].shape().dim_size(1);
    int output1_img_w0 = outputs[1].shape().dim_size(2);
    int output1_img_c0 = outputs[1].shape().dim_size(3);
    //std::cout << "Check1" << std::endl;

    for (unsigned int ni = 0; ni < output_img_n0; ni++)
    {
        for (unsigned int hi = 0; hi < output_img_h0; hi++)
        {
            for (unsigned int wi = 0; wi < output_img_w0; wi++)
            {
                {
                    // Get vaule through .flat()
                    //std::cout << "Check2" << std::endl;
                    unsigned int offset = ni * output_img_h0 * output_img_w0  +
                                hi * output_img_w0 +
                                wi;
                    //std::cout << "Check3" << std::endl;
                    value = outputs[0].flat<long long int>()(offset);
                    //std::cout << "Check4" << std::endl;
                    //unsigned int offset_prob = ni * output_img_h0 * output_img_w0 * output1_img_c0 +
                    //            hi * output_img_w0 * output1_img_c0 +
                    //            wi * output1_img_c0 +
                    //            value;
                    //std::cout << "Check5" << std::endl;
                    //val_prob = outputs[1].flat<float>()(offset_prob);
                    //std::cout << "Check6" << std::endl;
                    out_image.at<uchar>(int(hi),int(wi)) = value;
                    //std::cout << "Check7" << std::endl;
                    //out_prob.at<float>(int(hi),int(wi)) = (val_prob+15)/40; //normalized
                    //std::cout << "Check8" << std::endl;
                    if (value == 13)
                    {
                        mask_out.ptr<uchar>(hi)[wi] = 1;
                    }
                    
                    color_mask.at<Vec3b>(Point(wi, hi))[0] = colormap[value][0];
                    color_mask.at<Vec3b>(Point(wi, hi))[1] = colormap[value][1];
                    color_mask.at<Vec3b>(Point(wi, hi))[2] = colormap[value][2];

                }
            }
        }
    }

    //results.Prediction = out_image;
    //results.Prob = out_prob;
    //std::cout << "Make Share Matrix" << std::endl;
    resized_output.release();
    //resized_output_prob.release();
    resized_color_out.release();
    resized_mask_out.release();

    resized_output.create(inputHeight,inputWidth,CV_8U);
    //resized_output_prob.create(inputHeight,inputWidth,CV_8U);
    resized_color_out.create(inputHeight,inputWidth,CV_8UC3);
    resized_mask_out.create(inputHeight,inputWidth,CV_8U);

    //cv::Mat resized_output_prob(inputHeight,inputWidth,CV_32F);
    //cv::imwrite("/home/richard/tensorflow_testing/images/resize_before", color_image, compression_params);
    cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(cvImage_rgb, cvImg, cv::Size(target_width,target_height), 0, 0, INTER_LINEAR);
    //std::cout << "Resize" << std::endl;
    //cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(out_prob, resized_output_prob, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(color_mask, resized_color_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(mask_out, resized_mask_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    
    resultClasses = resized_output;
    resultImg = resized_color_out;
    resultMask = resized_mask_out;
    //std::cout << " Inference Complete" << std::endl;
    //results.Prob = resized_output_prob;

    
    //results.DynamicMask = dynamic_mask
    //std::ostringstream ss;
    //ss << "/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/image" << image_number << ".png";
    //cv::imwrite("/home/richard/Data/Ubuntu/codes/ORB_SLAM2_thesis/images/one.png",results.Prediction,mcompression_params);
    //cv::imwrite(ss.str(),results.Prediction,mcompression_params);

    // Create a window for display.
    //namedWindow( "Semantic", WINDOW_AUTOSIZE );
    //imshow( "Semantic", resized_output);
    //results.classes;
    //std::cout << "Writing to" << ss.str() << std::endl;
    //image_number++;
}


void SemSeg::RunSegLabel(const cv::Mat& cvImg)// bgr image, output will be written to results class
{
    tensorflow::Status status;

    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    cv::Mat cvImg_rgb;
    int tensor_image_width = 513;
    int tensor_image_height = 385;
    
    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});
    cv::Mat cvImg_resized(tensor_image_height, tensor_image_width, CV_8UC3, imgTensorWithSharedData.flat<uint8_t>().data());

    cv::resize(cvImg, cvImg_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);

    cv::cvtColor(cvImg_resized, cvImg_resized, COLOR_BGR2RGB); 

	//setup input tensor

	// Output
	//std::vector<tensorflow::Tensor> outputs;

    //clock_t t_inference;
    //t_inference = clock();
    status = sess->Run({{"ImageTensor:0", imgTensorWithSharedData}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    if (!status.ok()) {
  	    std::cout << status.ToString() << "\n";
        return;
    }

    
    //cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1,outputs[0].flat<long long int>().cast<uint8_t>());
    //cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1, (void*)outputs[0].bit_casted_tensor<uint8_t,3>().data());

    int value;
    float val_prob;

    int output_img_n0 = outputs[0].shape().dim_size(0);
    int output_img_h0 = outputs[0].shape().dim_size(1);
    int output_img_w0 = outputs[0].shape().dim_size(2);

    int output1_img_n0 = outputs[1].shape().dim_size(0);
    int output1_img_h0 = outputs[1].shape().dim_size(1);
    int output1_img_w0 = outputs[1].shape().dim_size(2);
    int output1_img_c0 = outputs[1].shape().dim_size(3);

    //out_image.data  = outputs[0].bit_casted_tensor<uchar,3>().data(); //Eigen::RowMajor

    //outputs[0].bit_casted_shaped<uchar,3>({1}).data();
    //outputs[0].bit_casted_tensor<uint8_t,3>().data();

   
    //out_image.data  = ;
    cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);
    
    for (unsigned int ni = 0; ni < output_img_n0; ni++)
    {
        for (unsigned int hi = 0; hi < output_img_h0; hi++)
        {
            for (unsigned int wi = 0; wi < output_img_w0; wi++)
            {
                {

                    unsigned int offset = ni * output_img_h0 * output_img_w0  +
                                hi * output_img_w0 +
                                wi;

                    value = outputs[0].flat<long long int>()(offset);

                    out_image.at<uchar>(int(hi),int(wi)) = value;

                }
            }
        }
    }

    LastSegLabel.create(inputHeight,inputWidth,CV_8U);

    cv::resize(out_image, LastSegLabel, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //std::cout << " Inference Complete" << std::endl;
    //results.Prob = resized_output_prob;

}

void SemSeg::SetSegMaxDepth(float val)
{
    unique_lock<mutex> lock(settingsMutex);
    this->SegMaxDepth = val;
}
void SemSeg::SetSegMinDepth(float val)
{
    unique_lock<mutex> lock(settingsMutex);
    this->SegMinDepth = val;
}
void SemSeg::SetSegMinProb(float val)
{
    unique_lock<mutex> lock(settingsMutex);
    this->SegMinProb =val;
}




void SemSeg::ComputeAllLatest(const cv::Mat& cvImg, const cv::Mat& cvDepth)// bgr image, output will be written to results class
{
    //std::cout << "Start RunColor" << std::endl;
    tensorflow::Status status;

    unique_lock<mutex> lock(settingsMutex);


    //template
    int inputHeight = cvImg.size().height;
    int inputWidth = cvImg.size().width;
    //std::string ty =  type2str( cvImg.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), cvImg.cols, cvImg.rows );
    cv::Mat cvImg_rgb;
    //cv::cvtColor(cvImg, cvImg, COLOR_BGR2RGB); 
    int tensor_image_width = 513;
    int tensor_image_height = 385;
    //convert image to tensor (513,385)
    
    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, tensor_image_height, tensor_image_width, 3});
    cv::Mat cvImg_resized(tensor_image_height, tensor_image_width, CV_8UC3, imgTensorWithSharedData.flat<uint8_t>().data());
    cv::Mat cvDepth_resized;
    cv::resize(cvImg, cvImg_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);

    if(useSegDepthTh)
    {
        cv::resize(cvDepth, cvDepth_resized, cv::Size(tensor_image_width,tensor_image_height), 0, 0, INTER_LINEAR);
    }
    

    //cv::cvtColor(cvImg_resized, cvImg_resized, COLOR_BGR2RGB); 




	//std::vector<tensorflow::Tensor> outputs;

    //clock_t t_inference;
    //t_inference = clock();
    status = sess->Run({{"ImageTensor:0", imgTensorWithSharedData}},
                          {output_node_label, output_node_prob}, 
                          {},
                          &outputs);
    if (!status.ok()) {
  	    std::cout << status.ToString() << "\n";
        return;
    }



    cv::Mat out_image(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1);
    cv::Mat out_image_remap(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC1);
    cv::Mat color_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2),CV_8UC3);
    cv::Mat out_prob(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_32F);
    //cv::Mat dynamic_mask(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);
    cv::Mat mask_out = cv::Mat::zeros(outputs[0].shape().dim_size(1),outputs[0].shape().dim_size(2), CV_8U);



    
    //out_image.release();
    //out_prob.release();

    int value;
    float val_prob;
    float val_prob_norm;
    //std::cout << "Get Tensor Values" << std::endl;
    //std::cout << "Out[0] size: "<< outputs[0].shape() << std::endl;
    //std::cout << "Out[1] size: "<< outputs[1].shape() << std::endl;
    int output_img_n0 = outputs[0].shape().dim_size(0);
    int output_img_h0 = outputs[0].shape().dim_size(1);
    int output_img_w0 = outputs[0].shape().dim_size(2);

    int output1_img_n0 = outputs[1].shape().dim_size(0);
    int output1_img_h0 = outputs[1].shape().dim_size(1);
    int output1_img_w0 = outputs[1].shape().dim_size(2);
    int output1_img_c0 = outputs[1].shape().dim_size(3);


    //clock_t t_get_data = clock();
    
    for (unsigned int ni = 0; ni < output_img_n0; ni++)
    {
        //#pragma omp parallel for
        for (unsigned int hi = 0; hi < output_img_h0; hi++)
        {
            for (unsigned int wi = 0; wi < output_img_w0; wi++)
            {
                {
                    unsigned int offset = ni * output_img_h0 * output_img_w0  +
                                hi * output_img_w0 +
                                wi;

                    value = outputs[0].flat<long long int>()(offset);

                    unsigned int offset_prob = ni * output_img_h0 * output_img_w0 * output1_img_c0 +
                                hi * output_img_w0 * output1_img_c0 +
                                wi * output1_img_c0 +
                                value;

                    val_prob = outputs[1].flat<float>()(offset_prob);
                    val_prob_norm = (val_prob+15)/40;


                    if(useSegDepthTh)
                    {
                        float _depth = cvDepth_resized.ptr<float>(hi)[wi];
                        if(_depth < SegMinDepth || _depth > SegMaxDepth)
                        {
                            value = 0;
                        }
                        
                    }

                    if(useSegProbTh)
                    {
                        if (val_prob_norm < SegMinProb)
                        {
                            value = 0;
                        }
                    }

                    out_image.at<uchar>(int(hi),int(wi)) = value;
                    out_image_remap.at<uchar>(int(hi),int(wi)) = remapLabel(value);


                    out_prob.at<float>(int(hi),int(wi)) =  val_prob_norm;   //normalized

                    if (isDynamic(value))
                    {
                        mask_out.ptr<uchar>(hi)[wi] = 1;
                    }
                    
                    color_mask.at<Vec3b>(Point(wi, hi))[0] = colormap[value][0];
                    color_mask.at<Vec3b>(Point(wi, hi))[1] = colormap[value][1];
                    color_mask.at<Vec3b>(Point(wi, hi))[2] = colormap[value][2];

                }
            }
        }
    }

    //std::cout << "Data Get " << ((float)(clock()-t_get_data))/CLOCKS_PER_SEC<< std::endl;

    //results.Prediction = out_image;
    //results.Prob = out_prob;
    //std::cout << "Make Share Matrix" << std::endl;
    resized_output.release();
    resized_output_remap.release();
    resized_output_prob.release();
    resized_color_out.release();
    resized_mask_out.release();

    resized_output.create(inputHeight,inputWidth,CV_8U);
    resized_output_prob.create(inputHeight,inputWidth,CV_8U);
    resized_color_out.create(inputHeight,inputWidth,CV_8UC3);
    resized_mask_out.create(inputHeight,inputWidth,CV_8U);

    //cv::Mat resized_output_prob(inputHeight,inputWidth,CV_32F);
    //cv::imwrite("/home/richard/tensorflow_testing/images/resize_before", color_image, compression_params);
    cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(out_image_remap, resized_output_remap, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    //cv::resize(cvImage_rgb, cvImg, cv::Size(target_width,target_height), 0, 0, INTER_LINEAR);
    //std::cout << "Resize" << std::endl;
    cv::resize(out_image, resized_output, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(out_prob, resized_output_prob, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(color_mask, resized_color_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);
    cv::resize(mask_out, resized_mask_out, cv::Size(inputWidth,inputHeight), 0,0, INTER_NEAREST);

    int dilation_size = this->mDilation_size;
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, Size( 2*dilation_size + 1, 2*dilation_size+1 ), Point( dilation_size, dilation_size ));
    cv::dilate(resized_mask_out,resized_mask_out,kernel);


    LastSegLabel = resized_output;
    LastSegLabelRemap = resized_output_remap;
    LastSegColor = resized_color_out;
    LastSegProb = resized_output_prob;
    LastDynMask = resized_mask_out;
}


int SemSeg::remapLabel(int _label){
    int label = _label;

    //table 16 desk 34
    if(label == 34)
    {
        label = 16;
    }
    //chair 20, armchair, 31, seat 32, swivel chair 76,
    if(label ==31 || label==32 || label ==76)
    {
        label = 20;
    }


    if(label == 73)
    {
        label = 5;
    }

    //computer 75, crt screen 142, monitor monitoring device 144, screen 131
    if(label == 142 || label == 144 || label==131)
    {
        label = 75;
    }



    return label;
}


bool SemSeg::isDynamic(int _label){

    //people 13, mirror 28, lamp 37, sky 3, vehicle 21
    if (_label == 13 || _label == 28 || _label==3)
    {
        return true;
    }
    else
    {
        return false;
    }
}