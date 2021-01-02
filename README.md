# Semantic 3D Mapping for Dynamic Environments

This algorithm is built on top of [ORB-SLAM2 RGB-D](https://github.com/raulmur/ORB_SLAM2), it is modified with dynamic point rejection and semantic point cloud reconstruction

The code was tested on a Laptop with:
Intel i7-8750H CPU,
NVIDIA 1060 Max-Q GPU 6GB and
16GB DDR4 RAM

References

[ORB_SLAM2_SSD_Semantic](https://github.com/Ewenwan/ORB_SLAM2_SSD_Semantic)

[DynaSLAM](https://github.com/BertaBescos/DynaSLAM)

# 1. Prerequisite and Dependencies

## 1.1 Ubuntu 16.04 LTS

The code was tested on a Laptop with dual boot, the code using the D435 camera live might not work on a virtual machine because of the drivers

## 1.2 C++11 Compiler

C++11 compiler is used

## 1.3 Pangolin

git clone [Pangolin](https://github.com/stevenlovegrove/Pangolin) and build from source

## 1.4 Eigen 3.3.6

A 3.3.X version of [Eigen](http://bitbucket.org/eigen/eigen/get/3.3.6.tar.bz2) is required, download and follow the instuctions in http://eigen.tuxfamily.org/index.php?title=Main_Page

## 1.5 OpenCV 3.4.7

git clone [OpenCV 3.4.7](https://github.com/opencv/opencv/tree/3.4.7) and build from source

## 1.6 CUDA 9.0

Install [CUDA 9.0 Toolkit](https://developer.nvidia.com/cuda-90-download-archive) 

## 1.7 cuDNN 7

Install [cuDNN](https://developer.nvidia.com/cudnn), registration of nvidia developer account is needed (free)

## 1.8 Protobuf 3.6.1

git clone [Protobuf 3.6.1](https://github.com/protocolbuffers/protobuf/tree/v3.6.1) and build from source 

## 1.9 Tensorflow 1.12.3 GPU

git clone [TensorFlow 1.12.3](https://github.com/tensorflow/tensorflow/tree/v1.12.3) repository and build from source

build for libtensorflow_cc.so with GPU support using Bazel

## 1.10 PCL 1.8.1

git clone [PCL 1.8.1](https://github.com/PointCloudLibrary/pcl/tree/pcl-1.8.1) and build from source 

## 1.11 librealsense 2.29.0

git clone [librealsense 2.29.0](https://github.com/IntelRealSense/librealsense/tree/v2.29.0) and build from source 

# 2. Build

After the installing the dependencies 

edit path of the **TensorFlow** *include_directories* and *target_link_libraries* in *CMakeLists.txt* in the root directory, to the path of your tensorflow installation

```
include_directories("/home/richard/dependencies/tensorflow")
include_directories("/home/richard/dependencies/tensorflow/bazel-genfiles")
include_directories("/home/richard/dependencies/tensorflow/tensorflow")
include_directories("/home/richard/dependencies/tensorflow/third_party")


/home/richard/dependencies/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
```

cd to the root directory of this code

```
cd s_orb_code
chmod +x build.sh
./build.sh
```

This will create **libORB_SLAM2.so**  at *lib* folder, the executables **rgbd_tum_seg** in *Examples/RGBD-D/bin* and **rs_live**, **rs_play** and **rs_fbf** in *realsense/bin* folder.


# 3. TUM Dataset Freiburg 3 Examples

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file.

## 3.1 Run fr3 dataset frame by frame

```
./Examples/Monocular/rgbd_tum_seg Vocabulary/ORBvoc.bin Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```
```
./Examples/RGB-D/bin/rgbd_tum_seg Vocabulary/ORBvoc.bin Examples/RGB-D/TUM3.yaml  /home/richard/Data/DATASETS/TUM/fr3_walking_xyz/ /home/richard/Data/DATASETS/TUM/fr3_walking_xyz/associations.txt
```
# 4. Realsense Example

For live SLAM with realsense D435 camera use **rs_live**

The realsense examples are taken using the realsense-viewer and saved as a **.bag** file, the **.bag** file is used as a dataset to run in frame by frame mode or in real time mode

for indoors scenes use rs.yaml, for outdoors use rs_out.yaml

## 4.1 Run d435 live 
```
./realsense/bin/rs_play Vocabulary/ORBvoc.bin realsense/rs.yaml
```
## 4.2 Run dataset real time
```
./realsense/bin/rs_fbf Vocabulary/ORBvoc.bin realsense/rs.yaml /home/richard/Data/DATASETS/RS_personal/ISL_01.bag
```
## 4.3 Run dataset frame by frame
```
./realsense/bin/rs_play Vocabulary/ORBvoc.bin realsense/rs.yaml /home/richard/Data/DATASETS/RS_personal/ISL_01.bag
```

