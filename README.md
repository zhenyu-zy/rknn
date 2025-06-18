RKNN Model Zoo = 2.3.2
rknn-toolkit2 = 2.3.2
rknn_toolkit_lite2 = 2.3.2
RKNPU2 = 2.3.2
ultralytics_yolov8 = 

git clone https://github.com/airockchip/rknn_model_zoo.git
git clone https://github.com/airockchip/rknn_toolkit2.git
git clone https://github.com/airockchip/rknn_toolkit_lite2.git
git clone https://github.com/airockchip/RKNPU2.git
git clone https://github.com/airockchip/ultralytics_yolov8.git
git clone https://github.com/LubanCat/lubancat_ai_manual_code.git
git clone https://github.com/ultralytics/ultralytics.git

YOLO环境配置
windows端
在conda终端
conda create -n yolov8 python=3.8
conda activate yolov8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
在pycharm中下载ultralytics,在终端中
pip install ultralytics

虚拟机端
cd rknn_toolkit2
pip install -r requirements_cp38-2.3.2.txt
pip install rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install numpy==1.23.5 pillow==9.5.0 torch==2.3.1 torchvision==0.18.1 python-dateutil==2.8.2
pip install ultralytics

rk3588
cd rknn_toolkit_lite2
pip install rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install numpy==1.23.5 pillow==9.5.0 torch==2.3.1 torchvision==0.18.1 python-dateutil==2.8.2
pip install ultralytics

demo编译
./build-linux.sh -t rk3588 -a aarch64 -d yolov8

超频
cd rknn_yolo
sudo chmod +x scaling_frequency.sh
./scaling_frequency.sh -c rk3588


