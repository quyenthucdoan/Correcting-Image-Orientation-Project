

DINOv2:
1. Cài đặt các gói Python cần thiết:
```
    !pip install torch torchvision torchaudio
    !pip install Pillow numpy tqdm scikit-learn matplotlib
    !pip install progressbar
```

2. Chuẩn bị Dữ liệu: Bước này tải bộ dữ liệu Indoor CVPR 2009, giải nén và tạo ra một bộ dữ liệu mới chứa các ảnh đã được xoay ngẫu nhiên.
```
    !wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
    !git clone https://github.com/geegatomar/Correcting-Image-Orientation-Project-Adrian-Rosebrock-ImageNet-Bundle-.git
    !cp -r Correcting-Image-Orientation-Project-Adrian-Rosebrock-ImageNet-Bundle-/* .
    !mkdir -p image_orientation/indoor_cvpr
    !tar -xvf ./indoorCVPR_09.tar -C image_orientation/indoor_cvpr/
```

3. Xoay ảnh để chuẩn bị cho bước train:
```
    %cd ./image_orientation
    !python create_dataset.py --dataset indoor_cvpr/Images --output indoor_cvpr/rotated_images
```

4. Train bằng model 
Parameters hiện tại của model:
    NUM_EPOCHS = 30                   
    BATCH_SIZE = 32                    
    LEARNING_RATE = 1e-4                      
Và đang sử dụng pretrained-model là 'dinov2_vitl14_reg'.

Để train model chạy command sau:
```
!python image_orientation/train_orientation.py --dataset image_orientation/indoor_cvpr/Images --output image_orientation/indoor_cvpr/model
```
