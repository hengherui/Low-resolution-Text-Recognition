
### Contributions

* We put forward an efficient multi-task network that can jointly handle low-resolution text recognition. We propose a multi-task learning approach for scene text recognition, which includes a text recognition branch and a super-resolution branch. The proposed super-resolution branch, incorporating residual super-resolution units, effectively captures rich information from low-resolution features. It is utilized to generate high-resolution text features by contrastive learning between high-resolution and low-resolution features.

* We conduct large number of experiments on real logistics express sheet images, which are not represented on our paper, here, we show these recognition results, and we mosaicked the sensitive information such as name and telephone num for preventing disclosure of private customer information during the presentation. We show different logistics scenes, including the blurred, tilted, dark background, low-resolution images, all these scenes can prove that our method have robustness on multi scenes of logistics express image text recognition. The details have shown from Figure 1 to Figure 13.

## training enviroment

Tesla V100 32G memory, 8 cards.
install required package "pip install -r requirments" 

## Data preparation

We give an example to construct your own datasets. Details please refer to `tools/create_svtp_lmdb.py`.
We provide datasets for [training](https://pan.baidu.com/s/1BMYb93u4gW_3GJdjBWSCSw&shfl=sharepset) (password: wi05) and [testing](https://drive.google.com/open?id=1U4mGLlsm9Ade1-gQOyd6He5R0yiaafYJ).

### Logistics text recognition Result

#### Example one, the area inside the blue box is the detected text region in the left side, the left side is the detected logistics sheet image, and the right side is the recognition result corresponding to the text area one-to-one. Although the image is blurred, but our model gives the correct results.
<img src="https://github.com/hengherui/Low-resolution-Text-Recognition/blob/master/Results/1.jpg" width="500px">

#### Drivable Area Segmentation Result

![](pictures/da.png)

#### Lane Detection Result

![](pictures/ll.png)

**Notes**: 

- The visualization of lane detection result has been post processed by quadratic fitting.

---

### Project Structure

```python
├─inference
│ ├─images   # inference images
│ ├─output   # inference result
├─lib
│ ├─config/default   # configuration of training and validation
│ ├─core    
│ │ ├─activations.py   # activation function
│ │ ├─evaluate.py   # calculation of metric
│ │ ├─function.py   # training and validation of model
│ │ ├─general.py   #calculation of metric、nms、conversion of data-format、visualization
│ │ ├─loss.py   # loss function
│ │ ├─postprocess.py   # postprocess(refine da-seg and ll-seg, unrelated to paper)
│ ├─dataset
│ │ ├─AutoDriveDataset.py   # Superclass dataset，general function
│ │ ├─bdd.py   # Subclass dataset，specific function
│ │ ├─hust.py   # Subclass dataset(Campus scene, unrelated to paper)
│ │ ├─convect.py 
│ │ ├─DemoDataset.py   # demo dataset(image, video and stream)
│ ├─models
│ │ ├─YOLOP.py    # Setup and Configuration of model
│ │ ├─light.py    # Model lightweight（unrelated to paper, zwt)
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
│ │ ├─split_dataset.py  # (Campus scene, unrelated to paper)
│ │ ├─utils.py  # logging、device_select、time_measure、optimizer_select、model_save&initialize 、Distributed training
│ ├─run
│ │ ├─dataset/training time  # Visualization, logging and model_save
├─tools
│ │ ├─demo.py    # demo(folder、camera)
│ │ ├─test.py    
│ │ ├─train.py    
├─toolkits
│ │ ├─deploy    # Deployment of model
│ │ ├─datapre    # Generation of gt(mask) for drivable area segmentation task
├─weights    # Pretraining model
```

---

### Requirement

This codebase has been developed with python version 3.7, PyTorch 1.7+ and torchvision 0.8+:

```
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

See `requirements.txt` for additional dependencies and version requirements.

```setup
pip install -r requirements.txt
```

### Data preparation

#### Download

- Download the images from [images](https://bdd-data.berkeley.edu/).

- Download the annotations of detection from [det_annotations](https://drive.google.com/file/d/1Ge-R8NTxG1eqd4zbryFo-1Uonuh0Nxyl/view?usp=sharing). 
- Download the annotations of drivable area segmentation from [da_seg_annotations](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view?usp=sharing). 
- Download the annotations of lane line segmentation from [ll_seg_annotations](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view?usp=sharing). 

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./lib/config/default.py`.

### Training

You can set the training configuration in the `./lib/config/default.py`. (Including:  the loading of preliminary model,  loss,  data augmentation, optimizer, warm-up and cosine annealing, auto-anchor, training epochs, batch_size).

If you want try alternating optimization or train model for single task, please modify the corresponding configuration in `./lib/config/default.py` to `True`. (As following, all configurations is `False`, which means training multiple tasks end to end).

```python
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task
```

Start training:

```shell
python tools/train.py
```



### Evaluation

You can set the evaluation configuration in the `./lib/config/default.py`. (Including： batch_size and threshold value for nms).

Start evaluating:

```shell
python tools/test.py --weights weights/End-to-end.pth
```



### Demo Test

We provide two testing method.

#### Folder

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --source inference/images
```



#### Camera

If there are any camera connected to your computer, you can set the `source` as the camera number(The default is 0).

```shell
python tools/demo.py --source 0
```



#### Demonstration

<table>
    <tr>
            <th>input</th>
            <th>output</th>
    </tr>
    <tr>
        <td><img src=pictures/input1.gif /></td>
        <td><img src=pictures/output1.gif/></td>
    </tr>
    <tr>
         <td><img src=pictures/input2.gif /></td>
        <td><img src=pictures/output2.gif/></td>
    </tr>
</table>



### Deployment

Our model can reason in real-time on `Jetson Tx2`, with `Zed Camera` to capture image. We use `TensorRT` tool for speeding up. We provide code for deployment and reasoning of model in  `./toolkits/deploy`.



### Segmentation Label(Mask) Generation

You can generate the label for drivable area segmentation task by running

```shell
python toolkits/datasetpre/gen_bdd_seglabel.py
```



#### Model Transfer

Before reasoning with TensorRT C++ API, you need to transfer the `.pth` file into binary file which can be read by C++.

```shell
python toolkits/deploy/gen_wts.py
```

After running the above command, you obtain a binary file named `yolop.wts`.



#### Running Inference

TensorRT needs an engine file for inference. Building an engine is time-consuming. It is convenient to save an engine file so that you can reuse it every time you run the inference. The process is integrated in `main.cpp`. It can determine whether to build an engine according to the existence of your engine file.



### Third Parties Resource  
* YOLOP OpenCV-DNN C++ Demo: [YOLOP-opencv-dnn](https://github.com/hpc203/YOLOP-opencv-dnn) from [hpc203](https://github.com/hpc203)  
* YOLOP ONNXRuntime C++ Demo: [lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolop.cpp) from [DefTruth](https://github.com/DefTruth)  
* YOLOP NCNN C++ Demo: [YOLOP-NCNN](https://github.com/EdVince/YOLOP-NCNN) from [EdVince](https://github.com/EdVince)  
* YOLOP MNN C++ Demo: [YOLOP-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolop.cpp) from [DefTruth](https://github.com/DefTruth) 
* YOLOP TNN C++ Demo: [YOLOP-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolop.cpp) from [DefTruth](https://github.com/DefTruth) 	



## Citation

If you find our paper and code useful for your research, please consider giving a star :star:   and citation :pencil: :

```BibTeX
@misc{2108.11250,
Author = {Dong Wu and Manwen Liao and Weitian Zhang and Xinggang Wang},
Title = {YOLOP: You Only Look Once for Panoptic Driving Perception},
Year = {2021},
Eprint = {arXiv:2108.11250},
}
```

