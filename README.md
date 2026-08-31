# Waste Detector
This project consists of a waste detector.

The main goal is to fine-tune a pretrained YOLO model using only the
[TACO dataset](http://tacodataset.org/) and evaluate the best
performance achievable with this dataset.

Because I am familiar with notebooks and find them more presentable, the training and evaluation are included in a notebook. The TACO dataset files are not included in this repository. However, the repository contains a script for converting the annotations from COCO format to YOLO format.

The most important resources that helped me understand how to approach this project are the following:

- [SortWaste: A Densely Annotated Dataset for Object Detection in Industrial Waste Sorting](https://arxiv.org/pdf/2601.02299) - This paper presents a recent project involving a waste-sorting conveyor and reports that YOLO achieves the best performance.

- [TACO: Trash Annotations in Context for Litter Detection](https://arxiv.org/pdf/2003.06975) - This paper helped me better understand how object detection works. I used its dataset to train the model, avoiding the need to annotate images manually. The paper reports a maximum mAP of 17.6%.

- [How to Train YOLO Object Detection Models in Google Colab (YOLO26, YOLO11, YOLOv8)](https://www.youtube.com/watch?v=r0RspiLG260) - This video also served as inspiration for the project by demonstrating a similar process using a small dataset of candy images instead of litter.


---

After executing many experiments in notebook I get the next results:

| Experiment | Epochs | Image size | Evaluation method                       |         mAP50 |      mAP50–95 |
| ---------- | -----: | ---------: | --------------------------------------- | ------------: | ------------: |
| Training 1 |    100 |        640 | 1200 train / 150 validation / 150 test |         0.209 |         0.149 |
| Training 2 |     50 |        960 | 1200 train / 150 validation / 150 test |         0.215 |         0.154 |
| Training 3 |     50 |        960 | Four-fold cross-validation (each run aprox 1012 train and 338 validation / 150 test)  | 0.270 ± 0.027 | 0.208 ± 0.022 |

