# Waste Detector
This project consists of a waste detector.

The main goal is to fine-tune a pretrained YOLO model using only the
[TACO dataset](http://tacodataset.org/) and evaluate the best
performance achievable with this dataset.

Because I am familiar with notebooks and find them more presentable (not in this case), the training and evaluation are included in a notebook. The TACO dataset files are not included in this repository. However, the repository contains a script for converting the annotations from COCO format to YOLO format.

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

At this point I saw that the comparison wasn't fair, so I decided to do a common test evaluation, adding also precision and recall:

|model|group|precision|recall|mAP50|mAP50-95|
|---|---|---|---|---|---|
|Training 1|single\_split|0\.310047|0\.233260|0\.201000|0\.150287|
|Training 2|single\_split|0\.357134|0\.277987|0\.212594|0\.152524|
|CV Fold 1|cross\_validation|0\.373824|0\.297314|0\.269231|0\.186291|
|CV Fold 2|cross\_validation|0\.346033|0\.243694|0\.233916|0\.173992|
|CV Fold 3|cross\_validation|0\.246805|0\.302374|0\.232977|0\.168367|
|CV Fold 4|cross\_validation|0\.362129|0\.258383|0\.234328|0\.177216|

CV Fold 1 achieved the best overall test performance according to the mAP metrics. However, CV Fold 3 achieved a slightly higher recall (0.302), but its precision and mAP values were lower.

These results show that training with different data splits and larger validation sets helped achieve better mAP scores on this test set. Still, cross-validation does not guarantee a better model every time.

The model could be improved with more relevant images, better class balancing and further tuning. However, these results are enough for this exploratory project and demonstrate the complete process of preparing the data, fine-tuning YOLO and evaluating the model.


Here we have some validation results (CV Fold 1)
<img width="1920" height="1428" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/7a6606fd-67ce-458d-8b1d-f0ca88adca0d" />


In the end I think it is remarkable to say that the main reason of such a low rates, is that there is a small amount of images in the TACO dataset, comes with highly unbalanced classes.

For fun, I decided to try it using my own photo of litter I found in the trash at home. As you can see, the results matched what the indicators showed (the only correctly classified object was the plastic bottle in the top-left corner). The other objects, like the glass jar (which should have been classified as 'Other'), the wrapper (which should have been classified as 'Plastic bag + wrapper') and the napkin (which should have been classified as 'Other') were classified incorrectly.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e0c11f43-559d-47af-822c-0af299b40e84" alt="Original litter photograph" width="49%">
  <img src="https://github.com/user-attachments/assets/fbb3a634-39e7-4fd1-bb91-4aabfe3028fb" alt="Litter classification results" width="49%">
</p>
