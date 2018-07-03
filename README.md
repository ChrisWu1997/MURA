# Abstract
In this project, we implemented the task of abnormality detection, abnormality localization and image retrieval for musculoskeletal radiographs on MURA dataset. In task one, we trained some models based on ResNet and DenseNet to classify the input study as either normal or abnormal and use the ensembled prediction for output. We designed FuseNet, which fuses the outcome of the global image and that of the local area, to improve the performance of prediction. In task two, we tried to localize and visualize the abnormal areas on radio graph by applying gradient class activation map. In task three, we used the features extracted by our model as codes to output the images in the training data that is most similar to the input image.

# Environment

- python 3.6.2
- numpy 1.14.3
- pytorch 0.4.0
- pillow 3.4.2
- opencv 3.1.0

# Run

## Abnormality Detection

To test the performance on test dataset, run `python3 predict.py --data_dir=<parent directory of MURA-v1.0> --save_dir=<parent directory to write result> --phase='test'.` The results of the five single models and the ensembled model will be printed. The final results of accuracy and AUC will be saved in  ` prediction_result.txt ` and the ROC curve will be saved in  ` ROC_curve.png ` in the parent directory to write result. 

For example , run `python3 predict.py --data_dir=/data1/wurundi/ML/data --save_dir=results --phase='valid'` .

## Abnormality Localization

To locate the abnormality area of a query input, run `python3 locate.py --img_path=<filepath of the query input>`. The heatmap and the marked window will be shown in the same directory of the query input.

For example, run `python3 locate.py --img_path=results/localize_result/elbow.png`, the heatmap will be shown in `results/localize_result/elbow_m.png`and the marked window will be shown in `results/localize_result/elbow_w.png`.



## Image Retrieval

To get the similar images of a query input, run `python3 retrieval.py --img_path=<filepath of the query input> --data_dir=<parent directory of MURA-v1.0>`.  The top 5 similar images will be saved in the  same directory of the query input.

To specify the type, add `--img_type=<study type of the image>`, the default is searching from all types.

For example, run `python3 retrieval.py --img_path=results/retrieval_result/elbow.png --data_dir=/data1/wurundi/ML/data --img_type=ELBOW`, the top 5 similar images `elbow1.png` `elbow2.png` `elbow3.png` `elbow4.png` `elbow5.png` will be saved in the same directory of `elbow.png`.
