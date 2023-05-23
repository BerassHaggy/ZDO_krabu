# ZDO_krabu
GitHub repository for computer vision (ZDO). 

## Description
The main goal is to detect incision a stitches in each picture. 
We are using traditional computer vision methods such as Hough transformation, 
adaptive thresholding, etc. Moreover, we are trying to evaluate the quality of surgical
stitching based mainly on the angle between the incision and the stitch. 

## Input data
As the input data we were using annotated medical pictures. The pictures were
annotated in [CVAT](https://www.cvat.ai/). We had to deal with the poor quality of the pictures (as you can see below).
The annotations were saved in .json file.

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/SA_20211012-164802_incision_crop_0.jpg)

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/SA_20211012-165505_incision_crop_0.jpg)

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/SA_20211012-181437_incision_crop_0.jpg)

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/SA_20220503-113941_incision_crop_0.jpg)

## Output data
![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/output_1.png)

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/output_2.png)

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/output_3.png)

![alt text](https://github.com/BerassHaggy/ZDO_krabu/blob/main/graphics/output_4.png)


