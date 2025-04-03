# AicraftSpotter

This is a project propose in the course of Data Driven Engineering for the MA1 Students

## Description
It is proposed to build an aircraft classifier based on photographs. This kind
of image recognition is very valuable in airport management in order to improve
quality and ground operation - refuelling, luggage management etc. - along with
safety like following aircraft on the taxi and runways. The dataset is composed
of 10.000 pictures, located in the data/images directory. Each image name is a
number, using this number several information could be found in the text files :

* images_box.txt gives a bounding box for each image that encompasses the
aircraft on the picture allowing to remove non-relevant information. The
bounding box is specified by four numbers: xmin, ymin, xmax and ymax. The
top-left pixel of an image has coordinate (1,1).

* The images_family_* gives the picture number, aircraft manufacured - Boe-
ing, Airbus ... - and family - 707, 747, 320, etc ....

  * images_family_trainval.txt gives trainning dataset.
  * images_family_train.txt and images_family_val.txt gives a subset of
the trainning dataset.
  * images_family_test.txt gives the test dataset on which the model is to
be evaluated.


The images_manufacturer_* gives the picture number, aircraft manufacured - Boeing, Airbus ... -. The trainval, train and test subsets are defined as for
images_family_*. The images_variant_* gives the picture number, aircraft
vairant i.e : the specific model, 707-320, A340-400 etc ... . The trainval,
train and test subsets are defined as for images_family_*. The images_*
gives the list of pictures. The trainval, train and test subsets are defined as
for images_family_*. The manufacturers.txt gives the list of manufacturers
found in the dataset. The variant.txt gives the list of variants found in the
dataset.

The picture in the trainval, train and test subset are the same for each list -
images_manufacturer_*, images_variant_*, images_*, images_family_*
You are asked to analyse the data and find a data-driven approach that allows
inferring - as best as you could :
* the aircraft manufacurer, by training the model on the images_manufacturer_trainval.txt
dataset and testing on the images_manufacturer_test.txt
* the aircraft family, by training the model on the images_family_trainval.txt
dataset and testing on the images_family_test.txt
