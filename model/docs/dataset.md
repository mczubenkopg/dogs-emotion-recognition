# Dataset information
## Table of contents
- [Dataset information](#dataset-information)
  - [Table of contents](#table-of-contents)
- [General information and making subsets](#general-information-and-making-subsets)
- [Labeling data](#labeling-data)
- [Final thoughts about dataset creation](#final-thoughts)
  
# General information and making subsets

The creation of a dataset cannot go without combining different datasets. Creating a dataset is dependent on labeling. We have decided to highlight five features, in labeling we will follow Dr. Chubenko's dataset. The other datasets considered for creating the general subset are ***Tsinghua Dogs Dataset***, ***Stanford Dogs Dataset***, and our own contribution of photos. 
The general subset can be created before labeling begins by creating a common photo folder. Then, after labeling against the master subset (Dr. Czubenko's dataset) is complete, the sets can already be merged as a **Python Data Frame** using, for example, the **pandas** library. The method works similarly to splitting the dataset into a teaching and a testing dataset.

# Labeling data

Data labeling involves assigning tags or labels to data cells, needed for model learning.
The specifics of our project require the development of a specific dataset. Labeling of such a dataset will be carried out based on the sample dataset provided by Dr. Czubenko.
Of the many labeling options used in the market, we are interested in two approaches:
 - ***Manual labeling*** -
  this involves manually assigning labels to dog photos. A widget for JupyterNotebook - [pidgey](https://github.com/wbwvos/pidgey) - can be very helpful in this approach,   which speeds up the work. An interesting way of labeling data is to add labels in the form of numerical labels to the photo and saving in csv format. The way is           convenient for post-merging subsets [[LINK](https://towardsdatascience.com/label-your-images-easily-using-this-jupyter-notebook-code-4102037b7821)].
  
  We can speculate that labeling 500 photos per hour, the whole team working for three hours per week, would prepare 7500 photos per week and 30000 photos per month.
 - ***Semi-automatic labeling*** -
	this involves preparing a certain number of photos and creating a model that labels successive photos by itself. One should supervise the model's work by making corrections and teaching the model on a larger, better set until it reaches a satisfying state.

The semi-automatic approach is several times faster, does not require the involvement of the entire team, is less laborious and more educational, however, it is fraught with more risk (possible poor quality of the dataset) and is definitely more complicated.

# Final thoughts

Our assumption about labelling the images was not quite correct, as our main task is image classification, not object detection. 
I have come up with a way in which we can prepare our dataset. We have exactly 5 features to classify, so everyone will take one image folder from our dataset and clear it of unsupported images. Then, thanks to a script I wrote (rename.sh) in the folder, it will rename the files to the type "1-emotion.jpg" so that we can treat these names as id in our csv file, which will allow us to quickly tag the image. Example of operation:
![image](https://user-images.githubusercontent.com/77082422/200986148-b017f5b1-154a-4e76-8fc0-cee6cb937e96.png)

We can then quickly create a csv file of our images - example:
![image](https://user-images.githubusercontent.com/77082422/200986376-0b5ee090-7833-4a9c-a233-cfae7d880c5a.png) \
Now it is enough to know how many photos are in a folder and to quickly drag the slider of a column to the last photo and all the photos are tagged.

