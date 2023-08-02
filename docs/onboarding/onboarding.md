# Onboarding document

## Table of contents

- [Onboarding document](#onboarding-document)
  - [Table of contents](#table-of-contents)
- [General description](#general-description)
- [First steps - software setup](#first-steps---software-setup)
  - [Cloning git repositories](#cloning-git-repositories)
  - [Discord](#discord)
  - [Jira](#jira)
- [Techstack](#techstack)
  - [Colab vs Jupyter](#jupyterlab-vs-google-colab)
  - [Dash - webapp framework](#dash)
  - [Amazon SageMaker](#amazon-sagemaker)
  - [PyTorch model](#pytorch-model)
  - [Explainable AI (XAI)](#explainable-ai)
- [Research - deep learning](#research---deep-learning)
  - [Emotion Recognition in Images](#emotion-recognition-in-images)
  - [Explainable Model from FER paper](#explainable-model)
- [Dog emotions recognition](#dog-emotions-recognition)

# General description

A project dedicated to the creation of an application that allows you to get emotional dog photos based on videos.

The basic concept is a create simple desktop app which will allow user to test the model (probably neural networks) that we are going to provide.

The app will comprise of two modules - API and model.

# First steps - software setup

If you may have any questions, do not hesitate to ask anyone in our team or post
them on the `help` channel on discord.

These items is expected to be a checklist, make sure you have access to
everything.

## Cloning git repositories

- create a workspace for all repos
  - `mkdir deeplab`
  - `cd deeplab`
- clone the main repository
  - `git clone https://github.com/DeepLabPG/deeplab.git`
- initialise the submodules
  - `git submodule update --init --recursive`

## Discord

- the main way of communication
- you should've already received access to our channel

## Jira

- send your email to Michał Kopczyński
- click the link you received in the email to create an account or log in to an
  existing one

## Techstack
  
### JupyterLab vs Google Colab
  ***Feature comparison:***
 - Jupyter has worse portability than Colab, it usually runs on local hardware, but you can support Jupyter for multiple users through JupyterHub. This hub can run a remote server (gcp, aws, ...). Colab runs on Google server, and stores files in google drive, so it provides access to files anywhere.
 - JupyterLab uses RAM, CPU and storage, so it is recommended for high-end computers. Colab uses computing power from Google server, so it gives you access to more RAM and CPU (about 13GB Ram, 70GB disk storage, K80/Tesla T4 GPU 15GB).
 - Jupyter is preferred when working alone, and Colab for teamwork.
 - Jupyter can work for extended periods of time, while Colab works 12/24 hours and the process can be disrupted by Google.
 - Jupyter requires installation of most libraries, while Colab comes with pre-installed libraries.
 - Jupyter is more secure because you store the data on the hard drive, while colab is not required for very sensitive work.
 
  ***Summary***
  Google Colab is more convenient for teamwork and more affordable to work with. It is less secure, but the specifics of our project do not require it. It gives access to computing power that we will not achieve using Jupyther. 
 In conclusion, Google Colab seems to be a better solution because of the project specifications and the organization of the team.
 
 ### Dash
 Dash is the original low-code framework for rapidly building data apps in Python, R, Julia. It is written on top of plotly.js and react.js. It's particularly suited for anyone who works with data. Dash is mobile ready! The good thing is also that Dash is open-source project and can be used for commercial use without any payment.\
 There is exemplary product of Dash:
 ![Zrzut ekranu 2022-10-15 o 11 23 09](https://user-images.githubusercontent.com/77082422/195979160-7fe2f6f9-a756-4c40-8e13-abec0ea825af.png)
Dash docs is well-written so following the docs will allow you to easily make simple app in 10 minutes. 
There is a good tutorial how to deploy model to Dash webapp - https://towardsdatascience.com/deploy-machine-learning-model-using-dash-and-pipenv-c543569c33a6 \
We will handle it later.
 
 ### Amazon SageMaker
 Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy ML models quickly. It is very powerful tool. For less than 1$ we can build, train and deploy our model but to be honest I do not know if we really need that. It is the topic to discuss
 
 ### PyTorch model
 He is using in his model PIL library and pretrained PyTorch model to first of all clean the images. He is removing the photos where there are no dogs - typically data cleaning.\
 The next step is to load data as DataFrame using pandas (labeled data - that is the step that we have take care of).\
 Then, there is a step to create the model class - loading data, augmenting images.\
 The next step is to divide the dataset to test and train set - typically ML workflow.\
 Subsequently, creating DataLoaders (PyTorch feature) with given number of epochs and batch size (in our model we will pick it by ourselves).\
 The last step is to create Neural Network class (ConvNET architecture - conv layer (stride 2 at the beginning), max-pool layer, dense layer), choose the optimizer and loss function - after that, the training is started. \
 The accuracy is around 60% but I think that we can achieve more by improving the model approach. \
 At the very beginning to achieve some results we can base our work on that model. \
 This is also topic to discuss - maybe we have to think about the GANs...\
 Link to the interesting article: https://www.researchgate.net/publication/325808679_Emotion_Classification_with_Data_Augmentation_Using_Generative_Adversarial_Networks
 
 ### Explainable AI 
 Explainable AI is not a technology per se, but the methodology of applying a model to the specific problem and understanding the decision process behind its actions. The key concept of XAI is "white-box", as opposed to the "black-box", meaning that every step of a deployed model should be understandable even by someone outside the world of AI and ML. Such transparency allows for continuous model evaluation, that could benefit teams other than the dev team. Easy to understand models are also easier to debug and scale. XAI algorithms should be constructed with 3 principles in mind; 1. Transparency 2.Interpretability and 3. Explainability. In our case, implementing XAI methodology would mean presenting the reasoning behind our model predictions in a clear and understandable way. 
Further reading:
https://www.ibm.com/watson/explainable-ai
Interesting paper about presenting neural nets as decision trees, making them more explainable and even computationally superior:
https://arxiv.org/abs/2210.05189
 
 # Research - deep learning
 
 ### Emotion Recognition in Images
  *Emotion recognition involves collecting signals by sensors and looking for patterns based on labeled data sets.*
 
 ***Ways of recognizing emotions***
 - Facial expression is one way to obtain information about emotions. The software traces key points on the face and makes matches to possible emotions. The problem   with this method is its low effectiveness in real life. The key points on the face are not always visible and the software can have trouble identifying them. The method also does not take into consideration the activity being performed, sometimes facial expressions have nothing to do with emotions. Context strongly influences the way we perceive the emoticons.
 - Using CV to recognize emotions from pictures containing whole profiles of people. The proposed model consists of two parts:
    - One fully convolutional extracts the features of a person from the separated part of the photo.
    - The other, also fully convolutional, recognizes the features of the environment and the context of the photo.
 - Both parts are then merged into one fully connected layer and we separate the features of the person from the context. **Proposed loss function is regression.**
 
 ***Some words about dogs emotions***
 
 Scientists believe that dogs, once emotionally formed, experience similar emotions as a two- to two-and-a-half-year-old child. Dogs have all the basic emotions such as joy, sadness, anger, distaste, and love, but they are unable to experience more complex emotions such as guilt, pride, or shame.
 The dog's body language includes many aspects such as tail wagging, raised hackles, posture, raised ears, and finally facial features.
 
 ***Summary***
 
 The key question for our project is the selection of photos. We may be limited to photos of the dog's muzzle preferably in the frontal position, in which case the model will be less useful in real life so commercially. A more complex approach to the subject poses the problem of a suitable dataset containing postures of entire dogs. The third approach, even more complex, includes the silhouette of the entire dog with a focus on its muzzle. The human emotion model described above was based on the Emotic Dataset, where the people studied are singled out of context by a red box. Such an approach in our case would require a large dataset preparation. 
 
 ### Explainable Model
 New approach, which is the XAI - Explainable AI has appeared to make the machine learning algorithms more understandable. This approach improves transparency of the models and trust of them. \
 Two most frequent interpretation methods in deep learning:
 1. Saliency Map
 2. Grad-CAM
 
 1) This map is used to clarify the importance of some areas in the input image that might be involved in generating the class prediction.
 Saliency is calculated at a particular position by the degree of the diﬀerence between that position and its surrounding.\
 The first step is to calculate the class saliency map which is 
 
  ![Zrzut ekranu 2022-10-15 o 11 50 24](https://user-images.githubusercontent.com/77082422/195980115-267b5e6a-2363-4096-afd8-68f33a7d16d2.png) 
  
 W is the derivative of the class score S with respect to image I:
 
  ![Zrzut ekranu 2022-10-15 o 11 51 29](https://user-images.githubusercontent.com/77082422/195980151-2b57b77d-0476-4bd2-b2bb-3e365e3e76a0.png) 
 
 2) This map speciﬁes each neuron’s important values for a speciﬁed decision of interest; it utilizes the gradient information ﬂowing into CNN’s last convolutional layer. In other words, it shows which regions in the image are related to this class. In this approach, 2 methods are combine: 
 - Class Activation Map (CAM)
 - Guided Backpropagation
 To specify values of each neuron for a specified choice we have to use gradient information at the last convolutional layer.\
 First of all, determine the importance of weights:
 
 ![Zrzut ekranu 2022-10-15 o 11 56 33](https://user-images.githubusercontent.com/77082422/195980339-cf6a12bf-71ba-4037-8ed6-c61fb833fa99.png)
 
 Z is the number of pixels in the feature map, the gradient is the gradient of score y for class c with respect to the feature map activation Ak.
 After that, we have to apply ReLU function on the weighted combination of forward activation maps to determine the localization map Grad-CAM: 
 
 ![Zrzut ekranu 2022-10-15 o 11 58 45](https://user-images.githubusercontent.com/77082422/195980415-1a890f8c-2ba7-43bd-a80e-d2f9a84907c0.png)
 
 Thanks to use of ReLU we are able to highlight only the positive effect on the class of interest.
 The model of classifier in the paper is constructed with 3 Conv layers with a 3x3 conv kernel and the stride value of 1, followed by a max-pooling layer with a 2x2 kernel. The number of filters is 32, 64, 128 for three layers respectively. The 1st layer has 256 nodes at the fully connected layers, and the second has 128 nodes with a dropout ratio of 0.5. At the final, there are six nodes for classifying the six basic emotions.

Note - this is also the topic to discuss. In addition, if our dataset is not going to be large, we can use classic ML methods, which will classify emotions also with the good accuracy like SVM or PCA.

 ### Dog emotions recognition

   According to scientist dogs can show facial emotions with a wide range. In saying that it’s more of a mixture of whole body language, face and even the way dogs are breathing. Picking an emotion based only on facial expressions really limits our capability of high accuracy, adding whole body can help to bump it up but it is still far from high accuracy. Research shows that People recognize anger and happiness well ( about 80%), sadness and fear on about 50% occasions. Only 20% people recognize disguist and surprised emotions. Also important aspect is that dog facial expresion can be neutral, not shwoing any emotion.    
   
![Zrzut ekranu 2022-10-23 173140](https://user-images.githubusercontent.com/115160997/197401150-ed985096-af12-421b-be25-fb7f9056a1a0.png)
 
According to scientist there are 5-6 primary emotions that hard-wired to our brains. Secondary emotions are more of a blend of primary ones and are often not easy to pick up on. In saying that dogs do have some particullar facial expressions that can widen our emotions list up to about 10.  
***After further research I propose that we label and work with these (facial) emotions:***


- neutral /easy,medium,hard           (to recognize)
- provoked aggression (stage 1 aggresion)  /easy 
- anger  (stage 2 aggresion) /easy 
- sadness  /medium 
- fear   /medium 
- happy,joy   /easy 
- courious,surprised      /easy,medium 
- pleasure,relaxation    / easy 
- lip licking /easy 
- pain /hard 
- disguist /hard


 ***To describe our methodology I propose these articles:***
  - https://www.rd.com/list/what-dog-facial-expressions-really-mean/
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8614696/
  - https://manukavet.com.au/you-can-identify-your-dogs-emotion-through-its-facial-expressions/
  
Which emotions do we pick and work on is a matter to speak on at next weekly meeting.
