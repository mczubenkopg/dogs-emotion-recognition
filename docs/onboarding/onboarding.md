# Onboarding document

## Table of contents

- [Onboarding document](#onboarding-document)
  - [Table of contents](#table-of-contents)
- [General description](#general-description)
- [Explainable AI (XAI)](#explainable-ai)
- [Research - deep learning](#research---deep-learning)
  - [Emotion Recognition in Images](#emotion-recognition-in-images)
  - [Explainable Model from FER paper](#explainable-model)
- [Dog emotions recognition](#dog-emotions-recognition)

# General description

A project dedicated to the creation of an application that allows you to get emotional dog photos based on videos.
The basic concept is a create simple desktop app which will allow user to test the model that we are going to provide.
The app will comprise of two modules - API and model.

# Explainable AI 
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
