# Building Deep Learning models with PyTorch

# Project 1: Facial Keypoint Features extraction

In this project, knowledge of computer vision techniques and deep learning architectures will be combined to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face!

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face. Some examples of these keypoints are pictured below.

### The project will be broken up into a few main parts:

- Loading and Visualizing the Facial Keypoint Data

- Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

- Facial Keypoint Detection Using Haar Cascades and your Trained CNN


# Project 2: Image Captioning

In this project, a neural network architecture that allow to automatically generate captions from images will be created, it will combine pretrained ResNet50 model with custom LSTM Decoder Neural Network that will generate a custom description based on image sent to the network.

After using the Microsoft Common Objects in COntext (MS COCO) dataset to train your network, you will test your network on novel images!



## LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!

## Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are `embed_size`.

## Sequential Inputs
So, an LSTM looks at inputs sequentially. In PyTorch, there are two ways to do this.

The first is pretty intuitive: for all the inputs in a sequence, which in this case would be a feature from an image, a start word, the next word, the next word, and so on (until the end of a sequence/batch), you loop through each input like so:
```
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
```
The second approach, which this project uses, is to give the LSTM our entire sequence and have it produce a set of outputs and the last hidden state:

# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state

# Add the extra 2nd dimension
```
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
```
