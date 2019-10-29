# Problem Statement

Interpretation of hand written or drawn characters is an important area for machine learning because of its benefits: it helps the USPS and other logistics agencies interpret countless handwritten labels.  Handwriting recognition has a wide variety of uses: from digitalizing paper records to recognizing who wrote a piece of text. Because systems to read handwritten characters have so much potential, it is important to have them perform as well as possible.   Having such a system will allow a client or company to greatly improve their business efficiency.

# Executive Summary

The Quick, Draw! game can be found [here](https://quickdraw.withgoogle.com/).

The data set we used can be found [here](https://github.com/googlecreativelab/quickdraw-dataset#get-the-data).

For the purposes of this project, we used the data in two formats:

- The raw datafiles from Google which we used for its metadata: date/time, country, and whether or not the drawing was recognized.

- The numpy bitmap files which are simplified from the 4D drawing array in the image data from the raw data: they have been resized to 28x28 and centered.

The data have already been formatted for us, so we did not have to do any cleaning in the traditional sense.  However, there was worked to be done before we could begin modeling with the bitmap files.

Firstly, we had three different files which had to be concatenated together and we then saved the concatenated files as a new .csv file.

The files were natively black & white so all we had to do was scale each pixel: they range in value from 0 to 255 so we just divided each by 255 and forced each pixel to be floats.  Once that was finished, we then were able to force each row in the dataframe to be a 28x28x1 array.  After this point, we were able to start the modeling process.

We used a Keras convolutional neural network to classify each image.  The geometry of the network is fairly simple: two convolutional and maxpooling layers followed by three hidden layers.  We were concerned about overfitting, which is common with neural networks, we decided to add in several regularization methods: $\ell$<sub>2</sub> regularization in the hidden nodes, dropout between each node, and finally early stopping in the fitting stage.

# Conclusions & Recommendations

# Links

The presentation for this project can be found [here]()

The blog post for this project can be read [here]()