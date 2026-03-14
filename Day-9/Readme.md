Image Classification using ANN 
steps building ANN architecture to predict hand written digits
1. acccess mnist dataset in train-test splited form
2. scale images into 0 to 1 range by dividing 255 by each pixels
3. convert image pixels into 1 dimentional flatten layers using Flatten library
4. create hidden layer and final layer architerture -> make sure if in final layer multiple neurons must use softmax architecture
5. compile model loss function (sparse_categorical_crossentropy) not need to encoding dataset, optimizer(adam), metrics(accuracy)
6.  train model using parameters - X_train, y_train, epochs=10, validation_split=0.2
7.  modle return maximum probablity of each image with compare to final layer neurons , using argmax function
8.  using accuracy_score libarary get accuracy
