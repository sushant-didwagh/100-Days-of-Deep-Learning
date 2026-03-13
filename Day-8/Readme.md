Starting with Artifical Neural Network.
steps to make ANN:
1. collece and preprocess data
2. split data into training and testing
3. Scale data or data augumentation  -> using StandardScaler() function
4. building model architecture
   a. the Model is sequential use -> sequential Function "model = Sequential()"
   b. accouding to needs create layers known as hidden layers "Dense(3,activation='sigmoid', input_dim=11"
      total layers are 3, activation is sigmoid and inputs are 11.
5. loss function is binary_crossentropy with optimizer "adam"
6. train the model with epoches -> after training model calculate weights, and bais
7. decide threshold for classification -> greter 0.5 yes, else No
8. calculate accuracy
9. by your need manage layers
   
