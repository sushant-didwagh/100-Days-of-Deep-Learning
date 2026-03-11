CNN basic
3️⃣ Dataset
CIFAR-10 Dataset

The CIFAR-10 dataset is a widely used dataset for image classification tasks.

Features of the dataset:

Total Images: 60,000

Training Images: 50,000

Test Images: 10,000

Number of Classes: 10

Image Size: 32 × 32 × 3 (RGB)

Classes in CIFAR-10

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

Each image belongs to one of these 10 categories.

4️⃣ Methodology
CNN Architecture

A Convolutional Neural Network (CNN) is designed specifically for image processing tasks. It extracts features using convolution filters and classifies images using fully connected layers.

Typical CNN components:

Convolution Layer

Activation Function (ReLU)

Pooling Layer

Fully Connected Layer

Softmax Output Layer

Model Design

Students should experiment with different architectures.

Possible architecture example:

Input Image (32×32×3)

↓

Conv Layer (32 filters)

↓

ReLU

↓

Max Pooling

↓

Conv Layer (64 filters)

↓

ReLU

↓

Max Pooling

↓

Conv Layer (128 filters)

↓

Flatten

↓

Fully Connected Layer

↓

Dropout

↓

Output Layer (10 classes)

Hyperparameters to Tune

Hyperparameters significantly affect model performance.

Learning Rate

Controls how fast the model learns.

Examples:

0.1

0.01

0.001

Too high → unstable training
Too low → very slow learning

Optimizer

Used to update weights during training.

Common optimizers:

SGD (Stochastic Gradient Descent)

Simple

Slower convergence

Adam (Adaptive Moment Estimation)

Faster learning

Automatically adjusts learning rate

Batch Size

Number of samples processed in one iteration.

Common values:

32

64

128

Small batch → stable learning
Large batch → faster training

Dropout

Dropout randomly disables neurons during training to reduce overfitting.

Typical values:

0.2

0.3

0.5

Number of Convolution Layers

Students should experiment with:

2 layers

3 layers

4 layers

5 layers

More layers → deeper feature extraction
But too many layers → overfitting or slow training

Number of Filters

Filters detect image patterns like edges, textures, and shapes.

Common filter sizes:

32

64

128

More filters → better feature extraction but more computation.

Data Augmentation

Data augmentation increases dataset diversity by modifying images.

Techniques used:

Horizontal Flipping

Randomly flips images horizontally.

Example:
Car facing left → flipped → facing right.

Rotation

Rotates images slightly.

Example:
Rotate image by 10° – 20°.

Random Cropping (Optional)

Randomly crops part of the image.

Helps model learn robust features.

5️⃣ Evaluation Metrics

Model performance is evaluated using the following metrics.

Training Accuracy

Accuracy on the training dataset.

Indicates how well the model learns training data.

Validation Accuracy

Accuracy on validation data used during training.

Helps detect overfitting.

Test Accuracy

Accuracy on unseen test dataset.

Represents real-world performance.

Loss Curves

Plot graphs for:

Training Loss

Validation Loss

Interpretation:

Loss decreasing → model learning
Loss increasing → possible overfitting

6️⃣ Observations

Students should analyze:

Effect of Learning Rate

Large learning rate → unstable training

Small learning rate → slow convergence

Effect of Optimizer

Adam usually converges faster than SGD.

Effect of Data Augmentation

Improves generalization and reduces overfitting.

Effect of Dropout

Higher dropout reduces overfitting but may reduce accuracy if too high.
