# neural-networks

I would like to create a neural network that identifies birds which land in the
tree outside my window.  In doing so I hope to learn about neural networks, 
Python, and Git.

To test the current code I have trained small networks to predict the output of
an or gate, and it worked.

I have used the following abbreviations which I hope makes the code easier
to read:

    W      Weights
    b      Bias values
    Z      Linear activations
    A      Non-linear activations
    Aprev  Non-linear activations of the previous layer
    Yhat   Output layer activations
    dW     Differential of the cost function w.r.t. the weights
    db     Differential of the cost function w.r.t. the bias values
    dA     Differential of the cost function w.r.t. the non-linear activations
    dYhat  Differential of the cost function w.r.t. the output layer activation
    dAdZ   Differential of the non-linear activations w.r.t the linear
           activations

My next steps will be to introduce more hyperparameters and to train on more 
complex data.  I have started developing a digit method below for training a 
model to recognise hand-written digits.

Have a great day!

