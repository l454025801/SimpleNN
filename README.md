# Simple Neural Network

A library for Deep learning neural network. Currently support linear DNN. Require Eigen library for matrix manipulation.

usage:

```
int main(){

  //create SiN object
  SimpleNN YourNetwork;
  
  // add layer, set cost function, parameter initiator, iteration times, learning rate;
  SimpleNN.add_layer(num_of_neuron, "activation_type")
  SimpleNN.set_cost("Binary Cross-entropy");
  SimpleNN.set_initiator("HE");
  SimpleNN.iter_times = 2501;
  SimpleNN.alpha = 0.0075;  

  SimpleNN.X_train = your_train_X;
  SimpleNN.y_train = your_train_y;

  // train
  SimpleNN.train();

}
```
