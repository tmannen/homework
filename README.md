###Berkeley Deep Reinforcement Learning course

##Homework 1

In homework one we used Behavioural cloning and the DAgger algorithm to learn different scenarios.

#Behaviour cloning:

- Get expert/human labeled data as observations and actions it takes. Save this to a variable or a file.
- Use this data to learn a new model. Actions are targets, observations are input. An L2 loss can be used, so the model targets converge to the expert model

(Done in file run_custom_bc.py)


#DAgger:

- Get some expert/human labeled data first as in Behavioural cloning and train a model for some epochs.
- Run your model on new data. Take new actions based on your model, and have the expert model 'label' these new observations according to it.
The custom model will explore areas the expert data might not have, and we need to know what to do in these cases. Add this new data to
your old data, and train again.

(Done in file run_custom_dagger.py)