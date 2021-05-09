Steps to run my model:
1. Put the three jupyter notebooks (“data_transform.ipynb”, “train.ipynb”, “test.ipynb”) into the same directory as the data files (train.npy, etc). 
2. Run “data_transform.ipynb” which will stack each dataset and save the stacked datasets in the current directory. 
3. Then run the “train.ipynb” and keep the hyper parameters as the values they are in the file. 
   1. The model is: 
Simple_MLP(
  (net): Sequential(
    (0): Linear(in_features=1240, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=1024, out_features=1024, bias=True)
    (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Linear(in_features=1024, out_features=1024, bias=True)
    (7): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): Linear(in_features=1024, out_features=1024, bias=True)
    (10): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
    (12): Linear(in_features=1024, out_features=512, bias=True)
    (13): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): Linear(in_features=512, out_features=256, bias=True)
    (16): ReLU()
    (17): Linear(in_features=256, out_features=71, bias=True)
  )
)


   2. The hyparameters include: 
      1. context = 30
      2. num_workers = 8 if cuda else 0
      3. input_layer = [(2 * context + 1) * 40]
      4. output_layer = [71]
      5. hidden_layers = [1024,1024,1024,1024,512,256]
      6. Learning rate = 0.001
      7. Number of epochs = 10 (in my model, I use the model from the 6th epoch.)
4. Then compare the accuracies of models by their titles. Model of each epoch is saved in the same directory as the code files. You are able to compare the evaluation accuracies of each model based on its title (the tile is in the format like “trial5epoch6acc75.66240361962392.pth”). Whether “acc75.66240361962392” means the evaluation accuracy is 75.66240361962392 percent. By manually comparing the accuracies, you pick the model with the highest accuracy as the test model. 
5. Then you run “test.ipynb”. The only place you need to modify in this file is in the line 
model_test.load_state_dict(torch.load("trial5epoch6acc75.66240361962392.pth”)). 
You need to change the string in the torch.load to the title of the model file you choose.
