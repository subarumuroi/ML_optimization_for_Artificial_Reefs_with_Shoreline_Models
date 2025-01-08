# ML optimization for Artificial Reefs with Shoreline Models

Neural net and Gaussian Process Regressor methods for learning shoreline model outputs. 

#Make sure to change folder paths and everything else will run as long as you've setup your environment 

As the shoreline can be considered a 1D line averaged over multiple equidistant shoreline points, ML models can be used to converge towards a shoreline solution based on database with multiple process based models' shoreline predictions (mathematical computations or simulations). Though the ML approach would also incorporate natural and measured values through its process-agnostic data-driven methods, of interest here is the simulated data of submerged breakwaters (SBW or artificial reef) and their potential effects on shorelines according to coastal engineering models. 

The main dataset is 'AllRuns.csv', which have been preseparated into 'AllRuns_SingleTime_Train.csv' and 'AllRuns_SingleTime_Test.csv'. This ensures the sparse data contains crucial boundary information to make interpolation possible (As modeling is still quite tedious when generating the dataset required for a large multidimensional variable space (including: Wave height, wave period, SBW length, SBW width, and SBW distance from the shoreline (in the dataset for this demonstration))) There are also separations that focus on the middle point of the shoreline, effectively the shoreline perpindicular and adjacent to the submerged breakwater (ML was found to be able to predict the entire 1D line (51 output values in this dataset)  

'model' files and folders exist for posterity of published analysis, but the overall modeling is simple, fast, and using the given dataset, can be calculated in moments, as the dataset has been optimized to give the sparse space solutions from model results after 1 year of process-based model time elapse (The model space uses identical straight shorelines as starting points to measure the net effect after 1 year of wave forcing conditions).

'sensitivity-analysis.py' is for testing the optimized best.h5 model to see how sensitive it is to its input variables, and is still being tested.

'Average Coastline Erosion Predictor.py' is also still being tested, but this model was supposed to select the ideal hyperparameter tunings based on average of midpoint values (middle of shoreline values in separated dataset)  

'Hyperband Tester.py' was the model used to generate the nn model parameters in the machine learning for submerged breakwaters paper. 

Nick's stuff is Nick's stuff, it is what I've learnt from and good reference material.
(''SL_ModelComparison_sklearn.html' used as basis for model justification in published material compares various ML model performance on the prediction of midpoint values 'sklearn_SUB.py' has some functions that need to be read in first for these)





_____________________
Shoreline prediction from SBW/shoreline models
_____________________
Shoreline ANN Tester.py is a basic midpoint predictor using nn (neural network) model. it plots model learning and prediction performance and saves the model.

SMInputOutputDataset Exctractor series are the code used to convert the MIKE process files data for ML (machine learning), meaning if you can access MIKE (free for students), you can also generate your own dataset for specific artificial reef design parameters of interest to see how they can dynamically influence shorelines for accretion and erosion.

'Artificial Reef Optimizer.py' is a full-scale model that shows how the dataset can be used to learn the variables of interest and model nn(neural network) and gpr(Gaussian Process Regressor) to predict the entire shoreline. The model smooths and preprocesses the data, and calculates both models predictive abilities (MAE, MSE, R2), The nn has two layers, but you can add more. The model charts its learning and various relevant performance metrics (you can make edits to check different outputs charts). The model than uses MC (Monte Carlo) of the classic kind to rapidly predict the solution space across the multidimensional combinations to find an optimized SBW design. This can be significantly changed to optimize for the attributes that the coastal engineer seeks (as they are quite variable, contain trade-offs, and require complex experimental proofs to validate all these computational processes combined on top of each other.

and finally,

Shoreline prediction chart for GPR and NN.py is for posterity for published material and the coding used to generate the shoreline model prediction using the models in this repo. They are used to compare the separate ML models results and demonstrate how the learning method of the models influence their interpolative predictions. Further work is necessary to validate interpolation errors, and to test the sensitivty of the model to data, as the models here perform quite well, but they use constrained, idealised conditions, and in vivo processes, such as at real beaches with underwater structures are much more complex and dynamic than what had been modeled here to generate the dataset preceding the ML. 

However, the important contribution here, is that this modelling platform can be used to generate ML(machine-learning based) predictions from data-driven shoreline values, which could be applied to myriad data sources, or application to extend AI/ML into shoreline data and coastal predictions. Of particular interest in this study, was for submerged breakwaters (or artificial reefs), which require compuational simulations as they are to be designed and placed in sensitive coastal (and in this case, sandy)
environments, where limited data exists on the parametric effect of SBW and artificial reefs in this complex environment.


