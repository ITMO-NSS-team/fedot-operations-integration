# fedot-operations-integration
The repository contains the wrappers to launch FEDOT framework. The main goal: to compare the effectiveness of the implemented operations

The module is a system for checking the models implemented in FEDOT according to two indicators: 
* Velocity (the model fit quickly or slowly);
* Effectiveness (forecasting error has become larger or smaller)

---

## Initial assumption

The base initial assumption is a Pipeline of the following structure:
<img src="./docs/images/init_base_example.png" width="550"/>

Next, the new model is inserted instead of Random Forest and is perform launch on 4 tables 
for classification and on 3 tables for regression. 

The run is performed in two modes: 
* simple (hold-out validation, fit models with default parameters)
* advanced (cross-validation for each dataset, perform hyperparameters tuning on each fold)

## Composing correctness

In progress