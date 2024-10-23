# KDDProject3

Usage: 

- ``python -m venv kdd3`` or use the provided env
- ``source ~./KDD3/kdd3/bin/activate`` which activates the venv
- ``pip install -r requirements.txt`` install required packages
- ``python get_data.py`` or if on slurm ``sbatch init.sh `` to get the data from huggingface locally
- Run relevant model with respective shell script or py script i.e. ``python <name_of_model>.py |  sbatch <name_of_model>.sh``
- If using slurm: check ``logs`` for results of respective model, named ``<name_of_model>_<number_of_job>.txt``
- Otherwise, check console for output

Currently Supported Models:

- ``Ordinary Least Squares Regression (OLS)`` via ``ols.py | ols.sh ``
- ``Random Forest Regression (RFR)`` via `` rf.py | rf.sh``
- ``Gradient Boost Regressor (GBR) `` via ``gb.py | gb.sh ``
- ``Stochastic Gradient Descent Linear Regression (SGD)`` via ``sgd.py | sgd.sh``
- ``Deep Neural Network Regression (DNN) `` via ``dnn.py | dnn.sh``

WIP/Scrapped Models: 

- ``Long Short Term Memory(LSTM) `` via ``lstm.ipynb ``
- ``Support Vector Regression (SVR) ``

Source of Data: 
https://huggingface.co/datasets/gauss314/options-IV-SP500
Paper: https://www.overleaf.com/project/6718263f88142f32df7f5d90
