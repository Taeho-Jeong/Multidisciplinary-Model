# Multidisciplinary-Model
Multidisciplinary model of IFEWs coded in Python using the OpenMDAO package.

- **`IFEWs_model_v5.py`**\
Python OpenMDAO model that represents the multidisciplinary Iowa Food Energy Water system.

- **`batch_analysis.py`**\
Python script used to run the IFEWs_model for each county at each year. Accesses the input data of crop yield, animal agriculture and weather to run the IFEWs_model to give the Results.csv

- **`tool_data_plot.ipynb`**\
Jupyter Notebook script that plots the model output for over time. The tool is interactive and can show the model data for each county. Also plots the ethanol production validation plot.

- **`tool_counter_map.ipynb`**\
Jupyter Notebook script that plots the model outputs as a contour on the map of Iowa. The tool is interactive and can show the contour map at each year for each model output.
