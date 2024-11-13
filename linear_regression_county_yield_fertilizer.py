import numpy as np
import openmdao.api as om
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# Load Animal Agriculture data -----------
loc_AA = 'animal_agriculture_data/IFEW_Counties_1997_2019.csv'
df_AA = pd.read_csv(loc_AA) # Animal Agricultural dataframe

# Crop field yields (bu/acre) -----------
corn_yield = df_AA["CornGrainYield_bupacre"]
soybean_yield = df_AA["SoybeansYield_bupacre"]

# Commercial N (kg/ha) -----------
RCN_data = df_AA["CommercialN_kg_ha"]

# Manual N (kg/ha) -----------
manual_data = df_AA["ManureN_kg_ha"]


counties = [
    "ADAIR", "ADAMS", "ALLAMAKEE", "APPANOOSE", "AUDUBON", "BENTON", "BLACK HAWK", "BOONE", "BREMER", "BUCHANAN", 
    "BUENA VISTA", "BUTLER", "CALHOUN", "CARROLL", "CASS", "CEDAR", "CERRO GORDO", "CHEROKEE", "CHICKASAW", "CLARKE", 
    "CLAY", "CLAYTON", "CLINTON", "CRAWFORD", "DALLAS", "DAVIS", "DECATUR", "DELAWARE", "DES MOINES", "DICKINSON", 
    "DUBUQUE", "EMMET", "FAYETTE", "FLOYD", "FRANKLIN", "FREMONT", "GREENE", "GRUNDY", "GUTHRIE", "HAMILTON", 
    "HANCOCK", "HARDIN", "HARRISON", "HENRY", "HOWARD", "HUMBOLDT", "IDA", "IOWA", "JACKSON", "JASPER", 
    "JEFFERSON", "JOHNSON", "JONES", "KEOKUK", "KOSSUTH", "LEE", "LINN", "LOUISA", "LUCAS", "LYON", 
    "MADISON", "MAHASKA", "MARION", "MARSHALL", "MILLS", "MITCHELL", "MONONA", "MONROE", "MONTGOMERY", "MUSCATINE", 
    "O’BRIEN", "OSCEOLA", "PAGE", "PALO ALTO", "PLYMOUTH", "POCAHONTAS", "POLK", "POTTAWATTAMIE", "POWESHIEK", "RINGGOLD", 
    "SAC", "SCOTT", "SHELBY", "SIOUX", "STORY", "TAMA", "TAYLOR", "UNION", "VAN BUREN", "WAPELLO", 
    "WARREN", "WASHINGTON", "WAYNE", "WEBSTER", "WINNEBAGO", "WINNESHIEK", "WOODBURY", "WORTH", "WRIGHT"
]

# Dictionary to store the slices
slices = {}
start = 0
step = 23

# Generate the ranges for each county
for county in counties:
    end = start + step
    slices[county] = f"{start}:{end}"
    start = end

user_input = input("County name : ")
user_input = user_input.upper()

# Check if the user input is one of the keys in the slices dictionary
if user_input in slices:
    start_index, end_index = map(int, slices[user_input].split(':'))
    sliced_yield_c = corn_yield[start_index:end_index]
    sliced_yield_s = soybean_yield[start_index:end_index]
    sliced_RCN = RCN_data[start_index:end_index]
    sliced_MN = manual_data[start_index:end_index]
else:
    print(f"'{user_input}' is not a county in Iowa.")
    
RCN_data = sliced_RCN.to_numpy().reshape(-1,1)
corn_yield = sliced_yield_c.to_numpy().reshape(-1,1)
regrC = linear_model.LinearRegression()
regrC.fit(RCN_data, corn_yield)
    

'''
plot of the data
'''
import matplotlib.pyplot as plt

yield_c = sliced_yield_c.to_numpy()
RCN = sliced_RCN.to_numpy()

sorted_indices = np.argsort(RCN)
sorted_yield = yield_c[sorted_indices]
sorted_RCN = RCN[sorted_indices]

plt.plot(sorted_RCN, sorted_yield, marker='o')  # Plot x and y using a line and markers
plt.title('Listoryne Plot of x vs. y')  # Add a title
plt.xlabel('RCN')  # Add x-axis label
plt.ylabel('corn yield')  # Add y-axis label
plt.grid(True)  # Add gridlines
plt.show()  # Display the plot

'''
plot of the NN approximate
'''
NN_test_x = np.arange(50,100,1).reshape(-1,1)
NN_test_y = regrC.predict(NN_test_x )

plt.plot(NN_test_x, NN_test_y, marker='o', color = 'red')  # Plot x and y using a line and markers
plt.title('NN Test')  # Add a title
plt.xlabel('RCN')  # Add x-axis label
plt.ylabel('corn yield')  # Add y-axis label
plt.grid(True)  # Add gridlines
plt.show()  # Display the plot

'''
User RCN Input
'''
user_RCN = input("Amount of commercial Nitrogen (kg/ha) : ")
user_RCN = np.array([user_RCN], dtype=int).reshape(-1,1)
user_yield = regrC.predict(NN_test_x )


print("Approximated corn yield : {} bu/acre".format(user_yield[0]), flush=True)
