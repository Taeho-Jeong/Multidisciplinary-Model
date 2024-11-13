import pandas as pd
import numpy as np


surrounding_counties = {
    'Adair': ['Guthrie', 'Madison', 'Union', 'Cass'],
    'Adams': ['Cass', 'Adair', 'Union', 'Taylor', 'Montgomery'],
    'Allamakee': ['Winneshiek', 'Clayton'],
    'Appanoose': ['Monroe', 'Wayne', 'Davis'],
    'Audubon': ['Carroll', 'Guthrie', 'Cass', 'Shelby'],
    'Benton': ['Tama', 'Linn', 'Iowa', 'Poweshiek', 'Black Hawk'],
    'Black Hawk': ['Bremer', 'Butler', 'Grundy', 'Tama', 'Benton', 'Buchanan'],
    'Boone': ['Greene', 'Webster', 'Story', 'Dallas'],
    'Bremer': ['Chickasaw', 'Floyd', 'Black Hawk', 'Butler', 'Fayette'],
    'Buchanan': ['Fayette', 'Delaware', 'Linn', 'Benton', 'Black Hawk'],
    'Buena Vista': ['Cherokee', 'Pocahontas', 'Sac', 'Clay'],
    'Butler': ['Floyd', 'Franklin', 'Grundy', 'Black Hawk', 'Bremer'],
    'Calhoun': ['Pocahontas', 'Webster', 'Greene', 'Sac'],
    'Carroll': ['Sac', 'Greene', 'Guthrie', 'Audubon', 'Crawford'],
    'Cass': ['Adair', 'Adams', 'Montgomery', 'Pottawattamie', 'Audubon'],
    'Cedar': ['Jones', 'Linn', 'Johnson', 'Muscatine', 'Scott', 'Clinton'],
    'Cerro Gordo': ['Floyd', 'Franklin', 'Hancock', 'Worth', 'Mitchell'],
    'Cherokee': ['Buena Vista', 'Ida', 'Plymouth', 'Sioux', 'O\'Brien'],
    'Chickasaw': ['Floyd', 'Howard', 'Winneshiek', 'Bremer', 'Fayette'],
    'Clarke': ['Lucas', 'Warren', 'Union', 'Madison', 'Decatur'],
    'Clay': ['Dickinson', 'Palo Alto', 'Buena Vista', 'O\'Brien'],
    'Clayton': ['Fayette', 'Allamakee', 'Dubuque', 'Winneshiek', 'Delaware'],
    'Clinton': ['Jackson', 'Jones', 'Cedar', 'Scott'],
    'Crawford': ['Harrison', 'Shelby', 'Carroll', 'Ida', 'Monona'],
    'Dallas': ['Greene', 'Boone', 'Story', 'Polk', 'Madison', 'Guthrie'],
    'Davis': ['Appanoose', 'Van Buren', 'Monroe', 'Wapello'],
    'Decatur': ['Clarke', 'Wayne', 'Ringgold'],
    'Delaware': ['Buchanan', 'Fayette', 'Clayton', 'Dubuque', 'Jones', 'Linn'],
    'Des Moines': ['Lee', 'Henry', 'Louisa'],
    'Dickinson': ['Osceola', 'Emmet', 'Clay', 'Palo Alto'],
    'Dubuque': ['Delaware', 'Jones', 'Jackson', 'Clayton'],
    'Emmet': ['Dickinson', 'Kossuth', 'Palo Alto'],
    'Fayette': ['Clayton', 'Buchanan', 'Bremer', 'Chickasaw', 'Winneshiek', 'Delaware'],
    'Floyd': ['Mitchell', 'Cerro Gordo', 'Butler', 'Chickasaw', 'Howard'],
    'Franklin': ['Butler', 'Cerro Gordo', 'Hardin', 'Wright'],
    'Fremont': ['Page', 'Mills', 'Montgomery'],
    'Greene': ['Calhoun', 'Boone', 'Dallas', 'Guthrie', 'Carroll'],
    'Grundy': ['Butler', 'Black Hawk', 'Hardin', 'Marshall', 'Tama'],
    'Guthrie': ['Greene', 'Carroll', 'Dallas', 'Adair', 'Cass', 'Audubon'],
    'Hamilton': ['Wright', 'Webster', 'Hardin', 'Story', 'Boone'],
    'Hancock': ['Winnebago', 'Kossuth', 'Wright', 'Franklin', 'Cerro Gordo'],
    'Hardin': ['Franklin', 'Grundy', 'Hamilton', 'Marshall', 'Story'],
    'Harrison': ['Monona', 'Shelby', 'Crawford', 'Pottawattamie'],
    'Henry': ['Jefferson', 'Washington', 'Des Moines', 'Louisa', 'Lee', 'Van Buren'],
    'Howard': ['Mitchell', 'Chickasaw', 'Winneshiek'],
    'Humboldt': ['Pocahontas', 'Wright', 'Webster', 'Kossuth'],
    'Ida': ['Cherokee', 'Sac', 'Crawford', 'Woodbury'],
    'Iowa': ['Benton', 'Johnson', 'Keokuk', 'Poweshiek', 'Washington'],
    'Jackson': ['Dubuque', 'Clinton', 'Jones'],
    'Jasper': ['Polk', 'Poweshiek', 'Marion', 'Story', 'Marshall'],
    'Jefferson': ['Keokuk', 'Van Buren', 'Henry', 'Washington', 'Wapello'],
    'Johnson': ['Cedar', 'Linn', 'Iowa', 'Washington', 'Muscatine'],
    'Jones': ['Dubuque', 'Delaware', 'Cedar', 'Jackson', 'Clinton', 'Linn'],
    'Keokuk': ['Washington', 'Iowa', 'Mahaska', 'Jefferson', 'Wapello'],
    'Kossuth': ['Humboldt', 'Palo Alto', 'Hancock', 'Winnebago', 'Emmet'],
    'Lee': ['Des Moines', 'Henry', 'Van Buren'],
    'Linn': ['Benton', 'Cedar', 'Jones', 'Johnson', 'Delaware'],
    'Louisa': ['Muscatine', 'Des Moines', 'Henry', 'Washington'],
    'Lucas': ['Marion', 'Clarke', 'Monroe', 'Warren'],
    'Lyon': ['Osceola', 'Sioux'],
    'Madison': ['Dallas', 'Warren', 'Clarke', 'Union', 'Adair'],
    'Mahaska': ['Keokuk', 'Iowa', 'Poweshiek', 'Marion', 'Wapello'],
    'Marion': ['Polk', 'Jasper', 'Lucas', 'Warren', 'Mahaska'],
    'Marshall': ['Hardin', 'Grundy', 'Tama', 'Story', 'Jasper'],
    'Mills': ['Pottawattamie', 'Fremont', 'Montgomery'],
    'Mitchell': ['Worth', 'Howard', 'Floyd', 'Cerro Gordo'],
    'Monona': ['Woodbury', 'Ida', 'Crawford', 'Harrison'],
    'Monroe': ['Lucas', 'Appanoose', 'Wapello', 'Marion', 'Wayne'],
    'Montgomery': ['Adams', 'Cass', 'Page', 'Fremont', 'Mills'],
    'Muscatine': ['Louisa', 'Cedar', 'Scott', 'Johnson', 'Washington'],
    'O\'Brien': ['Cherokee', 'Clay', 'Sioux', 'Plymouth', 'Osceola'],
    'Osceola': ['Dickinson', 'O\'Brien', 'Lyon', 'Sioux'],
    'Page': ['Montgomery', 'Taylor', 'Fremont'],
    'Palo Alto': ['Kossuth', 'Pocahontas', 'Emmet', 'Humboldt', 'Clay'],
    'Plymouth': ['Sioux', 'Cherokee', 'Woodbury', 'Ida'],
    'Pocahontas': ['Palo Alto', 'Humboldt', 'Webster', 'Calhoun', 'Buena Vista'],
    'Polk': ['Dallas', 'Story', 'Jasper', 'Marion', 'Warren'],
    'Pottawattamie': ['Harrison', 'Mills', 'Shelby', 'Cass'],
    'Poweshiek': ['Tama', 'Jasper', 'Mahaska', 'Iowa', 'Keokuk'],
    'Ringgold': ['Union', 'Decatur', 'Taylor'],
    'Sac': ['Buena Vista', 'Carroll', 'Ida', 'Calhoun'],
    'Scott': ['Clinton', 'Cedar', 'Muscatine'],
    'Shelby': ['Harrison', 'Audubon', 'Crawford', 'Pottawattamie'],
    'Sioux': ['Plymouth', 'O\'Brien', 'Osceola', 'Lyon'],
    'Story': ['Hamilton', 'Hardin', 'Boone', 'Marshall', 'Polk', 'Jasper'],
    'Tama': ['Marshall', 'Grundy', 'Benton', 'Poweshiek', 'Jasper'],
    'Taylor': ['Adams', 'Page', 'Ringgold'],
    'Union': ['Madison', 'Adair', 'Clarke', 'Ringgold', 'Adams'],
    'Van Buren': ['Henry', 'Jefferson', 'Lee', 'Davis'],
    'Wapello': ['Jefferson', 'Keokuk', 'Davis', 'Monroe', 'Mahaska'],
    'Warren': ['Madison', 'Polk', 'Marion', 'Lucas', 'Clarke'],
    'Washington': ['Louisa', 'Johnson', 'Jefferson', 'Keokuk', 'Iowa', 'Henry'],
    'Wayne': ['Lucas', 'Decatur', 'Appanoose'],
    'Webster': ['Hamilton', 'Humboldt', 'Calhoun', 'Boone'],
    'Winnebago': ['Worth', 'Hancock', 'Kossuth'],
    'Winneshiek': ['Allamakee', 'Chickasaw', 'Howard', 'Fayette', 'Clayton'],
    'Woodbury': ['Ida', 'Monona', 'Plymouth'],
    'Worth': ['Mitchell', 'Winnebago', 'Cerro Gordo'],
    'Wright': ['Hancock', 'Humboldt', 'Franklin', 'Hamilton'],
}

agricultural_districts = {
    "NW": ["Pocahontas", "Dickinson", "Emmet", "Buena Vista", "Clay", "Cherokee", "Sioux", 
        "Lyon", "O\'Brien", "Osceola", "Plymouth"],
    "NC": ["Kossuth", "Floyd", "Franklin", "Hancock", "Winnebago", "Wright", "Cerro Gordo", 
        "Mitchell", "Humboldt", "Webster", "Worth"],
    "NE": ["Allamakee", "Black Hawk", "Buchanan", "Bremer", "Chickasaw", "Clayton", "Delaware", 
        "Dubuque", "Fayette", "Howard", "Winneshiek"],
    "WC": ["Audubon", "Calhoun", "Carroll", "Crawford", "Greene", "Guthrie", "Harrison", "Ida", 
        "Monona", "Sac", "Shelby", "Woodbury"],
    "C": ["Boone", "Dallas", "Hamilton", "Jasper", "Marshall", "Polk", "Story", "Webster", "Tama"],
    "EC": ["Benton", "Cedar", "Clinton", "Iowa", "Jackson", "Johnson", "Jones", "Linn", "Muscatine", 
        "Scott"],
    "SW": ["Adair", "Adams", "Cass", "Fremont", "Mills", "Montgomery", "Page", "Pottawattamie", 
        "Taylor"],
    "SC": ["Appanoose", "Clarke", "Decatur", "Lucas", "Madison", "Marion", "Monroe", "Ringgold", 
        "Union", "Warren", "Wayne"],
    "SE": ["Davis", "Des Moines", "Henry", "Jefferson", "Keokuk", "Lee", "Louisa", "Mahaska", 
        "Van Buren", "Wapello", "Washington"]
}



watershed = {}
soil_type = {}

# Function to get surrounding counties
def get_surrounding_counties(county):
    return surrounding_counties.get(county, [])

def aggregate_surrounding_data(target_county, data):
    counties = [target_county] + get_surrounding_counties(target_county)
    aggregated_data = data[data['CountyName'].isin(counties)]
    return aggregated_data

def get_counties_in_same_district(user_input):
    for district, counties in agricultural_districts.items():
        if user_input in counties:
            return counties
    return None

def get_district_by_county(county_name):
    for district, counties in agricultural_districts.items():
        if county_name in counties:
            return district
    return "County not found in any district."


# 
def fill_nan(data):
    """
    Replace NaN values in the array with the average of the two neighboring values.
    If the NaN is at the beginning or end, it takes the value of the single neighbor.
    """
    data = np.array(data, dtype=np.float64)  # Ensure the data is a NumPy array of type float
    n = len(data)
    
    for i in range(n):
        if np.isnan(data[i]):
            if i == 0:
                data[i] = data[i+1]  # If NaN is at the beginning
            elif i == n-1:
                data[i] = data[i-1]  # If NaN is at the end
            else:
                data[i] = (data[i-1] + data[i+1]) / 2  # Average of the two neighbors
    
    return data