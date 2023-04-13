# -*- coding: utf-8 -*-
"""
============================================
# Created: April/01/2021
# Author: Vishal Raul, Yen-Chen Liu, Siddhesh Naidu

IFEWs - Model
This code takes input variables and compute CN, MN, FN, GN, and nitrogen surplus in soil

Input variables:
 x (array)
    RCN_c    = Rate of Corn commercial N (kg/ha)
    A_corn   = Area of corn planted (Acres)
    A_soy    = Area of corn planted (Acres)
    AH_corn  = Area of corn harvested (Acres)
    AH_soy   = Area of soybean harvested (Acres)
    y_c      =  Corn yield (bushels/Acres) 
    y_s      =  Soy yield (bushels/Acres) 
    cattle_H = Hogs cattle population
    cattle_B = Beef cattle population
    cattle_M = Milk cattle population
    cattle_O = Other cattle population
 w (array)
    May_P    = Average May plantation progress
    T_July   = July temperature
    PPT_July = July precipitation
    PPT_June = June precipitation
 
Output: N_surplus,'y_c','y_s','CN','MN','FN','GN'

Changelog:
v5:
- Updated static values of RCN_corn, A_corn, A_soy, AH_corn, AH_soy to use dynamic data values available per year per county from 1997-2019
- Updated the physics equations to neglect the effect of RCN_soy as per new information from Julia's Energy Methodology paper. The old lines have been commented out for now.
- Updated Ethanol production subsystem with the physics equations from Julia's Energy Methodology paper
============================================
"""

import numpy as np
import openmdao.api as om
import pandas as pd
from sklearn import linear_model

def corn_soy_weather(x0):
    # load corn yield data -----------
    loc1 = 'Data_Input/crop_yield_data/Corn_y_model_data_v2.csv'
    dfC = pd.read_csv(loc1)  
    
    loc2 = 'Data_Input/crop_yield_data/Soy_y_model_data_v2.csv'
    dfS = pd.read_csv(loc2)
    
    # Create linear regression object for corn -----------
    regrC = linear_model.LinearRegression()
    regrC.fit(dfC[['year','May_P','Jul_T','Jul_ppt','Jul_ppt_sq','June_ppt']].values, dfC['corn_yield'].values)
    
    # Create linear regression object for soybean
    regrS = linear_model.LinearRegression()
    regrS.fit(dfS[['year','JulAug_T','JulAug_ppt','JulAug_ppt_sq','June_ppt']].values, dfS['soy_yield'].values)
    
    #x0 = 'May_P','Jul_T','Jul_ppt','Jul_ppt_sq','June_ppt'    
    x = np.hstack((2020, x0))
    x = [x.tolist()]    
    xs = np.hstack((2020, x0[1:]))
    xs = [xs.tolist()]
    y_c = regrC.predict(x)[0]
    y_s = regrS.predict(xs)[0]
    
    return y_c,y_s


def IFEW(x,w,display):   
    """
    IFEWs function takes input variable x and weather data w 
    and output important parameters of IFEWs model        
    """

    class Weather_crop_yield(om.ExplicitComponent):
        """
        Weather crop yield model: compute corn and soybean yield
        """
        def setup(self):
            # Inputs 
            self.add_input('w', val=np.ones(5)) 
            
            # Outputs 
            self.add_output('y_c', val= 0)            # Corn yield (bushels/Acres) 
            self.add_output('y_s', val= 0)          # Soy yield (bushels/Acres) 
        
        def compute(self, inputs, outputs):            
            yc,ys = corn_soy_weather(w)
            
            outputs['y_c'] = yc
            outputs['y_s'] = ys            
    
    class Agriculture(om.ExplicitComponent):
        """
        Agriculture: Computes FN, GN, CN, P_corn
        """
        def setup(self):
            # Inputs 
            self.add_input('y_c', val= 0)            # Corn yield (bushels/Acres) 
            self.add_input('y_s', val= 0)          # Soy yield (bushels/Acres) 
            self.add_input('RCN_corn', val= 0)       # rate of commercial (N kg/ha)
            ### self.add_input('RCN_soy', val= 0)    # rate of commercial (N kg/ha)
            self.add_input('A_corn', val= 0)         # Corn planted (Acres) 
            self.add_input('A_soy', val= 0)          # Soybean planted (Acres) 
            self.add_input('AH_corn', val= 0)        # Corn harvested (Acres) 
            self.add_input('AH_soy', val= 0)         # Soybean harvested (Acres) 

            # Outputs 
            self.add_output('P_corn', val=1.0)      # Corn production (bushels)
            self.add_output('P_soy', val=1.0)       # Soy production (bushels)
            self.add_output('CN', val=1.0)          # CN
            self.add_output('FN', val=1.0)          # FN
            self.add_output('GN', val=1.0)          # GN
            
                    
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):
            """
            Evaluates P_corn, C_corn
            """                   
            #  1 bushels/acre = 67.25 kg/ha
    
            y_c = inputs['y_c']
            y_s = inputs['y_s']
            
            RCN_corn = inputs['RCN_corn']
            ### RCN_soy = inputs['RCN_soy']

            A_corn = inputs['A_corn'] 
            A_soy = inputs['A_soy'] 
            A = (A_corn + A_soy)
            
            AH_corn = inputs['AH_corn']
            AH_soy = inputs['A_soy'] 
            AH = (AH_corn + AH_soy)
                    
            outputs['P_corn'] = y_c * A_corn    # bushels 
            outputs['P_soy'] = y_s * A_soy      # bushels 
                    
            # (1 bushels/acre = 67.2511 kg/ha)
            outputs['FN'] = (81.1 *  (y_s*67.2511/1000) - 98.5)*A_soy / A                 # N kg/ha
            
            outputs['GN'] = ((y_c*67.2511)*(1.18/100) *AH_corn + (y_s*67.2511)*(6.4/100)* AH_soy)/AH   # N kg/ha
            
            ### outputs['CN'] = (RCN_corn*A_corn + RCN_soy*A_soy )/ A  # N kg/ha
            outputs['CN'] = RCN_corn                                   # N kg/ha
    
    class EtOH_Prod(om.ExplicitComponent):
        """
        ETOH production
        """
        def setup(self):
            
            # Inputs
            self.add_input('y_c', val= 0)            # Corn yield (bushels/Acres) 
            self.add_input('A_corn', val= 0)         # Corn planted (Acres) 
            
            # Outputs 
            self.add_output('P_EtOH', val=1.0)       # Ethanol production (mil/gal)
    
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):
            """
            Evaluates P_EtOH
            """
            # Equation Comes from Julia's Energy Methodology Paper in terms of thousands of barrels. Therefore, a conversion factor is added.
        
            y_c = inputs['y_c']
            A_corn = inputs['A_corn'] 
            
            outputs['P_EtOH'] = 0.02 * y_c * A_corn * 1.827 / 1000  * (1000 * 42) # Gals of EtOH production                
            
            
    class Animal_Ag(om.ExplicitComponent):
        """
        Animal Agriculture: computes MN
        """
        def setup(self):
            
            # Inputs
            self.add_input('Catt', val=np.array([1.0, 1.0, 1.0]))  # Number of Cattle of Beef, Milk, Other
            self.add_input('Hog', val=1.0)                         # Number of Hogs
            self.add_input('A_corn', val= 0)                       # Corn planted (Acres) 
            self.add_input('A_soy', val= 0)                        # Soybean planted (Acres) 
            
            # Outputs
            self.add_output('MN', val=1.0)
            
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):
            # life cycle days
            lf_Beef = 365   
            lf_Milk = 365   
            lf_HS = 365              # Heifer/steer
            lf_Slught_Catt = 170     # Slught Catt
            lf_Hog = 365
            
            # N kg /day per animal
            N_Beef = 0.15
            N_Milk = 0.204   
            N_HS = 0.1455     # Heifer/steer
            N_Slught_Catt = 0.104     # 
            N_HOG = 0.027
            
            A_corn = inputs['A_corn'] 
            A_soy = inputs['A_soy']     
            A = (A_corn + A_soy)
                    
            Catt_Beef = inputs['Catt'][0]
            Catt_Milk = inputs['Catt'][1]
            Catt_Othr = inputs['Catt'][2]
            
            Hog = inputs['Hog']
            
            Total_Catt_N = Catt_Beef*N_Beef*lf_Beef + Catt_Milk*N_Milk*lf_Milk + \
                           0.5*Catt_Othr*N_HS*lf_HS + 0.5*Catt_Othr*N_Slught_Catt*lf_Slught_Catt
            
            Total_Hog_N = Hog * N_HOG * lf_Hog
            
            outputs['MN'] = (Total_Catt_N + Total_Hog_N)/A  # N kg/ha
    
    
    class N_surplus(om.ExplicitComponent):
        """
        Nitrogen surplus in soil
        """
        def setup(self):
            
            # Inputs 
            self.add_input('MN', val=1.0)
            self.add_input('FN', val=1.0)
            self.add_input('GN', val=1.0)
            self.add_input('CN', val=1.0)
            
            # Outputs
            self.add_output('N_surplus', val=1.0)
    
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):     
            outputs['N_surplus'] = (inputs['CN']  + inputs['MN'] + inputs['FN'] - inputs['GN'] )  # N kg/ha
    
            
    class Demand_Corn(om.ExplicitComponent):
        """
        Constraint 1 - used for future calculations
        """
        def setup(self):
            
            # Inputs
            self.add_input('P_corn', val=1.0)
            self.add_input('P_EtOH', val=1.0)
            self.add_input('D_Corn', val=1.0)
            
            # Outputs
            self.add_output('const1', val=1.0)
              
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):
    
            outputs['const1'] =  inputs['D_Corn'] - (inputs['P_corn'] - inputs['P_EtOH'] )
    
            
    class Demand_EtOH(om.ExplicitComponent):
        """
        Constraint 2 - used for future calculations
        """
        def setup(self):
            
            # Inputs
            self.add_input('P_EtOH', val=1.0)
            self.add_input('D_EtOH', val=1.0)
            
            # Outputs
            self.add_output('const2', val=1.0)
            
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):
    
            outputs['const2'] =  inputs['D_EtOH'] - inputs['P_EtOH']
            
            
    class Demand_FP(om.ExplicitComponent):
        """
        Constraint 3 - used for future calculations
        """
        def setup(self):
            
            # Inputs
            self.add_input('D_catt_meat', val=1.0)
            self.add_input('D_Hog', val=1.0)
            self.add_input('Catt', val=np.array([1.0, 1.0, 1.0]))
            self.add_input('Hog', val=1.0)
            
            # Outputs
            self.add_output('const3', val=1.0)
            self.add_output('const4', val=1.0)
            
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')
    
        def compute(self, inputs, outputs):
    
            Catt_Beef = inputs['Catt'][0]
            Catt_Othr = inputs['Catt'][2]
            
            outputs['const3'] =  inputs['D_catt_meat']  - ( Catt_Beef + 0.5 * Catt_Othr)
            outputs['const4'] =  inputs['D_Hog'] - inputs['Hog']
            
            
            
    class SellarMDA(om.Group):
        """
        Group containing the Sellar MDA.
        """
        def setup(self):
            
            # Design variables
            indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
            
            indeps.add_output('D_EtOH', 1e6)            
            indeps.add_output('D_Corn', 20000)       
            indeps.add_output('D_catt_meat',10)
            indeps.add_output('D_Hog',10)

            indeps.add_output('w', np.ones(5)) 
            
            # indeps.add_output('y_c', 1)               # Uncomment to use real corn yield data instead of modelled approximations
            # indeps.add_output('y_s', 1)               # Uncomment to use real soy yield data instead of modelled approximations
            indeps.add_output('RCN_corn', 1)
            ### indeps.add_output('RCN_soy', 1)
            indeps.add_output('A_corn', 1)
            indeps.add_output('A_soy', 1)
            indeps.add_output('AH_corn', 1)
            indeps.add_output('AH_soy', 1)    
            indeps.add_output('Hog', 1)                 
            indeps.add_output('Catt', np.ones(3))
                    
            # Connections    
            self.add_subsystem('Weather_crop_yield', Weather_crop_yield(), promotes_inputs=['w'], promotes_outputs=['y_c', 'y_s'])
            ### self.add_subsystem('Agriculture', Agriculture(), promotes_inputs=['y_c','y_s','RCN_corn','RCN_soy', 'A_corn', 'AH_corn', 'A_soy', 'AH_soy'], promotes_outputs=['CN', 'GN', 'FN','P_corn'])
            self.add_subsystem('Agriculture', Agriculture(), promotes_inputs=['y_c','y_s','RCN_corn', 'A_corn', 'AH_corn', 'A_soy', 'AH_soy'], promotes_outputs=['CN', 'GN', 'FN','P_corn'])
            self.add_subsystem('EtOH_Prod', EtOH_Prod(), promotes_inputs=['y_c', 'A_corn'], promotes_outputs=['P_EtOH'])        
            self.add_subsystem('Animal_Ag', Animal_Ag(), promotes_inputs=['Catt','Hog', 'A_corn','A_soy'], promotes_outputs=['MN'])
            
            # Objective function
            self.add_subsystem('Obj', N_surplus(), promotes_inputs=['CN', 'GN', 'FN', 'MN'], promotes_outputs = ['N_surplus'])
    
    
            # Constraints function
            self.add_subsystem('con_Demand_Corn', Demand_Corn(), promotes_inputs=['D_Corn','P_corn','P_EtOH'],promotes_outputs=['const1'])
            self.add_subsystem('con_Demand_EtOH', Demand_EtOH(), promotes_inputs=['P_EtOH','D_EtOH'], promotes_outputs=['const2'])
            self.add_subsystem('con_Demand_FP', Demand_FP(), promotes_inputs=['D_catt_meat', 'D_Hog', 'Catt','Hog'], promotes_outputs=['const3','const4'])


    prob = om.Problem()
    prob.model = SellarMDA()
        
    prob.model.add_objective('N_surplus')
    
    # Add constraint 
    prob.model.add_constraint('const1',  upper=0 )
    prob.model.add_constraint('const2',  upper=0)
    prob.model.add_constraint('const3',  upper=0)
    prob.model.add_constraint('const4',  upper=0)
        
    prob.setup()
            
    prob.set_val('indeps.D_EtOH', 4350*1e6)      # mil gal
    prob.set_val('indeps.D_Corn', 100*1e6)       # bushels
    prob.set_val('indeps.D_catt_meat', 1e5)      # bushels
    prob.set_val('indeps.D_Hog', 10e5)           # bushels
        
    prob.set_val('indeps.w', w)                    # weather data input
    prob.set_val('indeps.RCN_corn', x[0])          # kg/ha
    ### prob.set_val('indeps.RCN_soy', x[1])       # kg/ha
    prob.set_val('indeps.A_corn', x[1])            # Corn planted (Acres) 
    prob.set_val('indeps.A_soy', x[2])             # Corn harvested (Acres)
    prob.set_val('indeps.AH_corn', x[3])           # Soybean planted (Acres)
    prob.set_val('indeps.AH_soy', x[4])            # Soybean harvested (Acres)
    prob.set_val('indeps.Hog', x[5])               # population of Hog
    prob.set_val('indeps.Catt', np.array(x[6:]) )  # population of cattles
    
    prob.run_model()
    
    if display == True:
        print('\n Agriculture ----------------------------')
        print('y_c (bu/acre) =',prob['y_c'][0])
        print('P_corn (bu) =',prob['P_corn'][0])
        print('y_s (bu/acre) =',prob['y_s'][0])
        
        print('\n EtOH Production ----------------------------')
        print('P_EtOH (gal) =',prob['P_EtOH'][0])
        
        print('\n Animal Ag ----------------------------')
        print('Catt (population) =',prob['Catt'])
        print('Hog (population) =',prob['Hog'][0])
            
        print('\n N_surplus ----------------------------')
        print('MN (kg/ha) =',prob['MN'][0])
        print('CN (kg/ha) =',prob['CN'][0])
        print('FN (kg/ha) =',prob['FN'][0])
        print('GN (kg/ha) =',prob['GN'][0])
        print('N_surplus (kg/ha) =',prob['N_surplus'][0])
        #
        print('\n Constraint (used for future calculations) ----------')
        print('const1  : con_Demand_Corn ',prob['const1'])
        print('const2  : con_Demand_EtOH ',prob['const2'])
        print('const3  : con_Demand_FP_Catt ',prob['const3'])
        print('const4  : con_Demand_FP_Hog ',prob['const4'])
    
    return prob['N_surplus'][0], prob['y_c'][0], prob['y_s'][0], prob['P_EtOH'][0], prob['CN'][0], prob['MN'][0], prob['FN'][0], prob['GN'][0]