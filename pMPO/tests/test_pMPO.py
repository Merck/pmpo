import unittest
import csv
import os
import pickle
import pandas as pd
from io import StringIO
from pMPO import pMPOBuilder

########################################################################################################################
########################################################################################################################

# The DataFrame with the reference data used in the original model building
REFERENCE_DATAFRAME = os.path.join(os.path.join(os.path.dirname(__file__), 'assets'), 'CNS_pMPO.df.pkl')

# The refrence pMPO values for each molecule in Hakan's paper
REFERENCE_CNS_PMPO_VALUES = {'Abacavir': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.157},
                             'Acetohexamide': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.163},
                             'Acetyldigitoxin': {'CNS_pMPO': 0.1, 'CNS_pMPO_withSigmoidal': 0.102},
                             'Acrivastine': {'CNS_pMPO': 0.97, 'CNS_pMPO_withSigmoidal': 0.949},
                             'Acyclovir': {'CNS_pMPO': 0.19, 'CNS_pMPO_withSigmoidal': 0.111},
                             'Adefovir': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.15},
                             'Albuterol': {'CNS_pMPO': 0.39, 'CNS_pMPO_withSigmoidal': 0.266},
                             'Alendronate': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.203},
                             'Alfuzosin': {'CNS_pMPO': 0.41, 'CNS_pMPO_withSigmoidal': 0.134},
                             'Aliskiren': {'CNS_pMPO': 0.18, 'CNS_pMPO_withSigmoidal': 0.086},
                             'Allopurinol': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.226},
                             'Alogliptin': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.483},
                             'Alosetron': {'CNS_pMPO': 0.96, 'CNS_pMPO_withSigmoidal': 0.754},
                             'Altretamine': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.694},
                             'Alvimopan': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.115},
                             'Ambenonium': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.327},
                             'Ambrisentan': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.438},
                             'Amiloride': {'CNS_pMPO': 0.23, 'CNS_pMPO_withSigmoidal': 0.224},
                             'Aminocaproic acid': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.379},
                             'Aminosalicylic acid': {'CNS_pMPO': 0.26, 'CNS_pMPO_withSigmoidal': 0.06},
                             'Amoxicillin': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.067},
                             'Amprenavir': {'CNS_pMPO': 0.1, 'CNS_pMPO_withSigmoidal': 0.059},
                             'Anagrelide': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.858},
                             'Anastrozole': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.376},
                             'Anisindione': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.701},
                             'Anisotropine': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.527},
                             'Apixaban': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.263},
                             'Aspirin': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.607},
                             'Astemizole': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.747},
                             'Atazanavir': {'CNS_pMPO': 0.06, 'CNS_pMPO_withSigmoidal': 0.027},
                             'Atenolol': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.254},
                             'Atorvastatin': {'CNS_pMPO': 0.15, 'CNS_pMPO_withSigmoidal': 0.039},
                             'Atovaquone': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.73},
                             'Avanafil': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.087},
                             'Azathioprine': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.415},
                             'Azilsartan': {'CNS_pMPO': 0.34, 'CNS_pMPO_withSigmoidal': 0.279},
                             'Azithromycin': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.234},
                             'Balsalazide': {'CNS_pMPO': 0.15, 'CNS_pMPO_withSigmoidal': 0.094},
                             'Bedaquiline': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.705},
                             'Benazepril': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.011},
                             'Bendroflumethiazide': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.127},
                             'Bentiromide': {'CNS_pMPO': 0.22, 'CNS_pMPO_withSigmoidal': 0.003},
                             'Betamethasone': {'CNS_pMPO': 0.35, 'CNS_pMPO_withSigmoidal': 0.136},
                             'Betaxolol': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.599},
                             'Bethanechol': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.647},
                             'Bicalutamide': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.044},
                             'Bisoprolol': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.564},
                             'Boceprevir': {'CNS_pMPO': 0.14, 'CNS_pMPO_withSigmoidal': 0.126},
                             'Bosentan': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.01},
                             'Bosutinib': {'CNS_pMPO': 0.6, 'CNS_pMPO_withSigmoidal': 0.432},
                             'Budesonide': {'CNS_pMPO': 0.41, 'CNS_pMPO_withSigmoidal': 0.111},
                             'Bufuralol': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.685},
                             'Busulfan': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.279},
                             'Cabozantinib': {'CNS_pMPO': 0.36, 'CNS_pMPO_withSigmoidal': 0.048},
                             'Canagliflozin': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.027},
                             'Capecitabine': {'CNS_pMPO': 0.23, 'CNS_pMPO_withSigmoidal': 0.087},
                             'Carbenicillin': {'CNS_pMPO': 0.15, 'CNS_pMPO_withSigmoidal': 0.029},
                             'Carbidopa': {'CNS_pMPO': 0.26, 'CNS_pMPO_withSigmoidal': 0.226},
                             'Carglumic acid': {'CNS_pMPO': 0.08, 'CNS_pMPO_withSigmoidal': 0.076},
                             'Carprofen': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.619},
                             'Carteolol': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.424},
                             'Cefaclor': {'CNS_pMPO': 0.27, 'CNS_pMPO_withSigmoidal': 0.059},
                             'Cefdinir': {'CNS_pMPO': 0.1, 'CNS_pMPO_withSigmoidal': 0.007},
                             'Cefditoren': {'CNS_pMPO': 0.06, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Cefpodoxime': {'CNS_pMPO': 0.1, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Cefuroxime': {'CNS_pMPO': 0.1, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Celecoxib': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.404},
                             'Ceritinib': {'CNS_pMPO': 0.27, 'CNS_pMPO_withSigmoidal': 0.195},
                             'Cerivastatin': {'CNS_pMPO': 0.3, 'CNS_pMPO_withSigmoidal': 0.001},
                             'Cetirizine': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.725},
                             'Chenodiol': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.197},
                             'Chlorambucil': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.737},
                             'Chloroquine': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.854},
                             'Chlorotrianisene': {'CNS_pMPO': 0.5, 'CNS_pMPO_withSigmoidal': 0.41},
                             'Chlorphenesin carbamate': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.287},
                             'Chlorpropamide': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.259},
                             'Chlorthalidone': {'CNS_pMPO': 0.26, 'CNS_pMPO_withSigmoidal': 0.137},
                             'Cimetidine': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.143},
                             'Cinoxacin': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.411},
                             'Ciprofloxacin': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.406},
                             'Cisapride': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.244},
                             'Clavulanate': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.101},
                             'Clindamycin': {'CNS_pMPO': 0.35, 'CNS_pMPO_withSigmoidal': 0.114},
                             'Clofazimine': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.708},
                             'Clofibrate': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.654},
                             'Clomiphene': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.384},
                             'Clonidine': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.539},
                             'Cloxacillin': {'CNS_pMPO': 0.27, 'CNS_pMPO_withSigmoidal': 0.01},
                             'Cobicistat': {'CNS_pMPO': 0.18, 'CNS_pMPO_withSigmoidal': 0.109},
                             'Colchicine': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.3},
                             'Crizotinib': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.278},
                             'Cromolyn': {'CNS_pMPO': 0.07, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Cyclacillin': {'CNS_pMPO': 0.32, 'CNS_pMPO_withSigmoidal': 0.181},
                             'Cyclophosphamide': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.741},
                             'Cysteamine': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.318},
                             'Dabrafenib': {'CNS_pMPO': 0.3, 'CNS_pMPO_withSigmoidal': 0.126},
                             'Dantrolene': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.523},
                             'Dapagliflozin': {'CNS_pMPO': 0.21, 'CNS_pMPO_withSigmoidal': 0.053},
                             'Darifenacin': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.805},
                             'Deferasirox': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.042},
                             'Delavirdine': {'CNS_pMPO': 0.22, 'CNS_pMPO_withSigmoidal': 0.103},
                             'Demeclocycline': {'CNS_pMPO': 0.09, 'CNS_pMPO_withSigmoidal': 0.048},
                             'Desogestrel': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.615},
                             'Dexlansoprazole': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.654},
                             'Diazoxide': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.741},
                             'Dichlorphenamide': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.18},
                             'Diclofenac': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.522},
                             'Dicumarol': {'CNS_pMPO': 0.5, 'CNS_pMPO_withSigmoidal': 0.152},
                             'Didanosine': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.136},
                             'Diethylcarbamazine': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.487},
                             'Diflunisal': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.576},
                             'Dimethyl fumarate': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.512},
                             'Diphemanil': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.485},
                             'Dipyridamole': {'CNS_pMPO': 0.16, 'CNS_pMPO_withSigmoidal': 0.11},
                             'Dirithromycin': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.242},
                             'Disopyramide': {'CNS_pMPO': 0.91, 'CNS_pMPO_withSigmoidal': 0.787},
                             'Dofetilide': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.127},
                             'Dolutegravir': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.011},
                             'Domperidone': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.421},
                             'Doxazosin': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.263},
                             'Doxercalciferol': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.328},
                             'Drospirenone': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.63},
                             'Dydrogesterone': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.665},
                             'Dyphylline': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.147},
                             'Edoxaban': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.087},
                             'Eltrombopag': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.123},
                             'Empagliflozin': {'CNS_pMPO': 0.18, 'CNS_pMPO_withSigmoidal': 0.091},
                             'Emtricitabine': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.146},
                             'Enalapril': {'CNS_pMPO': 0.45, 'CNS_pMPO_withSigmoidal': 0.044},
                             'Entecavir': {'CNS_pMPO': 0.22, 'CNS_pMPO_withSigmoidal': 0.152},
                             'Eplerenone': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.21},
                             'Eprosartan': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.247},
                             'Estradiol': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.537},
                             'Estramustine': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.623},
                             'Ethacrynic acid': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.699},
                             'Ethambutol': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.45},
                             'Ethoxzolamide': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.552},
                             'Ethylestrenol': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.615},
                             'Ethynodiol diacetate': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.554},
                             'Etodolac': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.456},
                             'Etoposide': {'CNS_pMPO': 0.12, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Etravirine': {'CNS_pMPO': 0.28, 'CNS_pMPO_withSigmoidal': 0.069},
                             'Ezetimibe': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.41},
                             'Fenoprofen': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.725},
                             'Fesoterodine': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.785},
                             'Fexofenadine': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.239},
                             'Flavoxate': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.643},
                             'Flecainide': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.412},
                             'Fludarabine': {'CNS_pMPO': 0.22, 'CNS_pMPO_withSigmoidal': 0.155},
                             'Fluoxymesterone': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.595},
                             'Fluvastatin': {'CNS_pMPO': 0.38, 'CNS_pMPO_withSigmoidal': 0.022},
                             'Fosfomycin': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.083},
                             'Fosinopril': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.381},
                             'Furazolidone': {'CNS_pMPO': 0.38, 'CNS_pMPO_withSigmoidal': 0.251},
                             'Furosemide': {'CNS_pMPO': 0.27, 'CNS_pMPO_withSigmoidal': 0.147},
                             'Gatifloxacin': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.181},
                             'Gefitinib': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.583},
                             'Gemifloxacin': {'CNS_pMPO': 0.38, 'CNS_pMPO_withSigmoidal': 0.124},
                             'Glimepiride': {'CNS_pMPO': 0.17, 'CNS_pMPO_withSigmoidal': 0.03},
                             'Glipizide': {'CNS_pMPO': 0.16, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Glyburide': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.127},
                             'Glycopyrrolate': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.751},
                             'Guaifenesin': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.407},
                             'Guanadrel': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.147},
                             'Guanethidine': {'CNS_pMPO': 0.55, 'CNS_pMPO_withSigmoidal': 0.329},
                             'Hetacillin': {'CNS_pMPO': 0.45, 'CNS_pMPO_withSigmoidal': 0.024},
                             'Hexocyclium': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.711},
                             'Hydralazine': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.362},
                             'Hydrocortisone': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.2},
                             'Ibandronate': {'CNS_pMPO': 0.25, 'CNS_pMPO_withSigmoidal': 0.249},
                             'Ibrutinib': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.371},
                             'Idelalisib': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.117},
                             'Imatinib': {'CNS_pMPO': 0.55, 'CNS_pMPO_withSigmoidal': 0.258},
                             'Indapamide': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.202},
                             'Indinavir': {'CNS_pMPO': 0.22, 'CNS_pMPO_withSigmoidal': 0.109},
                             'Irbesartan': {'CNS_pMPO': 0.6, 'CNS_pMPO_withSigmoidal': 0.388},
                             'Isoniazid': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.265},
                             'Isopropamide': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.694},
                             'Itraconazole': {'CNS_pMPO': 0.36, 'CNS_pMPO_withSigmoidal': 0.197},
                             'Ivacaftor': {'CNS_pMPO': 0.34, 'CNS_pMPO_withSigmoidal': 0.071},
                             'Ketoconazole': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.436},
                             'Ketorolac': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.713},
                             'Labetalol': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.247},
                             'Lactulose': {'CNS_pMPO': 0.15, 'CNS_pMPO_withSigmoidal': 0.131},
                             'Lapatinib': {'CNS_pMPO': 0.32, 'CNS_pMPO_withSigmoidal': 0.04},
                             'Lenvatinib': {'CNS_pMPO': 0.24, 'CNS_pMPO_withSigmoidal': 0.091},
                             'Levofloxacin': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.526},
                             'Linagliptin': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.342},
                             'Linezolid': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.573},
                             'Lisinopril': {'CNS_pMPO': 0.18, 'CNS_pMPO_withSigmoidal': 0.066},
                             'Lomitapide': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.43},
                             'Loperamide': {'CNS_pMPO': 0.81, 'CNS_pMPO_withSigmoidal': 0.783},
                             'Lopinavir': {'CNS_pMPO': 0.03, 'CNS_pMPO_withSigmoidal': 0.009},
                             'Loracarbef': {'CNS_pMPO': 0.28, 'CNS_pMPO_withSigmoidal': 0.115},
                             'Lubiprostone': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.036},
                             'Macitentan': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.097},
                             'Medroxyprogesterone acetate': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.522},
                             'Mefenamic acid': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.594},
                             'Meloxicam': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.121},
                             'Melphalan': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.505},
                             'Mepenzolate': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.754},
                             'Mercaptopurine': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.494},
                             'Mesalamine': {'CNS_pMPO': 0.3, 'CNS_pMPO_withSigmoidal': 0.06},
                             'Mesna': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.62},
                             'Metaproterenol': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.332},
                             'Metaxalone': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.825},
                             'Methantheline': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.567},
                             'Methazolamide': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.385},
                             'Methenamine': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.438},
                             'Methimazole': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.438},
                             'Methscopolamine': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.731},
                             'Methyltestosterone': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.789},
                             'Metolazone': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.202},
                             'Metronidazole': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.336},
                             'Metyrosine': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.202},
                             'Midodrine': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.253},
                             'Miglitol': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.201},
                             'Miltefosine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.419},
                             'Minocycline': {'CNS_pMPO': 0.09, 'CNS_pMPO_withSigmoidal': 0.047},
                             'Minoxidil': {'CNS_pMPO': 0.25, 'CNS_pMPO_withSigmoidal': 0.099},
                             'Mirabegron': {'CNS_pMPO': 0.35, 'CNS_pMPO_withSigmoidal': 0.114},
                             'Mitotane': {'CNS_pMPO': 0.39, 'CNS_pMPO_withSigmoidal': 0.385},
                             'Montelukast': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.22},
                             'Moxifloxacin': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.097},
                             'Mycophenolic acid': {'CNS_pMPO': 0.5, 'CNS_pMPO_withSigmoidal': 0.165},
                             'Nabumetone': {'CNS_pMPO': 0.6, 'CNS_pMPO_withSigmoidal': 0.598},
                             'Nadolol': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.269},
                             'Nalidixic acid': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.562},
                             'Naproxen': {'CNS_pMPO': 0.81, 'CNS_pMPO_withSigmoidal': 0.712},
                             'Nateglinide': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.411},
                             'Nelfinavir': {'CNS_pMPO': 0.19, 'CNS_pMPO_withSigmoidal': 0.105},
                             'Neostigmine': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.505},
                             'Nevirapine': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.853},
                             'Niacin': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.624},
                             'Niclosamide': {'CNS_pMPO': 0.42, 'CNS_pMPO_withSigmoidal': 0.188},
                             'Nilotinib': {'CNS_pMPO': 0.33, 'CNS_pMPO_withSigmoidal': 0.039},
                             'Nilutamide': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.545},
                             'Nintedanib': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.202},
                             'Nitisinone': {'CNS_pMPO': 0.42, 'CNS_pMPO_withSigmoidal': 0.289},
                             'Nizatidine': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.22},
                             'Norgestimate': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.663},
                             'Novobiocin': {'CNS_pMPO': 0.09, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Olaparib': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.287},
                             'Olsalazine': {'CNS_pMPO': 0.23, 'CNS_pMPO_withSigmoidal': 0.158},
                             'Orlistat': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.289},
                             'Oseltamivir': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.28},
                             'Oxamniquine': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.24},
                             'Oxandrolone': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.847},
                             'Oxaprozin': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.772},
                             'Oxyphenbutazone': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.716},
                             'Oxyphenonium': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.715},
                             'Palbociclib': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.122},
                             'Paliperidone': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.404},
                             'Pantoprazole': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.417},
                             'Pargyline': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.392},
                             'Paricalcitol': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.318},
                             'Pemoline': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.593},
                             'Penicillamine': {'CNS_pMPO': 0.42, 'CNS_pMPO_withSigmoidal': 0.353},
                             'Phenazone': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.427},
                             'Phenazopyridine': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.205},
                             'Phenprocoumon': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.876},
                             'Phensuximide': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.52},
                             'Phenylephrine': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.49},
                             'Pilocarpine': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.563},
                             'Pinacidil': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.385},
                             'Pipobroman': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.58},
                             'Pirenzepine': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.604},
                             'Pitavastatin': {'CNS_pMPO': 0.35, 'CNS_pMPO_withSigmoidal': 0.003},
                             'Ponatinib': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.722},
                             'Pralidoxime': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.588},
                             'Pravastatin': {'CNS_pMPO': 0.11, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Primaquine': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.524},
                             'Probenecid': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.525},
                             'Probucol': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.326},
                             'Proguanil': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.308},
                             'Propantheline': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.489},
                             'Propylthiouracil': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.535},
                             'Protokylol': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.255},
                             'Pyridostigmine': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.487},
                             'Quinestrol': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.593},
                             'Quinethazone': {'CNS_pMPO': 0.34, 'CNS_pMPO_withSigmoidal': 0.157},
                             'Quinidine': {'CNS_pMPO': 0.97, 'CNS_pMPO_withSigmoidal': 0.972},
                             'Rabeprazole': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.548},
                             'Raloxifene': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.334},
                             'Raltegravir': {'CNS_pMPO': 0.08, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Ranolazine': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.328},
                             'Regorafenib': {'CNS_pMPO': 0.19, 'CNS_pMPO_withSigmoidal': 0.028},
                             'Repaglinide': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.187},
                             'Reserpine': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.352},
                             'Ribavirin': {'CNS_pMPO': 0.14, 'CNS_pMPO_withSigmoidal': 0.129},
                             'Rifaximin': {'CNS_pMPO': 0.12, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Riociguat': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.01},
                             'Risedronate': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.154},
                             'Ritodrine': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.397},
                             'Ritonavir': {'CNS_pMPO': 0.03, 'CNS_pMPO_withSigmoidal': 0.025},
                             'Rivaroxaban': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.395},
                             'Roflumilast': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.608},
                             'Rosiglitazone': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.643},
                             'Rosuvastatin': {'CNS_pMPO': 0.06, 'CNS_pMPO_withSigmoidal': 0.0},
                             'Ruxolitinib': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.566},
                             'Sapropterin': {'CNS_pMPO': 0.23, 'CNS_pMPO_withSigmoidal': 0.228},
                             'Saquinavir': {'CNS_pMPO': 0.16, 'CNS_pMPO_withSigmoidal': 0.161},
                             'Sibutramine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.563},
                             'Sildenafil': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.387},
                             'Silodosin': {'CNS_pMPO': 0.35, 'CNS_pMPO_withSigmoidal': 0.154},
                             'Simeprevir': {'CNS_pMPO': 0.25, 'CNS_pMPO_withSigmoidal': 0.113},
                             'Sitagliptin': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.443},
                             'Sodium phenylbutyrate': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.506},
                             'Sofosbuvir': {'CNS_pMPO': 0.16, 'CNS_pMPO_withSigmoidal': 0.126},
                             'Sotalol': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.204},
                             'Sparfloxacin': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.122},
                             'Spirapril': {'CNS_pMPO': 0.39, 'CNS_pMPO_withSigmoidal': 0.01},
                             'Spironolactone': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.547},
                             'Stanozolol': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.514},
                             'Stavudine': {'CNS_pMPO': 0.5, 'CNS_pMPO_withSigmoidal': 0.169},
                             'Succimer': {'CNS_pMPO': 0.31, 'CNS_pMPO_withSigmoidal': 0.177},
                             'Sulfacytine': {'CNS_pMPO': 0.36, 'CNS_pMPO_withSigmoidal': 0.167},
                             'Sulfadoxine': {'CNS_pMPO': 0.35, 'CNS_pMPO_withSigmoidal': 0.167},
                             'Sulfameter': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.163},
                             'Sulfamethizole': {'CNS_pMPO': 0.41, 'CNS_pMPO_withSigmoidal': 0.158},
                             'Sulfamethoxazole': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.147},
                             'Sulfaphenazole': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.194},
                             'Sulfasalazine': {'CNS_pMPO': 0.2, 'CNS_pMPO_withSigmoidal': 0.005},
                             'Sulfinpyrazone': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.463},
                             'Sulfoxone': {'CNS_pMPO': 0.1, 'CNS_pMPO_withSigmoidal': 0.003},
                             'Sumatriptan': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.523},
                             'Sunitinib': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.276},
                             'Tamsulosin': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.125},
                             'Tedizolid': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.439},
                             'Tegaserod': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.38},
                             'Telaprevir': {'CNS_pMPO': 0.07, 'CNS_pMPO_withSigmoidal': 0.069},
                             'Telithromycin': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.454},
                             'Tenofovir': {'CNS_pMPO': 0.21, 'CNS_pMPO_withSigmoidal': 0.156},
                             'Testolactone': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.743},
                             'Thiabendazole': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.79},
                             'Thioguanine': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.147},
                             'Ticagrelor': {'CNS_pMPO': 0.15, 'CNS_pMPO_withSigmoidal': 0.127},
                             'Ticlopidine': {'CNS_pMPO': 0.55, 'CNS_pMPO_withSigmoidal': 0.452},
                             'Tiludronate': {'CNS_pMPO': 0.16, 'CNS_pMPO_withSigmoidal': 0.155},
                             'Tinidazole': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.272},
                             'Tiopronin': {'CNS_pMPO': 0.36, 'CNS_pMPO_withSigmoidal': 0.297},
                             'Tipranavir': {'CNS_pMPO': 0.29, 'CNS_pMPO_withSigmoidal': 0.108},
                             'Tofacitinib': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.424},
                             'Tolazamide': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.22},
                             'Tolrestat': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.693},
                             'Torsemide': {'CNS_pMPO': 0.41, 'CNS_pMPO_withSigmoidal': 0.244},
                             'Tranexamic acid': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.408},
                             'Treprostinil': {'CNS_pMPO': 0.38, 'CNS_pMPO_withSigmoidal': 0.02},
                             'Triamterene': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.25},
                             'Tridihexethyl': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.607},
                             'Trimethobenzamide': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.591},
                             'Trimethoprim': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.182},
                             'Trioxsalen': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.641},
                             'Troleandomycin': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.367},
                             'Trospium': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.606},
                             'Trovafloxacin': {'CNS_pMPO': 0.45, 'CNS_pMPO_withSigmoidal': 0.127},
                             'Uracil mustard': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.51},
                             'Valsartan': {'CNS_pMPO': 0.32, 'CNS_pMPO_withSigmoidal': 0.01},
                             'Vandetanib': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.74},
                             'Vemurafenib': {'CNS_pMPO': 0.38, 'CNS_pMPO_withSigmoidal': 0.11},
                             'Vismodegib': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.454},
                             'Vorapaxar': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.375},
                             'Zafirlukast': {'CNS_pMPO': 0.22, 'CNS_pMPO_withSigmoidal': 0.066},
                             'Zidovudine': {'CNS_pMPO': 0.36, 'CNS_pMPO_withSigmoidal': 0.156},
                             'Zileuton': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.452},
                             'Abiraterone': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.674},
                             'Acebutolol': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.243},
                             'Acetaminophen': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.388},
                             'Acetazolamide': {'CNS_pMPO': 0.33, 'CNS_pMPO_withSigmoidal': 0.118},
                             'Acetophenazine': {'CNS_pMPO': 0.9, 'CNS_pMPO_withSigmoidal': 0.821},
                             'Acitretin': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.851},
                             'Afatinib': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.234},
                             'Albendazole': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.496},
                             'Almotriptan': {'CNS_pMPO': 0.92, 'CNS_pMPO_withSigmoidal': 0.824},
                             'Alprazolam': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.742},
                             'Alprenolol': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.562},
                             'Amantadine': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.591},
                             'Aminoglutethimide': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.4},
                             'Amitriptyline': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.577},
                             'Amlodipine': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.234},
                             'Amoxapine': {'CNS_pMPO': 0.95, 'CNS_pMPO_withSigmoidal': 0.927},
                             'Amphetamine': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.606},
                             'Anileridine': {'CNS_pMPO': 0.97, 'CNS_pMPO_withSigmoidal': 0.936},
                             'Aniracetam': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.579},
                             'Apomorphine': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.718},
                             'Aprepitant': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.167},
                             'Aripiprazole': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.732},
                             'Armodafinil': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.806},
                             'Atomoxetine': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.674},
                             'Atropine': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.835},
                             'Axitinib': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.264},
                             'Azatadine': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.654},
                             'Baclofen': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.459},
                             'Benzphetamine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.563},
                             'Benztropine': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.621},
                             'Bepridil': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.527},
                             'Bethanidine': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.382},
                             'Bexarotene': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.701},
                             'Biperiden': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.849},
                             'Bromazepam': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.876},
                             'Bromocriptine': {'CNS_pMPO': 0.17, 'CNS_pMPO_withSigmoidal': 0.03},
                             'Bromodiphenhydramine': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.632},
                             'Brompheniramine': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.68},
                             'Buclizine': {'CNS_pMPO': 0.42, 'CNS_pMPO_withSigmoidal': 0.251},
                             'Budipine': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.589},
                             'Bumetanide': {'CNS_pMPO': 0.27, 'CNS_pMPO_withSigmoidal': 0.07},
                             'Buprenorphine': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.537},
                             'Bupropion': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.74},
                             'Buspirone': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.456},
                             'Butabarbital': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.327},
                             'Cabergoline': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.267},
                             'Caffeine': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.536},
                             'Carbamazepine': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.833},
                             'Carbinoxamine': {'CNS_pMPO': 0.76, 'CNS_pMPO_withSigmoidal': 0.76},
                             'Carisoprodol': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.279},
                             'Carvedilol': {'CNS_pMPO': 0.55, 'CNS_pMPO_withSigmoidal': 0.309},
                             'Cevimeline': {'CNS_pMPO': 0.5, 'CNS_pMPO_withSigmoidal': 0.454},
                             'Chlophedianol': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.853},
                             'Chloramphenicol': {'CNS_pMPO': 0.32, 'CNS_pMPO_withSigmoidal': 0.188},
                             'Chlordiazepoxide': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.879},
                             'Chlormezanone': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.629},
                             'Chlorphentermine': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.65},
                             'Chlorpromazine': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.587},
                             'Chlorprothixene': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.535},
                             'Chlorzoxazone': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.751},
                             'Cilostazol': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.445},
                             'Cinacalcet': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.66},
                             'Citalopram': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.681},
                             'Clemastine': {'CNS_pMPO': 0.6, 'CNS_pMPO_withSigmoidal': 0.58},
                             'Clidinium': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.71},
                             'Clobazam': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.742},
                             'Clomipramine': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.582},
                             'Clonazepam': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.548},
                             'Clozapine': {'CNS_pMPO': 0.91, 'CNS_pMPO_withSigmoidal': 0.807},
                             'Cycloserine': {'CNS_pMPO': 0.55, 'CNS_pMPO_withSigmoidal': 0.296},
                             'Danazol': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.776},
                             'Dapsone': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.168},
                             'Dasatinib': {'CNS_pMPO': 0.31, 'CNS_pMPO_withSigmoidal': 0.127},
                             'Desloratadine': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.776},
                             'Desvenlafaxine': {'CNS_pMPO': 0.81, 'CNS_pMPO_withSigmoidal': 0.58},
                             'Dexmethylphenidate': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.782},
                             'Dextromethorphan': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.65},
                             'Dicyclomine': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.712},
                             'Diethylpropion': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.662},
                             'Difenoxin': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.755},
                             'Dihydrocodeine': {'CNS_pMPO': 0.95, 'CNS_pMPO_withSigmoidal': 0.857},
                             'Diltiazem': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.685},
                             'Diphenylpyraline': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.627},
                             'Disulfiram': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.469},
                             'Dolasetron': {'CNS_pMPO': 0.92, 'CNS_pMPO_withSigmoidal': 0.816},
                             'Donepezil': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.681},
                             'Dronabinol': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.675},
                             'Dronedarone': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.386},
                             'Duloxetine': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.8},
                             'Dutasteride': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.346},
                             'Efavirenz': {'CNS_pMPO': 0.76, 'CNS_pMPO_withSigmoidal': 0.763},
                             'Eletriptan': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.686},
                             'Eliglustat': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.402},
                             'Entacapone': {'CNS_pMPO': 0.4, 'CNS_pMPO_withSigmoidal': 0.168},
                             'Enzalutamide': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.468},
                             'Erlotinib': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.499},
                             'Eszopiclone': {'CNS_pMPO': 0.51, 'CNS_pMPO_withSigmoidal': 0.155},
                             'Ethchlorvynol': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.615},
                             'Ethinamate': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.776},
                             'Ethionamide': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.726},
                             'Ethopropazine': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.566},
                             'Ethosuximide': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.631},
                             'Ethotoin': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.691},
                             'Ezogabine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.242},
                             'Famotidine': {'CNS_pMPO': 0.31, 'CNS_pMPO_withSigmoidal': 0.255},
                             'Febuxostat': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.564},
                             'Felbamate': {'CNS_pMPO': 0.44, 'CNS_pMPO_withSigmoidal': 0.229},
                             'Felodipine': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.586},
                             'Fenofibrate': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.595},
                             'Fentanyl': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.715},
                             'Finasteride': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.468},
                             'Fingolimod': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.625},
                             'Flibanserin': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.765},
                             'Fluconazole': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.448},
                             'Flucytosine': {'CNS_pMPO': 0.46, 'CNS_pMPO_withSigmoidal': 0.268},
                             'Fluoxetine': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.82},
                             'Flurazepam': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.66},
                             'Flurbiprofen': {'CNS_pMPO': 0.81, 'CNS_pMPO_withSigmoidal': 0.719},
                             'Fluvoxamine': {'CNS_pMPO': 0.96, 'CNS_pMPO_withSigmoidal': 0.919},
                             'Frovatriptan': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.37},
                             'Gabapentin': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.42},
                             'Galantamine': {'CNS_pMPO': 0.97, 'CNS_pMPO_withSigmoidal': 0.926},
                             'Gemfibrozil': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.857},
                             'Granisetron': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.819},
                             'Guanabenz': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.44},
                             'Guanfacine': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.294},
                             'Halazepam': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.582},
                             'Halofantrine': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.576},
                             'Haloperidol': {'CNS_pMPO': 0.93, 'CNS_pMPO_withSigmoidal': 0.853},
                             'Hydroxyzine': {'CNS_pMPO': 0.92, 'CNS_pMPO_withSigmoidal': 0.837},
                             'Ibuprofen': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.66},
                             'Iloperidone': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.631},
                             'Indomethacin': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.57},
                             'Isocarboxazid': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.401},
                             'Isotretinoin': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.784},
                             'Isradipine': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.394},
                             'Ketoprofen': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.731},
                             'Lacosamide': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.387},
                             'Lamotrigine': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.151},
                             'Lenalidomide': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.152},
                             'Letrozole': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.478},
                             'Levamisole': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.466},
                             'Levetiracetam': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.6},
                             'Levodopa': {'CNS_pMPO': 0.26, 'CNS_pMPO_withSigmoidal': 0.192},
                             'Levomepromazine': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.649},
                             'Levomethadyl': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.685},
                             'Levomilnacipran': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.795},
                             'Levopropoxyphene': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.714},
                             'Lidocaine': {'CNS_pMPO': 0.9, 'CNS_pMPO_withSigmoidal': 0.897},
                             'Lofexidine': {'CNS_pMPO': 0.9, 'CNS_pMPO_withSigmoidal': 0.897},
                             'Lomustine': {'CNS_pMPO': 0.81, 'CNS_pMPO_withSigmoidal': 0.789},
                             'Loratadine': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.497},
                             'Lorazepam': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.579},
                             'Lorcainide': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.6},
                             'Lorcaserin': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.556},
                             'Losartan': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.137},
                             'Lovastatin': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.469},
                             'Lurasidone': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.659},
                             'Maprotiline': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.734},
                             'Maraviroc': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.671},
                             'Mazindol': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.721},
                             'Mecamylamine': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.489},
                             'Mefloquine': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.548},
                             'Memantine': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.613},
                             'Meperidine': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.769},
                             'Mesoridazine': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.585},
                             'Metergoline': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.757},
                             'Metformin': {'CNS_pMPO': 0.17, 'CNS_pMPO_withSigmoidal': 0.041},
                             'Methamphetamine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.503},
                             'Metharbital': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.665},
                             'Methdilazine': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.595},
                             'Methixene': {'CNS_pMPO': 0.55, 'CNS_pMPO_withSigmoidal': 0.555},
                             'Methocarbamol': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.139},
                             'Methylergonovine': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.587},
                             'Methyprylon': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.664},
                             'Metoclopramide': {'CNS_pMPO': 0.76, 'CNS_pMPO_withSigmoidal': 0.487},
                             'Metoprolol': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.588},
                             'Metyrapone': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.67},
                             'Mexiletine': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.753},
                             'Mifepristone': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.614},
                             'Minaprine': {'CNS_pMPO': 0.99, 'CNS_pMPO_withSigmoidal': 0.885},
                             'Mirtazapine': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.71},
                             'Moclobemide': {'CNS_pMPO': 0.94, 'CNS_pMPO_withSigmoidal': 0.733},
                             'Molindone': {'CNS_pMPO': 0.97, 'CNS_pMPO_withSigmoidal': 0.872},
                             'Nabilone': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.644},
                             'Nalmefene': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.718},
                             'Naloxegol': {'CNS_pMPO': 0.28, 'CNS_pMPO_withSigmoidal': 0.01},
                             'Naratriptan': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.503},
                             'Nebivolol': {'CNS_pMPO': 0.6, 'CNS_pMPO_withSigmoidal': 0.411},
                             'Nefazodone': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.686},
                             'Nemonapride': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.597},
                             'Nicardipine': {'CNS_pMPO': 0.47, 'CNS_pMPO_withSigmoidal': 0.326},
                             'Nicergoline': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.635},
                             'Nicotine': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.467},
                             'Nifedipine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.492},
                             'Nortriptyline': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.718},
                             'Noscapine': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.344},
                             'Ondansetron': {'CNS_pMPO': 0.85, 'CNS_pMPO_withSigmoidal': 0.833},
                             'Ospemifene': {'CNS_pMPO': 0.64, 'CNS_pMPO_withSigmoidal': 0.549},
                             'Oxprenolol': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.586},
                             'Oxybate': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.347},
                             'Oxybutynin': {'CNS_pMPO': 0.9, 'CNS_pMPO_withSigmoidal': 0.864},
                             'Oxyphencyclimine': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.847},
                             'Palonosetron': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.601},
                             'Panobinostat': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.412},
                             'Paramethadione': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.526},
                             'Paroxetine': {'CNS_pMPO': 0.94, 'CNS_pMPO_withSigmoidal': 0.939},
                             'Pazopanib': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.137},
                             'Penbutolol': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.699},
                             'Pentazocine': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.843},
                             'Pentoxifylline': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.387},
                             'Perampanel': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.658},
                             'Pergolide': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.791},
                             'Perindopril': {'CNS_pMPO': 0.5, 'CNS_pMPO_withSigmoidal': 0.067},
                             'Phenacemide': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.233},
                             'Phenelzine': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.463},
                             'Phenmetrazine': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.636},
                             'Phenobarbital': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.348},
                             'Phenoxybenzamine': {'CNS_pMPO': 0.57, 'CNS_pMPO_withSigmoidal': 0.475},
                             'Phenylpropanolamine': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.499},
                             'Phenytoin': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.582},
                             'Pimozide': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.678},
                             'Pioglitazone': {'CNS_pMPO': 0.86, 'CNS_pMPO_withSigmoidal': 0.705},
                             'Pirfenidone': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.528},
                             'Pramipexole': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.537},
                             'Prasugrel': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.614},
                             'Praziquantel': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.734},
                             'Procainamide': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.531},
                             'Procarbazine': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.556},
                             'Prochlorperazine': {'CNS_pMPO': 0.56, 'CNS_pMPO_withSigmoidal': 0.478},
                             'Propofol': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.577},
                             'Propranolol': {'CNS_pMPO': 0.81, 'CNS_pMPO_withSigmoidal': 0.592},
                             'Pseudoephedrine': {'CNS_pMPO': 0.61, 'CNS_pMPO_withSigmoidal': 0.433},
                             'Pyrazinamide': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.497},
                             'Pyrilamine': {'CNS_pMPO': 0.76, 'CNS_pMPO_withSigmoidal': 0.656},
                             'Pyrimethamine': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.322},
                             'Pyrvinium': {'CNS_pMPO': 0.38, 'CNS_pMPO_withSigmoidal': 0.294},
                             'Quetiapine': {'CNS_pMPO': 0.93, 'CNS_pMPO_withSigmoidal': 0.742},
                             'Ramelteon': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.828},
                             'Rasagiline': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.672},
                             'Reboxetine': {'CNS_pMPO': 0.97, 'CNS_pMPO_withSigmoidal': 0.974},
                             'Remifentanil': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.385},
                             'Riluzole': {'CNS_pMPO': 0.84, 'CNS_pMPO_withSigmoidal': 0.828},
                             'Rimantadine': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.614},
                             'Rimonabant': {'CNS_pMPO': 0.66, 'CNS_pMPO_withSigmoidal': 0.611},
                             'Risperidone': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.677},
                             'Rivastigmine': {'CNS_pMPO': 0.77, 'CNS_pMPO_withSigmoidal': 0.679},
                             'Rizatriptan': {'CNS_pMPO': 0.88, 'CNS_pMPO_withSigmoidal': 0.841},
                             'Rofecoxib': {'CNS_pMPO': 0.74, 'CNS_pMPO_withSigmoidal': 0.719},
                             'Ropinirole': {'CNS_pMPO': 0.89, 'CNS_pMPO_withSigmoidal': 0.869},
                             'Ropivacaine': {'CNS_pMPO': 0.91, 'CNS_pMPO_withSigmoidal': 0.916},
                             'Rufinamide': {'CNS_pMPO': 0.71, 'CNS_pMPO_withSigmoidal': 0.509},
                             'Saxagliptin': {'CNS_pMPO': 0.58, 'CNS_pMPO_withSigmoidal': 0.284},
                             'Selegiline': {'CNS_pMPO': 0.53, 'CNS_pMPO_withSigmoidal': 0.514},
                             'Sertindole': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.747},
                             'Sertraline': {'CNS_pMPO': 0.76, 'CNS_pMPO_withSigmoidal': 0.761},
                             'Sulindac': {'CNS_pMPO': 0.83, 'CNS_pMPO_withSigmoidal': 0.691},
                             'Suvorexant': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.259},
                             'Tacrine': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.854},
                             'Tadalafil': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.501},
                             'Talipexole': {'CNS_pMPO': 0.9, 'CNS_pMPO_withSigmoidal': 0.726},
                             'Tamoxifen': {'CNS_pMPO': 0.52, 'CNS_pMPO_withSigmoidal': 0.441},
                             'Tapentadol': {'CNS_pMPO': 0.8, 'CNS_pMPO_withSigmoidal': 0.776},
                             'Tasimelteon': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.826},
                             'Telmisartan': {'CNS_pMPO': 0.63, 'CNS_pMPO_withSigmoidal': 0.441},
                             'Temozolomide': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.342},
                             'Terbinafine': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.387},
                             'Terguride': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.681},
                             'Teriflunomide': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.292},
                             'Tetrabenazine': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.694},
                             'Thiopental': {'CNS_pMPO': 0.68, 'CNS_pMPO_withSigmoidal': 0.489},
                             'Thiothixene': {'CNS_pMPO': 0.72, 'CNS_pMPO_withSigmoidal': 0.674},
                             'Tiagabine': {'CNS_pMPO': 0.89, 'CNS_pMPO_withSigmoidal': 0.809},
                             'Tianeptine': {'CNS_pMPO': 0.54, 'CNS_pMPO_withSigmoidal': 0.105},
                             'Timolol': {'CNS_pMPO': 0.62, 'CNS_pMPO_withSigmoidal': 0.299},
                             'Tizanidine': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.55},
                             'Tocainide': {'CNS_pMPO': 0.76, 'CNS_pMPO_withSigmoidal': 0.532},
                             'Tolcapone': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.287},
                             'Tolmetin': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.714},
                             'Topotecan': {'CNS_pMPO': 0.49, 'CNS_pMPO_withSigmoidal': 0.122},
                             'Tramadol': {'CNS_pMPO': 0.87, 'CNS_pMPO_withSigmoidal': 0.774},
                             'Tranylcypromine': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.634},
                             'Trazodone': {'CNS_pMPO': 0.82, 'CNS_pMPO_withSigmoidal': 0.75},
                             'Trimipramine': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.592},
                             'Triprolidine': {'CNS_pMPO': 0.67, 'CNS_pMPO_withSigmoidal': 0.674},
                             'Troglitazone': {'CNS_pMPO': 0.43, 'CNS_pMPO_withSigmoidal': 0.091},
                             'Tropisetron': {'CNS_pMPO': 0.94, 'CNS_pMPO_withSigmoidal': 0.87},
                             'Valdecoxib': {'CNS_pMPO': 0.7, 'CNS_pMPO_withSigmoidal': 0.555},
                             'Valproic acid': {'CNS_pMPO': 0.69, 'CNS_pMPO_withSigmoidal': 0.602},
                             'Varenicline': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.755},
                             'Vilazodone': {'CNS_pMPO': 0.37, 'CNS_pMPO_withSigmoidal': 0.123},
                             'Vinpocetine': {'CNS_pMPO': 0.73, 'CNS_pMPO_withSigmoidal': 0.702},
                             'Voriconazole': {'CNS_pMPO': 0.75, 'CNS_pMPO_withSigmoidal': 0.471},
                             'Vorinostat': {'CNS_pMPO': 0.48, 'CNS_pMPO_withSigmoidal': 0.206},
                             'Vortioxetine': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.786},
                             'Zaleplon': {'CNS_pMPO': 0.65, 'CNS_pMPO_withSigmoidal': 0.42},
                             'Ziprasidone': {'CNS_pMPO': 0.89, 'CNS_pMPO_withSigmoidal': 0.813},
                             'Zolmitriptan': {'CNS_pMPO': 0.78, 'CNS_pMPO_withSigmoidal': 0.58},
                             'Zolpidem': {'CNS_pMPO': 0.79, 'CNS_pMPO_withSigmoidal': 0.706},
                             'Zonisamide': {'CNS_pMPO': 0.59, 'CNS_pMPO_withSigmoidal': 0.369}}

# The reference intermediate CSV values from the model building in Hakan's paper
REFERENCE_INTERMEDIATE_VALUES_CSV = """name,p_value,good_mean,good_std,bad_mean,bad_std,good_nsamples,bad_nsamples,cutoff,b,c,z,w
TPSA,1.53302825374e-37,50.7017727635,28.3039124335,86.6483879508,39.3310947469,299,366,65.7447200619,0.151695487107,0.793679783519,0.531479431818,0.333254
HBD,2.6491093047e-25,1.08695652174,0.891691734071,2.03825136612,1.35245625655,299,366,1.46494485092,0.0940054673368,9.51515104848e-05,0.423900227773,0.265798
MW,2.32807969696e-10,304.703053545,94.0468619927,362.423958393,135.961231175,299,366,328.304266431,0.0319893623019,0.829287962918,0.250951625455,0.157354
cLogD_ACD_v15,2.66954664227e-07,1.80861953019,1.93092146089,0.838442630309,2.84658487103,297,366,1.41650379728,0.0208331236351,131996.985929,0.203071818744,0.127332
mbpKa,0.000219929547677,8.07348212768,2.20894961173,7.17077320052,2.65960541699,224,194,7.66390710544,0.0173381729161,1459310.7835,0.185416190602,0.116262
"""

########################################################################################################################
########################################################################################################################


# Transformation of the intermediate CSV values above into something usable
def _read_csv_to_dict(csv_text: str) -> dict:
    data = {}
    csv_io = StringIO(csv_text)
    reader = csv.DictReader(csv_io, delimiter=',')
    for row in reader:
        name = row.pop('name')
        data[name] = row
    return data

REFERENCE_INTERMEDIATE_VALUES = _read_csv_to_dict(REFERENCE_INTERMEDIATE_VALUES_CSV)

########################################################################################################################
########################################################################################################################


class test_suite001_building(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_pickle(REFERENCE_DATAFRAME)

    def test001_nonsigmoidal_builder(self):
        """
        Make sure we can construct a builder
        """
        builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO', sigmoidal_correction=False)
        self.assertIsNotNone(builder)

    def test002_nonsigmoidal_get_model(self):
        """
        Make sure we can retrieve a model
        """
        builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO', sigmoidal_correction=False)
        model = builder.get_pMPO()
        self.assertIsNotNone(model)

    def test003_sigmoidal_builder(self):
        """
        Make sure we can construct a builder
        """
        builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO', sigmoidal_correction=True)
        self.assertIsNotNone(builder)

    def test004_sigmoidal_get_model(self):
        """
        Make sure we can retrieve a model
        """
        builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO', sigmoidal_correction=True)
        model = builder.get_pMPO()
        self.assertIsNotNone(model)

class test_suite002_behavior(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_pickle(REFERENCE_DATAFRAME)
        self.builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO', sigmoidal_correction=False)
        self.model = self.builder.get_pMPO()
        # Build with the sigmoidal correction
        self.sig_builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO SIG', sigmoidal_correction=True)
        self.sig_model = self.sig_builder.get_pMPO()

    def test005_sigmoidal_correction_flag(self):
        self.assertFalse(self.model.sigmoidal_correction)
        self.assertTrue(self.sig_model.sigmoidal_correction)

    def test006_important_descriptors(self):
        """
        Check all the important descriptors from the publication are still considered important
        """
        IMPORTANT_DESCRIPTORS = ('TPSA', 'HBD', 'MW', 'CLOGD_ACD_V15', 'MBPKA')
        self.assertEqual(len(self.model.gaussians), len(IMPORTANT_DESCRIPTORS))
        self.assertEqual(len(self.model.sigmoidals), len(IMPORTANT_DESCRIPTORS))
        self.assertEqual(len(self.sig_model.gaussians), len(IMPORTANT_DESCRIPTORS))
        self.assertEqual(len(self.sig_model.sigmoidals), len(IMPORTANT_DESCRIPTORS))
        for desc in IMPORTANT_DESCRIPTORS:
            self.assertIn(desc, self.model.gaussians)
            self.assertIn(desc, self.model.sigmoidals)
            self.assertIn(desc, self.sig_model.gaussians)
            self.assertIn(desc, self.sig_model.sigmoidals)

    def test007_intermediate_statistics(self):
        """
        Check all the intermediate statistics are OK
        """
        stats = self.builder.get_pMPO_statistics()
        # List of all stats to check for important statistic
        all_stats = ['p_value', 'good_mean', 'good_std', 'bad_mean', 'bad_std', 'good_nsamples', 'bad_nsamples',
                     'cutoff', 'b', 'c', 'z', 'w']
        for row in stats.iterrows():
            row_data = row[1].to_dict()
            name = row_data.pop('name')
            if name in REFERENCE_INTERMEDIATE_VALUES:
                this_rows_stats = set(all_stats[:])
                for attribute_name, attribute_value in row_data.items():
                    if attribute_name in REFERENCE_INTERMEDIATE_VALUES[name]:
                        val = float(REFERENCE_INTERMEDIATE_VALUES[name][attribute_name])
                        self.assertTrue(val - 0.01 <= attribute_value <= val + 0.01,
                                        "{} does not match reference within 0.01 tolerance {} != {}".format(
                                            attribute_name, attribute_value, val))
                        this_rows_stats.remove(attribute_name)
                self.assertTrue(len(this_rows_stats) == 0, "{} did not check {}".format(name, this_rows_stats))

    def test008_non_sigmoidal_prediction(self):
        for row in self.df.iterrows():
            row_data = row[1].to_dict()
            name = row_data.pop('Drug')
            # Score this molecule
            score = self.model(**row_data)
            # The reference values underwent some heavy rounding so we give ourselves a little more wiggle room
            self.assertTrue(REFERENCE_CNS_PMPO_VALUES[name]['CNS_pMPO'] - 0.025 <= score <= REFERENCE_CNS_PMPO_VALUES[name]['CNS_pMPO'] + 0.025,  # noqa
                            "{} does not score within 0.025 tolerance of reference value {} != {}".format(
                                name, score, REFERENCE_CNS_PMPO_VALUES[name]['CNS_pMPO']))

    def test009_sigmoidal_prediction(self):
        for row in self.df.iterrows():
            row_data = row[1].to_dict()
            name = row_data.pop('Drug')
            # Score this molecule
            score = self.sig_model(**row_data)
            # The reference values underwent some heavy rounding so we give ourselves a little more wiggle room
            self.assertTrue(REFERENCE_CNS_PMPO_VALUES[name]['CNS_pMPO_withSigmoidal'] - 0.1 <= score <= REFERENCE_CNS_PMPO_VALUES[name]['CNS_pMPO_withSigmoidal'] + 0.1,  # noqa
                            "{} does not score within 0.1 tolerance of reference value {} != {}".format(
                                name, score, REFERENCE_CNS_PMPO_VALUES[name]['CNS_pMPO_withSigmoidal']))

    def test010_pickle(self):
        serialized_model = pickle.dumps(self.model)
        self.assertIsNotNone(serialized_model)
        unserialized_model = pickle.loads(serialized_model)
        self.assertEqual(str(self.model), str(unserialized_model))

    def test011_null_dataframe(self):
        """
        Try building a pMPO with an empty DataFrame
        """
        df = pd.DataFrame()
        with self.assertRaises(AssertionError):
            pMPOBuilder(df, good_column='CNS', model_name='CNS pMPO')

    def test012_invalid_column(self):
        """
        Try building a pMPO with an invalid column
        """
        with self.assertRaises(AssertionError):
            pMPOBuilder(self.df, good_column='I DO NOT EXIST', model_name='CNS pMPO')

    def test013_case_sensitivity(self):
        sensitive_builder = pMPOBuilder(self.df, good_column='CNS', model_name='CNS pMPO', case_insensitive=False)
        sensitive_model = sensitive_builder.get_pMPO()
        props = ('TPSA', 'HBD', 'MW', 'cLogD_ACD_v15', 'mbpKa')
        # Check that the props are case sensitive in the sensitive_model
        for p in props:
            self.assertIn(p, sensitive_model.gaussians)
            self.assertIn(p, sensitive_model.sigmoidals)
