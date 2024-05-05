import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
import logging

########################################
# Begin database stuff

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open(os.path.join('data', 'columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions
"""
def check_request(request):
    if "data" not in request:
        error = "data error"
        return False, error
    return True, ""
"""

def check_valid_column(observation):
    valid_columns = {"observation_id", "Type", "race", "Date", "Part of a standard enforcement protocol", 
                     "Galactic X", "Galactic Y",
                     "Reproduction", "Age range", "Self-defined species category",
                     "Officer-defined species category", "Governing law", "Object of inspection",
                     "Inspection involving more than just outerwear", "Enforcement station"}
    for key in observation.keys():
        if key not in valid_columns:
            error = f"{key} not in columns"
            return False, error
    for col in valid_columns:
        if col not in observation.keys():
            error = f"{col} not in columns"
            return False, error
    return True, ""


def check_categorical_values(observation):
    valid_category = {'Type': {"values":['Entity inspection', 'Entity and Spaceship search', 'Spaceship search']},
                      'Part of a standard enforcement protocol': {"values":[False, True], "default":None}, 
                      'Reproduction': {"values":['Asexual', 'Sexual']},
                      'Age range': {"values":['Senior', 'Adult', 'Young Adult', 'Young', 'Child']},
                      'Self-defined species category': {"values": ['Zoltrax - Diverse Clans', 'Terran - Northern Cluster','Xenar - Unspecified',
                                                        'Terran - Outer Colonies','Terran - Gaelic Cluster','Xenar - Diverse',
                                                        'Silicar - Various Sects', 'Zoltrax - Islander','Zoltrax - Continental',
                                                        'Silicar - Indusian', 'Hybrid - Terran/Zoltrax', 'Hybrid - Terran/Silicar',
                                                        'Silicar - Banglorian', 'Silicar - Pakterran', 'Hybrid - Mixed Sects',
                                                        'Hybrid - Terran/Zoltrax Continental', 'Terran - Nomadic Tribes', 'Silicar - Sino Sector'], "default":None},
                      'Officer-defined species category': {"values":['Zoltrax', 'Silicar', 'Terran', 'Xenar', 'Hybrid'], "default":None},
                      'Governing law': {"values":['Galactic Enforcement and Evidence Code 3984 (Clause 1)',
                                        'Interstellar Substance Control Ordinance 4719 (Directive 23)',
                                        'Stellar Arms Regulation 3968 (Provision 47)',
                                        'Universal Justice and Order Statute 3994 (Article 60)',
                                        'Cosmic Justice Decree 3988 (Subsection 139B)',
                                        'Starflight Security Act 3982 (Protocol 27(1))'], "default":None},
                      'Object of inspection': {"values":['Vandalism Tools', 'Regulated Star Substances','Thievery Equipment',
                                               'Prohibited Combat Implements','Confiscated Galactic Artifacts','Prohibited Blasters',
                                               'Proof of Interstellar Offenses', 'Hazardous Intimidation Devices', 'Banned Pyrotechnics',
                                               'Controlled Psychic Compounds'], "default":None},
                      'Inspection involving more than just outerwear': {"values":[False, True, None], "default":None},
                      'Enforcement station': {"values":['Galactic Hub X101-IO', 'Helios Command C210-ZR', 'Krypton Dock D175-WU',
                                              'Sirius Patrol Unit J153-AX','Mercury Base R525-VN','Southern Yorik Station F160-OK',
                                              'Eclipse Sector G262-TX','Westeron Division B234-YZ', 'Thallium Outpost V207-LQ',
                                              'Avalon Sector M148-CN','Western Meridian Line P111-GT','Saxon Shield U87-PE',
                                              'Hercules Node S188-MP','Binary Transition Point L115-QE','Crimson Core Q31-TE',
                                              'Lunar City Command F35-GC', 'Chevron Station R71-QB','Humeros Node M28-NR',
                                              'Nebular Frontier A92-TR','Dione Command E106-HD','Gallium Ground V47-LM','Lancer Base O114-XV',
                                              'Derelict Dominion C50-MV','Warlock Ward E49-UN','Beta Site E56-SD',
                                              'Nova Guard K116-PL','Sulfur Station T44-PD','Durham Dock W39-HB','Leviathan Lookout Y54-KH',
                                              'Dyson Sphere F76-JK','Starfield Watch N108-BC','Linear Lightway B59-XN','Nebula Yard Y27-FA',
                                              'Graviton Manor Z54-VL','Willow Wavefront S16-TO','Northern Apex O42-IR','Narwhal Base T65-ZV',
                                              'Cliffside Colony B21-JI','Dorset Delta I55-FO','Camelot Axis Z14-XP','Nexus Orbit U55-WQ', 'Gwydion’s Gate L11-ME'], "default":None}}

    for key, valid in valid_category.items():
        if key in observation:
            value = observation[key]
            if value not in valid_category[key]:
                error = f"{value} not valid for {key}"
                return False, error 
        else:
            error = "error"
            return False, error 
    return True, ""



def check_numericals(observation):
    o_x = observation.get("Galactic X")
    if not isinstance(o_x, float): 
        error = f"age {o_x} not valid"
        return False, error 
    o_y = observation.get("Galactic Y")
    if not isinstance(o_y, int): 
        error = f"capital-gain {o_y} not valid"
        return False, error
    return True, ""

def check_datetime(observation):
    dat = observation["Date"]
    if not isinstance(dat, str):
        error = f"date should be string"
        return False, error
    return True, ""


# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
  
    if 'observation_id' in obs_dict:
        _id = str(obs_dict['observation_id'])
        response = {}
        response["observation_id"] = _id
    else:
        _id = None
        error = "observation_id error"
        response = {'error': error}
        return jsonify(response)

    observation = obs_dict
    
    columns_ok, error = check_datetime(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    nums, error = check_numericals(observation)
    if not nums:
        response = {'error': error}
        return jsonify(response)

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response["label"] = prediction
    
    p = Prediction(
        observation_id=_id,
        prediction=prediction,
    )
    logger.info("Created variable 'p'")
    try:
        logger.info("Trying to save p...")
        p.save()
        logger.info("p saved successfully!")
    except IntegrityError:
        logger.info("Couldn't save p :(")
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
        
    return jsonify(response)

    
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['label']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
