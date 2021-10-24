from flask import Flask, jsonify, request
from flask.templating import render_template
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)

# class for Kickstarter objects with attributes now corresponding to encoded feature set used by neural network model
# note that these no longer match the features used by the baseline regression, so that predict function does nothing


class Kickstarter:
    def __init__(self, df):  # __init__ now shortened to take df input (which is what we'll be using anyway) directly
        feature = []
        for x in range(len(df.columns)):
            feature.append(df.iloc[0][x])

        self.id = feature[0]
        self.backers_count = feature[1]
        self.country = feature[2]
        self.currency = feature[3]
        self.currency_trailing_code = feature[4]
        self.current_currency = feature[5]
        self.disable_communication = feature[6]
        self.fx_rate = feature[7]
        self.goal = feature[8]
        self.is_starrable = feature[9]
        self.staff_pick = feature[10]
        self.static_usd_rate = feature[11]
        self.usd_exchange_rate = feature[12]
        self.usd_pledged = feature[13]  # might be leaky
        self.pre_campaign = feature[14]
        self.planned_campaign = feature[15]
        self.actual_campaign = feature[16]
        self.post_campaign = feature[17]  # might be leaky
        self.year_created = feature[18]
        self.month_created = feature[19]
        self.day_created = feature[20]
        self.weekday_created = feature[21]
        self.year_deadline = feature[22]
        self.month_deadline = feature[23]
        self.day_deadline = feature[24]
        self.weekday_deadline = feature[25]
        self.year_launched = feature[26]
        self.month_launched = feature[27]
        self.day_launched = feature[28]
        self.weekday_launched = feature[29]
        self.year_state_changed = feature[30]  # state change may be leaky in conjunction with other features?
        self.month_state_changed = feature[31]  # state change may be leaky in conjunction with other features?
        self.day_state_changed = feature[32]  # state change may be leaky in conjunction with other features?
        self.weekday_state_changed = feature[33]  # state change may be leaky?
        self.average_pledge_amount = feature[34]
        self.blurb_vector_length = feature[35]
        self.name_vector_length = feature[36]
        self.features = df  # features stored redundantly here for ease of calling in barebones flask app

    def nn_predict(self):  # predictor function using loaded neural network
        loaded_model = keras.models.load_model("ks_NN_model.h5")
        if loaded_model.predict(self.features) > .5:
            return 'success'
        else:
            return 'failure'

    def predict_success(self):  # predictor based on the baseline logistic regression model, now nonfunctional
        temp = pd.DataFrame(
                columns=['id', 'disable_communication', 'country', 'currency',
                         'currency_trailing_code', 'staff_pick', 'backers_count',
                         'static_usd_rate', 'category', 'name_len', 'name_len_clean',
                         'blurb_len', 'blurb_len_clean', 'deadline_weekday',
                         'created_at_weekday', 'launched_at_weekday', 'deadline_month',
                         'deadline_day', 'deadline_yr', 'deadline_hr', 'created_at_month',
                         'created_at_day', 'created_at_yr', 'created_at_hr', 'launched_at_month',
                         'launched_at_day', 'launched_at_yr', 'launched_at_hr'],
                data=[[self.goal, self.disable_communication, self.country, self.currency, self.currency_trailing_code,
                       self.staff_pick, self.backers_count, self.static_usd_rate, self.category, self.name_len,
                       self.name_len_clean, self.blurb_len, self.blurb_len_clean, self.deadline_weekday,
                       self.created_at_weekday, self.launched_at_weekday, self.deadline_month, self.deadline_day,
                       self.deadline_yr, self.deadline_hr, self.created_at_month, self.created_at_day,
                       self.created_at_yr, self.created_at_hr, self.launched_at_month, self.launched_at_day,
                       self.launched_at_yr, self.launched_at_hr]])
        y_pred = ks_baseline_pipeline.predict(temp)[0]
        return y_pred




@app.route('/')
def index():
    return 'Predict success of kickstarter campaigns in the 10 item test sample with /predict[n]/ routes'


@app.route('/inputs/', methods=['GET', 'POST'])
def process_input():
    return 'this is where user input is taken and stored as a Kickstarter object'


# predict routes currently use campaigns lifted from the included test set of the neural network
# once we have front end input we'll generate Kickstarters from that


@app.route('/predict1/', methods=['GET', 'POST'])
def output_prediction1():
    sample = test_set.iloc[0:1]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict2/', methods=['GET', 'POST'])
def output_prediction2():
    sample = test_set.iloc[1:2]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict3/', methods=['GET', 'POST'])
def output_prediction3():
    sample = test_set.iloc[2:3]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict4/', methods=['GET', 'POST'])
def output_prediction4():
    sample = test_set.iloc[3:4]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict5/', methods=['GET', 'POST'])
def output_prediction5():
    sample = test_set.iloc[4:5]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict6/', methods=['GET', 'POST'])
def output_prediction6():
    sample = test_set.iloc[5:6]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict7/', methods=['GET', 'POST'])
def output_prediction7():
    sample = test_set.iloc[6:7]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict8/', methods=['GET', 'POST'])
def output_prediction8():
    sample = test_set.iloc[7:8]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict9/', methods=['GET', 'POST'])
def output_prediction9():
    sample = test_set.iloc[8:9]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


@app.route('/predict10/', methods=['GET', 'POST'])
def output_prediction10():
    sample = test_set.iloc[9:10]
    test_object = Kickstarter(sample)
    prediction = test_object.nn_predict()
    return prediction


if __name__ == '__main__':
    #  model = keras.models.load_model("ks_NN_model")
    test_set = pd.read_csv('ks_test_set.csv', index_col=0)
    app.run()
