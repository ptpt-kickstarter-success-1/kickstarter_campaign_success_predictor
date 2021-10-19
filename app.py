from flask import Flask, jsonify, request
from flask.templating import render_template
import pandas as pd
import joblib


app = Flask(__name__)

# class for Kickstarter objects with attributes corresponding to features in the baseline model
# these will change depending on what features we actually end up using


class Kickstarter:
    def __init__(self, goal, disable_communication, country,
                 currency, currency_trailing_code, staff_pick,
                 backers_count, static_usd_rate, category,
                 name_len, name_len_clean, blurb_len, blurb_len_clean,
                 deadline_weekday, created_at_weekday, launched_at_weekday,
                 deadline_month, deadline_day, deadline_yr,
                 deadline_hr, created_at_month, created_at_day,
                 created_at_yr, created_at_hr, launched_at_month,
                 launched_at_day, launched_at_yr, launched_at_hr):
        self.goal = goal
        self.disable_communication = disable_communication
        self.country = country
        self.currency = currency
        self.currency_trailing_code = currency_trailing_code
        self.staff_pick = staff_pick
        self.backers_count = backers_count
        self.static_usd_rate = static_usd_rate
        self.category = category
        self.name_len = name_len
        self.name_len_clean = name_len_clean
        self.blurb_len = blurb_len
        self.blurb_len_clean = blurb_len_clean
        self.deadline_weekday = deadline_weekday
        self.created_at_weekday = created_at_weekday
        self.launched_at_weekday = launched_at_weekday
        self.deadline_month = deadline_month
        self.deadline_day = deadline_day
        self.deadline_yr = deadline_yr
        self.deadline_hr = deadline_hr
        self.created_at_month = created_at_month
        self.created_at_day = created_at_day
        self.created_at_yr = created_at_yr
        self.created_at_hr = created_at_hr
        self.launched_at_month = launched_at_month
        self.launched_at_day = launched_at_day
        self.launched_at_yr = launched_at_yr
        self.launched_at_hr = launched_at_hr

    def predict_success(self):
        temp = pd.DataFrame(
                columns=['goal', 'disable_communication', 'country', 'currency',
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


@app.route('/inputs/', methods=['GET', 'POST'])
def process_input():
    return 'this is where user input is taken and stored as a Kickstarter object'


# predict route currently has a test object lifted from the validation set of the baseline model
# once we have front end input we'll generate Kickstarters from that


@app.route('/predict/', methods=['GET', 'POST'])
def output_prediction():
    test_object = Kickstarter(3200, False, 'CA', 'CAD', True, False, 74, 0.818860, 'Plays', 4.0, 4.0, 21.0, 21.0, 'Tuesday', 'Monday', 'Tuesday', 5, 26, 2015, 21, 4, 6, 2015, 21, 4, 21, 2015, 16)
    prediction = test_object.predict_success()
    return prediction


if __name__ == '__main__':
    ks_baseline_pipeline = joblib.load('ks_baseline.joblib')
    app.run()
