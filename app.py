from flask import Flask, jsonify, request
from flask.templating import render_template
import pandas as pd
import joblib


app = Flask(__name__)

# class for Kickstarter objects with attributes corresponding to features in Matt Rager's wrangle function
# these might still change depending on what features we actually end up using
# note that these no longer match the features used by the baseline regression, so that predict function won't work


class Kickstarter:
    def __init__(self, id, backers_count, blurb, country, currency,
       currency_trailing_code, current_currency, disable_communication,
       fx_rate, goal, is_starrable, name, pledged, source_url,
       spotlight, staff_pick, state, static_usd_rate, urls,
       usd_exchange_rate, usd_pledged, pre_campaign, planned_campaign,
       actual_campaign, post_campaign, year_created, month_created,
       day_created, weekday_created, year_deadline, month_deadline,
       day_deadline, weekday_deadline, year_launched, month_launched,
       day_launched, weekday_launched, year_state_changed,
       month_state_changed, day_state_changed, weekday_state_changed,
       average_pledge_amount, percent_pledged):
        self.id = id
        self.backers_count = backers_count
        self.blurb = blurb
        self.country = country
        self.currency = currency
        self.currency_trailing_code = currency_trailing_code
        self.current_currency = current_currency
        self.disable_communication = disable_communication
        self.fx_rate = fx_rate
        self.goal = goal
        self.is_starrable = is_starrable
        self.name = name
        self.pledged = pledged  # might be leaky
        self.source_url = source_url
        self.spotlight = spotlight
        self.staff_pick = staff_pick
        self.static_usd_rate = static_usd_rate
        self.urls = urls
        self.usd_exchange_rate = usd_exchange_rate
        self.usd_pledged = usd_pledged  # might be leaky
        self.pre_campaign = pre_campaign
        self.planned_campaign = planned_campaign
        self.actual_campaign = actual_campaign
        self.post_campaign = post_campaign  # might be leaky
        self.year_created = year_created
        self.month_created = month_created
        self.day_created = day_created
        self.weekday_deadline = weekday_deadline
        self.year_launched = year_launched
        self.month_launched = month_launched
        self.day_launched = day_launched
        self.weekday_launched = weekday_launched
        self.year_state_changed = year_state_changed  # state change may be leaky in conjunction with other features?
        self.month_state_changed = month_state_changed  # state change may be leaky in conjunction with other features?
        self.day_state_changed = day_state_changed  # state change may be leaky in conjunction with other features?
        self.weekday_state_changed = weekday_state_changed  # state change may be leaky?
        self.average_pledge_amount = average_pledge_amount
        self.percent_pledged = percent_pledged  # might be leaky if duration is expired? Sub 100 after deadline = fail


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
