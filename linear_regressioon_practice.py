import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Source https://mathbitsnotebook.com/Algebra1/StatisticsReg/ST2LinRegPractice.html

def problem_one():
    # Get Data

    widgets = np.array([3, 4, 5, 6]).reshape((-1, 1))
    package_length = np.array([9.00, 9.25, 9.5, 9.75])

    # fit model
    model_1 = LinearRegression()
    model_1_fit = model_1.fit(widgets, package_length)

    r_sq = model_1.score(widgets, package_length)
    print(f'r_sq {r_sq}')

    print(f'intercept {model_1.intercept_}')
    print(f'slope {model_1.coef_}')


def problem_two():
    total_fat = np.array([9, 13, 21, 30, 31, 32, 34]).reshape((-1, 1))
    total_calories = np.array([260, 320, 420, 530, 560, 580, 590])

    model_two = LinearRegression()
    model_two_fit = model_two.fit(total_fat, total_calories)

    model_two_score = model_two_fit.score(total_fat, total_calories)
    model_two_intercept = np.round(model_two.intercept_, 0)
    model_two_slope = np.round(model_two.coef_, 0)

    print(f"Model_two\n{model_two}\nmodel_two_fit\t{model_two_fit}\nmodel_two_score\t{model_two_score}")
    print(f"Model_two_intercept\t{model_two_intercept}\nmodel_two_slope\t{model_two_slope}")


def problem_three():
    time_in_min = np.array([0, 1, 2, 3, 4, 5]).reshape((-1, 1))
    tim_in_min_long = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    temp_in_f = np.array([180, 160, 138, 125, 110, 95])
    temp_in_f_ln = np.log(temp_in_f)

    print(f"time_in_min\t{time_in_min}\ntemp_in_f\t{temp_in_f}\ntemp_in_f_ln\t{temp_in_f_ln}")
    model_3 = LinearRegression()
    model_3_fit = model_3.fit(time_in_min, temp_in_f_ln)

    temp_in_f_ln_pred = np.exp(model_3.predict(tim_in_min_long))
    print(temp_in_f_ln_pred)
    print("fin")


def problem_four():
    day_of_campaign = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    no_of_coats = np.array([860, 930, 1000, 1150, 1200, 1360])
    model_4 = LinearRegression()
    model_4_fit = model_4.fit(day_of_campaign, no_of_coats)
    print(f"slope ={model_4.coef_}  intercept = {model_4.intercept_}")

    target_coats = 2100

    for test_day in range(0, 15):
        prediction_array = np.array([test_day]).reshape((-1, 1))
        day_7_prediciton = model_4.predict(prediction_array)[0]
        if day_7_prediciton > target_coats:
            print(f"Goal Met {day_7_prediciton} coats on day {test_day}")
    print('fin')

def electric_problem():
    electric_data = pd.read_csv('electric-energy.txt', sep=r'\s+')

    model_electric = LinearRegression()
    drop_cond = electric_data['energy'] != 0
    # electric_data=electric_data[drop_cond]
    energy_use = np.array(electric_data['energy']).reshape((-1, 1))
    energy_cost = np.array(electric_data['cost'])

    mef = model_electric.fit(energy_use, energy_cost)

    mes = model_electric.score(energy_use, energy_cost)

    print(f"params\n{model_electric.get_params()}")
    print(f"score\t{mes}")
    print(f"coef\t{model_electric.coef_}")
    print(f"intercept\t{model_electric.intercept_}")

    print('fin')
