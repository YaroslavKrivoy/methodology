import pmdarima as pm
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv('files/Daily-stats.csv', sep=";")
    data = data.rename(columns={'FL_DATE': 'date', 'FLIGHTS': 'flights'})
    data = data.drop(columns=['CANCELLED', 'DISTANCE'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(data['date'], inplace=True)
    data = data.drop(columns=['date'])

    # train, test = train_test_split(data, test_size=0.8)

    train = data[:200]
    test = data[200:100]

    model = auto_arima(train, start_p=0, d=1, start_q=0,
                       max_p=5, max_d=5, max_q=5, start_P=0,
                       D=1, start_Q=0, max_P=5, max_D=5,
                       max_Q=5, m=12, seasonal=True,
                       error_action='warn', trace=True,
                       supress_warnings=True, stepwise=True,
                       random_state=20, n_fits=50)

    prediction = pd.DataFrame(model.predict(n_periods=len(test)), index=test.index)
    prediction.columns = ['predicted_flights']
    plt.figure(figsize=(8, 5))
    plt.plot(train, label="Training")
    plt.plot(test, label="Test")
    plt.plot(prediction, label="Predicted")
    plt.show()
