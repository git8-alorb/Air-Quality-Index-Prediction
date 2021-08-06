# Air-Quality-Index-Prediction
Prediction of particulate matters and suspended particles for outdoor air quality.


The Air Quality Index is based on measurement of particulate matter (PM2.5 and PM10), Ozone (O3), Nitrogen Dioxide (NO2), Sulphur Dioxide (SO2),  Carbon Monoxide (CO) and Ozone (O3) emissions. The air quality index of a particular data point is the aggregate of maximum indexed pollutant on that particular area. That pollutants max sub index is taken as the air quality index of that particular location. Basically, for outdoor air pollution, PM2.5 has the highest value among all for most of the case. So, PM2.5 has been considered here for Air Quality Prediction.


## Dataset
US_Embassy New Delhi dataset and DTU New Delhi dataset for particulate matter PM2.5 has been taken here for the trainig and prediction of AQI model. <br/>
Link:- https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-comparison-data


## Model
Here ARIMA model is used for the training and testing of data.<br/>
/python_files/DTU_using_ARIMA.py:- <br/>
In this python file, the model is trained on DTU New Delhi dataset using ARIMA model and then tested on 70% of data. Next 7 days prediction                                         is also showed for the same model.

/python_files/AQI_pred_US_Embassy_ND_ARIMA.py:- <br/> In this python file, the model is trained on whole US_embassy, ND dataset and then tested on 70% of the data. Next 7 days                                                          prediction is also showed for the same.

/python_files/AQI_pred_US_Embassy_ND_ARIMA_Incremental_training.py:-  <br/>In this python file, the model is first trained on 66% of data and predicted next one data. Then the original                                               data corresponding to that predicted data is also taken along with original train and model is trained again and this is repeated                                                   till the last testing data.

For facesafe attendence work and detail ppt of above project refer this link:- <br/>
          Link:- https://drive.google.com/drive/folders/1XIhY5Y5-xcFBgqLW8TvFJBsrI4RVs8TH?usp=sharing
