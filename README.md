`GrabSafety` is meant to check if the trip is dangerous.

## Installation

```bash
tar -xvf GrabSafety*.tar.gz
cd GrabSafety*/
python setup.py install
```

## API usage.
```bash
from grabsafety import predict_danger

>>> predict_danger(df)
array([0.07552388, 0.07544506], dtype=float32)
```

## Data Cleaning Approach
The data cleaning has three major parts:
1) Correct the data fields.
It is observed that many people do not place their phones in the correct orientation, as some records show gravity along
x and z axis (assume phone is always upright, y axis will measure gravity). Re-arranging columns is important.
2) Sensor data filtering.
As the accelerometer data has gravity and also some noise, it will be good to use high and low pass filter for it.
Gyroscope data has drift, so using Kalman Filter would be good.
3) New features generation.
bearing_change: the bearing change is calculated sequentially. We need to adjust using 360 to minus the value if needed.
acceleration: sqrt of sum of squares for x/y/z.
speed_*: multiplication between speed and some other features such as acceleration, gyro_y, bearing, etc.
