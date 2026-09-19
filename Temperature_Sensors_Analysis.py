# NumPy Assignment - Real-World Data Work
import numpy as np

# show numbers with 2 decimals so the output is easy to read
np.set_printoptions(precision=2, suppress=True)
print("=" * 50)
print("PROBLEM 1")
print("=" * 50)

# --- 1.1 ---
data = np.loadtxt("sensor_readings.csv", delimiter=",")
print("1.1 shape:", data.shape)
print("1.1 dtype:", data.dtype)
# shape is (30, 6) -> 30 rows = 30 days, 6 columns = 6 sensors

# --- 1.2 ---
# (data == -999) gives True/False, and True counts as 1 when we sum
total_broken = np.sum(data == -999)
# axis=0 goes down the rows, so we get one number per column (per sensor)
broken_per_sensor = np.sum(data == -999, axis=0)
print("\n1.2 total -999 readings:", total_broken)
print("1.2 -999 readings per sensor:", broken_per_sensor)

# --- 1.3 ---
# np.where builds a NEW array, so the original data stays the same
clean = np.where(data == -999, np.nan, data)
print("\n1.3 NaN values in the new array:", np.sum(np.isnan(clean)))

# --- 1.4 ---
faulty = np.argmax(broken_per_sensor)  # sensor with the most broken readings
faulty_pct = broken_per_sensor[faulty] / data.shape[0] * 100
print("\n1.4 faulty sensor number:", faulty)
print("1.4 percentage of its broken readings: {:.1f}%".format(faulty_pct))

# --- 1.5 ---
# nanmean ignores the NaN values, normal mean would return NaN
sensor_avg = np.nanmean(clean, axis=0)  # axis=0 -> one average per sensor
print("\n1.5(a) average of each sensor:", sensor_avg)

day_avg = np.nanmean(clean, axis=1)  # axis=1 -> one average per day
print("1.5(b) average of each day:", day_avg)

hottest = np.argmax(sensor_avg)
print("1.5(c) hottest sensor on average: sensor", hottest,
      "with", round(sensor_avg[hottest], 2), "C")

# --- 1.6 ---
# np.where returns the positions (day numbers) where the condition is True
hot_days = np.where(day_avg > 26)[0]
print("\n1.6 days with average above 26 C:", hot_days.tolist())
print("1.6 number of such days:", len(hot_days))

# --- 1.7 ---
# np.delete removes the whole column of the faulty sensor (axis=1)
clean_no_faulty = np.delete(clean, faulty, axis=1)
print("\n1.7 new shape:", clean_no_faulty.shape)
print("1.7 overall average BEFORE removing:", round(np.nanmean(clean), 2))
print("1.7 overall average AFTER removing:", round(np.nanmean(clean_no_faulty), 2))

# --- 1.8 ---
# min and max of EACH column (axis=0), shape (5,)
col_min = np.nanmin(clean_no_faulty, axis=0)
col_max = np.nanmax(clean_no_faulty, axis=0)
# broadcasting: the (30,5) array and the (5,) arrays match on the last axis,
# so every column uses its own min and max without any loop
normalized = (clean_no_faulty - col_min) / (col_max - col_min)
print("\n1.8 shape:", normalized.shape)
print("1.8 min of each column:", np.nanmin(normalized, axis=0))
print("1.8 max of each column:", np.nanmax(normalized, axis=0))

# --- 1.9 ---
# first 28 days -> reshape to 4 weeks x 7 days
weeks = day_avg[:28].reshape(4, 7)
week_avg = weeks.mean(axis=1)  # axis=1 -> average across the 7 days of each week
print("\n1.9 shape:", weeks.shape)
print("1.9 average temperature of each week:", week_avg)
# +1 because argmax starts from 0 but weeks are numbered from 1
print("1.9 warmest week number:", np.argmax(week_avg) + 1)

# --- Bonus A ---
# repair the faulty sensor instead of removing it:
# replace its NaN values with the average of the other sensors on the same day
other_sensors_day_avg = np.nanmean(clean_no_faulty, axis=1)  # one value per day
faulty_col = clean[:, faulty]
repaired_col = np.where(np.isnan(faulty_col), other_sensors_day_avg, faulty_col)
print("\nBonus A: faulty sensor average BEFORE repair:", round(np.nanmean(faulty_col), 2))
print("Bonus A: faulty sensor average AFTER repair:", round(repaired_col.mean(), 2))
