from machine import Pin, I2C
import time
import collections
from decision_tree_25hz_acc_x_acc_y_acc_z_gyro_x import score

# --- MPU6886 Setup ---
i2c = I2C(scl=Pin(22), sda=Pin(21), freq=400000)
MPU_ADDR = 0x68

# Wake up MPU6886 (exit sleep mode)
try:
    i2c.writeto_mem(MPU_ADDR, 0x6B, b'\x00')
except Exception as e:
    print("Failed to initialize MPU6886:", e)

# --- Labels (based on model output index) ---
labels = ["circle", "x", "y", "other"]  # Ensure it matches score() output

# --- Sliding Window Buffer ---
WINDOW_SIZE = 26  # Time steps
window = collections.deque([], 26)


# --- Sensor Read Function ---
def read_sensor():
    try:
        accel = i2c.readfrom_mem(MPU_ADDR, 0x3B, 6)
        gyro = i2c.readfrom_mem(MPU_ADDR, 0x43, 6)

        def twos_complement(high, low):
            value = (high << 8) | low
            return value - 65536 if high & 0x80 else value

        ax = twos_complement(accel[0], accel[1]) / 2048.0 * 9.81
        ay = twos_complement(accel[2], accel[3]) / 2048.0 * 9.81
        az = twos_complement(accel[4], accel[5]) / 2048.0 * 9.81

        gx = twos_complement(gyro[0], gyro[1]) / 16.4 * 0.0174533
        gy = twos_complement(gyro[2], gyro[3]) / 16.4 * 0.0174533
        gz = twos_complement(gyro[4], gyro[5]) / 16.4 * 0.0174533

        return [ax, ay, az, gx, gy, gz]

    except Exception as e:
        print("Sensor read error:", e)
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Safe fallback

# --- Main Loop ---
while True:
    try:
        features = read_sensor()
        window.append(features)

        if len(window) == WINDOW_SIZE:
            # Flatten buffer: [ [f1..f6], [f1..f6], ..., ] → [f1, f2, ..., f156]
            flat_input = [val for frame in window for val in frame]

            # Pad to match model input length if needed (some models expect 158)
            while len(flat_input) < 158:
                flat_input.append(0.0)

            prediction = score(flat_input)
            class_index = prediction.index(max(prediction))

            if 0 <= class_index < len(labels):
                print("Movement:", labels[class_index])
            else:
                print("Unknown movement class:", class_index)

    except Exception as e:
        print("Main loop error:", e)

    time.sleep(0.04)  # 25Hz sampling




