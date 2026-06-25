import serial
import csv

# connect to ESP32
ser = serial.Serial('COM8', 115200, timeout=1)

# number of rows to record
max_rows = 300
count = 0

with open("classroom.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # header
    writer.writerow(["time","temperature","pressure","gas","pm25","aqi"])

    try:
        while count < max_rows:

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(line)

            if "," in line:
                data = line.split(",")

                if len(data) == 6:
                    writer.writerow(data)
                    file.flush()   # force save immediately
                    count += 1

        print("Finished collecting", count, "rows")

    except KeyboardInterrupt:
        print("\nLogging stopped manually.")