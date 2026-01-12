import csv
import os

class QuaternionLogger:
    def __init__(self, filename="quat_log.csv"):
        """
        filename: 输出的日志文件名
        """
        self.filename = filename
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)

        # 写表头（Excel友好）
        self.writer.writerow([
            "time",
            "meas_qw", "meas_qx", "meas_qy", "meas_qz",
            "true_qw", "true_qx", "true_qy", "true_qz",
            "wx","wy","wz"
        ])

    def log(self, time, q_meas, q_true, gyro):
        """
        time: 当前仿真时间 (float)
        q_meas: 测量四元数 [w, x, y, z]
        q_true: 真值四元数 [w, x, y, z]
        gyro: gyro
        """
        self.writer.writerow([
            time,
            q_meas[0], q_meas[1], q_meas[2], q_meas[3],
            q_true[0], q_true[1], q_true[2], q_true[3],
            gyro[0],gyro[1],gyro[2]
        ])

    def close(self):
        self.file.close()
