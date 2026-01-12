import csv
import numpy as np
import matplotlib.pyplot as plt

from solver.solver import quatShit
from observer.observer import EKF as AttitudeEKF7D


def read_quat_log(csv_path: str):
    t_list = []
    q_meas_list = []
    q_true_list = []
    w_list = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time"])

            q_meas = np.array(
                [float(row["meas_qw"]), float(row["meas_qx"]), float(row["meas_qy"]), float(row["meas_qz"])],
                dtype=float,
            )
            q_true = np.array(
                [float(row["true_qw"]), float(row["true_qx"]), float(row["true_qy"]), float(row["true_qz"])],
                dtype=float,
            )
            w = np.array([float(row["wx"]), float(row["wy"]), float(row["wz"])], dtype=float)

            q_meas = np.asarray(quatShit.quat_normalize(q_meas), dtype=float)
            q_true = np.asarray(quatShit.quat_normalize(q_true), dtype=float)

            if len(q_meas_list) > 0:
                if np.dot(q_meas, q_meas_list[-1]) < 0:
                    q_meas = -q_meas
                if np.dot(q_true, q_true_list[-1]) < 0:
                    q_true = -q_true

            t_list.append(t)
            q_meas_list.append(q_meas)
            q_true_list.append(q_true)
            w_list.append(w)

    t_arr = np.asarray(t_list, dtype=float)
    q_meas_arr = np.asarray(q_meas_list, dtype=float)  # (N,4)
    q_true_arr = np.asarray(q_true_list, dtype=float)  # (N,4)
    w_arr = np.asarray(w_list, dtype=float)            # (N,3)
    return t_arr, q_meas_arr, q_true_arr, w_arr


def quat_angle_error_rad(q_est, q_true):
    """
    返回姿态误差角（rad），用 q_err = q_true ⊗ inv(q_est)，取最短旋转角：
      angle = 2*acos(clamp(q_err.w, -1, 1)), 并映射到 [0, pi]
    """
    q_est = np.asarray(quatShit.quat_normalize(q_est), dtype=float)
    q_true = np.asarray(quatShit.quat_normalize(q_true), dtype=float)
    q_err = np.asarray(quatShit.quat_mul(q_true, quatShit.quat_inv(q_est)), dtype=float)

    if q_err[0] < 0:
        q_err = -q_err
    w = float(np.clip(q_err[0], -1.0, 1.0))
    ang = 2.0 * np.arccos(w)
    if ang > np.pi:
        ang = 2.0 * np.pi - ang
    return ang


def run_ekf_replay(csv_path="quat_log.csv"):
    t, q_meas, q_true, w = read_quat_log(csv_path)

    x0 = np.zeros((7, 1), dtype=float)
    x0[0:4, 0] = q_meas[0]
    ekf = AttitudeEKF7D(x0=x0)

    q_pred_hist = np.zeros_like(q_true)  # predict后、update前（先验）
    q_post_hist = np.zeros_like(q_true)  # update后（后验）
    b_hist = np.zeros((len(t), 3), dtype=float)

    for k in range(len(t)):
        if k == 0:
            dt = float(t[1] - t[0]) if len(t) > 1 else 0.002
        else:
            dt = float(t[k] - t[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1e-3

        # 预测
        ekf.predict(w[k], dt)
        q_pred = ekf.get_quat()
        q_pred = np.asarray(quatShit.quat_normalize(q_pred), dtype=float)

        # 让预测四元数的符号与真值对齐
        if np.dot(q_pred, q_true[k]) < 0:
            q_pred = -q_pred
        q_pred_hist[k] = q_pred

        # 更新
        ekf.update(q_meas[k])
        q_post = ekf.get_quat()
        q_post = np.asarray(quatShit.quat_normalize(q_post), dtype=float)
        if np.dot(q_post, q_true[k]) < 0:
            q_post = -q_post
        q_post_hist[k] = q_post

        b_hist[k] = ekf.get_bias()

    # 计算误差角曲线（先验 vs 后验）
    err_pred = np.array([quat_angle_error_rad(q_pred_hist[i], q_true[i]) for i in range(len(t))], dtype=float)
    err_post = np.array([quat_angle_error_rad(q_post_hist[i], q_true[i]) for i in range(len(t))], dtype=float)

    return t, q_meas, q_true, q_pred_hist, q_post_hist, b_hist, err_pred, err_post


def plot_quats(t, q_meas, q_true, q_pred, q_post, err_pred, err_post, b_hist):
    labels = ["w", "x", "y", "z"]

    fig1 = plt.figure(figsize=(14, 10))
    for i in range(4):
        ax = fig1.add_subplot(4, 1, i + 1)
        ax.plot(t, q_meas[:, i], label=f"meas q{labels[i]}", linewidth=1.0)
        ax.plot(t, q_true[:, i], label=f"true q{labels[i]}", linewidth=1.0)
        ax.plot(t, q_pred[:, i], label=f"pred(prior) q{labels[i]}", linewidth=1.0)
        ax.plot(t, q_post[:, i], label=f"post(posterior) q{labels[i]}", linewidth=1.0)
        ax.set_ylabel(f"q{labels[i]}")
        ax.grid(True)
        if i == 0:
            ax.legend(ncol=4, fontsize=9, loc="upper right")
    fig1.suptitle("Quaternion components: meas vs true vs EKF prior vs EKF posterior")
    fig1.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    fig2 = plt.figure(figsize=(14, 4))
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(t, np.rad2deg(err_pred), label="angle error (prior)", linewidth=1.2)
    ax.plot(t, np.rad2deg(err_post), label="angle error (posterior)", linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle error (deg)")
    ax.grid(True)
    ax.legend()
    fig2.tight_layout()

    fig3 = plt.figure(figsize=(14, 6))
    ax = fig3.add_subplot(1, 1, 1)
    ax.plot(t, b_hist[:, 0], label="bwx")
    ax.plot(t, b_hist[:, 1], label="bwy")
    ax.plot(t, b_hist[:, 2], label="bwz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gyro bias (rad/s)")
    ax.grid(True)
    ax.legend()
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    csv_path = "quat_log.csv"  
    t, q_meas, q_true, q_pred, q_post, b_hist, err_pred, err_post = run_ekf_replay(csv_path)
    plot_quats(t, q_meas, q_true, q_pred, q_post, err_pred, err_post, b_hist)
