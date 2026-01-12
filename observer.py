import numpy as np
from solver.solver import dataSaver, quatShit


class EKF:
    """
    7D EKF for quadrotor attitude:
      x = [qw, qx, qy, qz, bwx, bwy, bwz]^T
    Measurement:
      z = IMU quaternion (4D), H = [I4 | 0]
    """

    def __init__(self, x0=None, P0=None):
        # ---- state ----
        if x0 is None:
            self.x = np.zeros((7, 1), dtype=float)
            self.x[0, 0] = 1.0
        else:
            x0 = np.asarray(x0, dtype=float).reshape(7, 1)
            self.x = x0.copy()
            self._normalize_q_inplace()

        # ---- covariance ----
        if P0 is None:
            P = np.eye(7, dtype=float)
            P[0:4, 0:4] *= 1e-3
            P[4:7, 4:7] *= 1e-2
            self.P = P
        else:
            self.P = np.asarray(P0, dtype=float).reshape(7, 7).copy()
            self._symmetrize_P_inplace()

        # ----- measurement noise in ANGLE domain (3x3) -----
        self.sigma_theta_meas = np.deg2rad(0.1)  # rad
        self.R_theta = (self.sigma_theta_meas ** 2) * np.eye(3, dtype=float)

        # ---- process noise params ----
        self.sigma_omega = np.deg2rad(0.05)      # rad/s
        self.sigma_bias_rw = np.sqrt(1e-11)

        self.I7 = np.eye(7, dtype=float)

    # =========================
    # Public API
    # =========================
    def predict(self, u_omega_raw, dt):
        """
        Predict with raw angular velocity input u (3,).
        Quaternion propagation uses first-order (Euler) discretization:
          q_{k+1} = q_k + 0.5*dt * ( q_k ⊗ [0, ω] ),  with ω = u - b
        Bias assumed random walk:
          b_{k+1} = b_k
        """
        dt = float(dt)
        if dt <= 0:
            return

        u = np.asarray(u_omega_raw, dtype=float).reshape(3, 1)

        q = self.x[0:4, 0:4].copy()
        b = self.x[4:7, 0:1].copy()

        omega = u - b  # bias-corrected body rates

        # --- state propagation ---
        q_next = self._quat_euler_step(q, omega, dt)
        b_next = b

        self.x[0:4, 0] = q_next[:, 0]
        self.x[4:7, 0] = b_next[:, 0]
        self._normalize_q_inplace()

        # --- linearization: F and Q ---
        F = self._compute_F(q, omega, dt)  # linearize around (q_k, b_k)
        Q = self._compute_Q(q, dt)         # use q_k for mapping rate noise -> quaternion noise

        # --- covariance propagation ---
        P = F @ self.P @ F.T + Q
        self.P = P
        self._symmetrize_P_inplace()

        # safety: finite
        self.x = dataSaver.finite_array(self.x)
        self.P = dataSaver.finite_array(self.P)

    def update(self, z_quat):
        """
        Quaternion error (multiplicative) update:
          q_err = q_meas ⊗ q_pred^{-1}, enforce q_err.w >= 0
          innovation y = δθ ≈ 2 * q_err_vec   (3x1)
        Use Jacobian: H = [2 E(q_pred)^T | 0]  (3x7)
        Then apply correction by left-multiplying:
          q_new = δq_corr ⊗ q_pred
        Bias updated additively.
        """
        z = np.asarray(z_quat, dtype=float).reshape(4,)
        z = np.asarray(quatShit.quat_normalize(z), dtype=float).reshape(4,)

        q_pred = self.x[0:4, 0].reshape(4,)
        q_pred = np.asarray(quatShit.quat_normalize(q_pred), dtype=float).reshape(4,)

        # q_err = z ⊗ inv(q_pred)
        q_pred_inv = np.asarray(quatShit.quat_inv(q_pred), dtype=float).reshape(4,)
        q_err = np.asarray(quatShit.quat_mul(z, q_pred_inv), dtype=float).reshape(4,)

        # shortest rotation: enforce scalar >= 0
        if q_err[0] < 0.0:
            q_err = -q_err

        # innovation in angle space
        y = (2.0 * q_err[1:4]).reshape(3, 1)  # δθ (3x1)
        y = dataSaver.finite_array(y)

        # build H_theta = [2 E(q)^T | 0]
        q_mat = q_pred.reshape(4, 1)
        E = self._E_matrix(q_mat)                 # (4x3)
        H = np.zeros((3, 7), dtype=float)
        H[:, 0:4] = 2.0 * E.T                     # (3x4)

        # EKF update
        S = H @ self.P @ H.T + self.R_theta
        S = dataSaver.finite_array(S)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ H.T @ S_inv                  # (7x3)

        dx = K @ y                                # (7x1)
        dx = dataSaver.finite_array(dx)

        # ---- apply correction ----
        # bias add
        self.x[4:7, 0:1] = self.x[4:7, 0:1] + dx[4:7, 0:1]

        # quaternion: convert dx_q(4) -> delta_theta_hat via δθ ≈ 2 E(q)^T δq
        dq_add = dx[0:4, 0:1]                      # (4x1) "additive quat delta" variable inside EKF
        dtheta_hat = 2.0 * (E.T @ dq_add)          # (3x1)

        dq_corr = self._quat_from_small_angle(dtheta_hat)  # (4,)
        q_new = np.asarray(quatShit.quat_mul(dq_corr, q_pred), dtype=float).reshape(4,)
        q_new = np.asarray(quatShit.quat_normalize(q_new), dtype=float).reshape(4,)

        self.x[0:4, 0] = q_new

        # Joseph covariance update
        I_KH = self.I7 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_theta @ K.T

        # robustness requirements
        self._normalize_q_inplace()
        self._symmetrize_P_inplace()

        self.x = dataSaver.finite_array(self.x)
        self.P = dataSaver.finite_array(self.P)


    # =========================
    # Core math
    # =========================
    @staticmethod
    def _omega_matrix(omega):
        """
        G(ω) such that: qdot = 0.5 * G(ω) q
        omega: (3,1) = [wx, wy, wz]^T
        """
        wx, wy, wz = float(omega[0, 0]), float(omega[1, 0]), float(omega[2, 0])
        return np.array([
            [0.0, -wx, -wy, -wz],
            [wx,  0.0,  wz, -wy],
            [wy, -wz,  0.0,  wx],
            [wz,  wy, -wx,  0.0],
        ], dtype=float)


    def _quat_euler_step(self, q, omega, dt):
        """
        First-order discretization:
          q_{k+1} = q_k + 0.5*dt * ( q_k ⊗ [0, ω] )
        """
        # build pure quaternion [0, ω]
        wq = np.array([0.0, float(omega[0, 0]), float(omega[1, 0]), float(omega[2, 0])], dtype=float)

        # use provided quaternion multiply (expects array-like)
        q_arr = q.reshape(4,)
        dq = quatShit.quat_mul(q_arr, wq)  # q ⊗ [0,ω]
        dq = np.asarray(dq, dtype=float).reshape(4, 1)

        q_next = q + 0.5 * dt * dq
        q_next = np.asarray(quatShit.quat_normalize(q_next.reshape(4,)), dtype=float).reshape(4, 1)
        return q_next

    def _compute_F(self, q, omega, dt):
        """
        State transition Jacobian F (7x7) for Euler discretization:

        q_{k+1} = (I4 + 0.5*dt*G(ω)) q_k
        ω = u - b

        => F_qq = I4 + 0.5*dt*G(ω)

        Also:
        q_{k+1} = q_k + 0.5*dt * E(q_k) * (u - b_k)
        => ∂q_{k+1}/∂b = -0.5*dt * E(q_k)

        Bias:
        b_{k+1} = b_k  => F_bb = I3
        """
        F = np.eye(7, dtype=float)

        G = self._omega_matrix(omega)
        Fqq = np.eye(4, dtype=float) + 0.5 * dt * G

        E = self._E_matrix(q)
        Fqb = -0.5 * dt * E  # (4x3)

        F[0:4, 0:4] = Fqq
        F[0:4, 4:7] = Fqb
        # F[4:7,0:4] = 0 already
        # F[4:7,4:7] = I already
        return F

    def _compute_Q(self, q, dt):
        """
        Process noise:
          - attitude driven by angular-rate noise (sigma_omega rad/s)
            q_{k+1} ≈ q_k + 0.5*dt*E(q_k)*ω
            So quaternion noise covariance:
              Qq = (0.5*dt)^2 * E(q) * (sigma_omega^2 I3) * E(q)^T

          - bias random walk, discretized:
              Qb = (sigma_bias_rw^2) * dt * I3
            (Very small as requested; sigma_bias_rw^2 = 1e-11)
        """
        Q = np.zeros((7, 7), dtype=float)

        E = self._E_matrix(q)
        Sigma_omega = (self.sigma_omega ** 2) * np.eye(3, dtype=float)
        Qq = (0.5 * dt) ** 2 * (E @ Sigma_omega @ E.T)

        Q[0:4, 0:4] = Qq

        Qb = (self.sigma_bias_rw ** 2) * dt * np.eye(3, dtype=float)
        Q[4:7, 4:7] = Qb

        return Q

    # =========================
    # Robustness helpers
    # =========================
    @staticmethod
    def _quat_from_small_angle(dtheta):
        """
        small-angle quaternion (left-multiplicative):
          δq ≈ [1, 0.5*δθ]
        dtheta: (3,1)
        """
        dtheta = np.asarray(dtheta, dtype=float).reshape(3,)
        dq = np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=float)
        dq = np.asarray(quatShit.quat_normalize(dq), dtype=float).reshape(4,)
        return dq

    # ---- keep these from your previous version ----
    @staticmethod
    def _E_matrix(q):
        """
        E(q) such that: q ⊗ [0, ω] = E(q) ω
        with q = [qw,qx,qy,qz]^T
        """
        qw, qx, qy, qz = float(q[0, 0]), float(q[1, 0]), float(q[2, 0]), float(q[3, 0])
        return np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw],
        ], dtype=float)

    def _normalize_q_inplace(self):
        q = self.x[0:4, 0].reshape(4,)
        qn = quatShit.quat_normalize(q)
        self.x[0:4, 0] = np.asarray(qn, dtype=float).reshape(4,)

    def _symmetrize_P_inplace(self):
        self.P = 0.5 * (self.P + self.P.T)
        self.P = dataSaver.finite_array(self.P)
    # Convenience getters
    def get_quat(self):
        return self.x[0:4, 0].copy()

    def get_bias(self):
        return self.x[4:7, 0].copy()
