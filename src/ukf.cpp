#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.15;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.67;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Additional Initializations
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  int n_sig = n_aug_*2 + 1;
  P_.fill(0.0);
  for (int i = 0; i < n_x_; i++) {
    P_(i, i) = 0.2;
  }
  time_us_ = 0;
  Xsig_pred_ = MatrixXd(n_x_, n_sig);
  weights_ = VectorXd(n_sig);
  lambda_ =  3 - n_aug_;
  NIS_radar_ = 0;
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
 /*****************************************************************************
  *  Initialization
  ****************************************************************************/
  if (!is_initialized_) {
    cout << "UKF: " << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.

      // Polar Measurements
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float rhoDot = meas_package.raw_measurements_(2);

      // Convert and update estimate, update timestamp, and intialization
      x_ << sin(phi)*rho, cos(phi)*rho, 0, 0, 0;
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1], 0, 0, 0;
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
 /*****************************************************************************
  *  Prediction
  ****************************************************************************/
  // dt - expressed in seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Call prediction
  Prediction(dt);

  // Call correct update for sensor type
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Initialize variables
  int n_sig = n_aug_ * 2 + 1;
  VectorXd x_aug(7);
  MatrixXd P_aug(7, 7);
  MatrixXd Xsig_aug(n_aug_, n_sig);

  // create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  double delta_t_sq = delta_t*delta_t;

  // predict sigma points
  for (int i = 0; i < (n_sig); i++) {
      // Set variables
      double p_x = Xsig_aug(0, i);
      double p_y = Xsig_aug(1, i);
      double v = Xsig_aug(2, i);
      double yaw = Xsig_aug(3, i);
      double yaw_rate = Xsig_aug(4, i);
      double std_v = Xsig_aug(5, i);
      double std_yaw_rate = Xsig_aug(6, i);

      if (yaw_rate == 0.0) {
        Xsig_pred_(0, i) = p_x + (v)*cos(yaw)*delta_t;
        Xsig_pred_(1, i) = p_y + (v)*sin(yaw)*delta_t;
      } else {
        Xsig_pred_(0, i) = p_x + (v/yaw_rate)*(sin(yaw + yaw_rate*delta_t) -
                           sin(yaw)) + 0.5*delta_t_sq*cos(yaw)*std_v;
        Xsig_pred_(1, i) = p_y + (v/yaw_rate)*(-cos(yaw + yaw_rate*delta_t) +
                           cos(yaw)) + 0.5*delta_t_sq*sin(yaw)*std_v;
      }

      Xsig_pred_(2, i) = v + delta_t*std_v;
      Xsig_pred_(3, i) = yaw + yaw_rate*delta_t + 0.5*delta_t_sq*std_yaw_rate;
      Xsig_pred_(4, i) = yaw_rate + delta_t*std_yaw_rate;
  }

  // set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig; i++) {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }
  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig; i++) {
      x_ += weights_(i) * Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig; i++) {
    VectorXd xsig_pred_diff = (Xsig_pred_.col(i) - x_);
    while (xsig_pred_diff(3) < -M_PI) {
      xsig_pred_diff(3) + 2*M_PI;
    }
    while (xsig_pred_diff(3) > M_PI) {
      xsig_pred_diff(3) - 2*M_PI;
    }
    P_ += weights_(i) * (xsig_pred_diff * xsig_pred_diff.transpose());
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Laser updates
  MatrixXd H(2, n_x_);
  MatrixXd R(2, 2);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
  R << std_laspx_, 0,
       0, std_laspy_;

  // Get measurement values
  VectorXd z = meas_package.raw_measurements_;

  // Standard Kalman Filter
  VectorXd z_pred = H * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H) * P_;

  // Calculate NIS
  NIS_laser_ = y.transpose() * Si * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Initialize variables
  int n_z = 3;
  MatrixXd Zsig(n_z, 2*n_aug_+1);
  VectorXd z_pred(n_z);
  MatrixXd Tc(n_x_, n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  VectorXd z(3);
  z = meas_package.raw_measurements_;
  S.fill(0.0);
  z_pred.fill(0.0);

  // Convert Sigma point predictions to measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v  = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    float rho = sqrt(p_x*p_x + p_y*p_y);

    // Ensure no division by zero when setting sigma values in measurement space
    Zsig(1, i) = atan2(p_y, p_x);
    if (rho > 0.001) {
      Zsig(0, i) = rho;
      Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);
    } else {
      Zsig(0, i) = 0.001;
      Zsig(2, i) = (p_x*v1 + p_y*v2) / 0.001;
    }
  }

  // calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  // calculate measurement covariance matrix S
  MatrixXd R(n_z, n_z);
  R.fill(0.0);
  R(0, 0) = std_radr_*std_radr_;
  R(1, 1) = std_radphi_*std_radphi_;
  R(2, 2) = std_radrd_*std_radrd_;

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd diff(n_z);
    diff = Zsig.col(i) - z_pred;

    // Normalize phi between -pi and pi
    while (diff(1) < -M_PI) {
      diff(1) += 2*M_PI;
    }
    while (diff(1) > M_PI) {
      diff(1) -= 2*M_PI;
    }

    S = S + weights_(i)*(diff*diff.transpose());
  }

  S = S + R;

  // calculate cross correlation matrix
  Tc.fill(0.0);
  int n_sig = 2*n_aug_ + 1;
  for (int i = 0; i < n_sig; i++) {
    // State difference and angle normalization
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2*M_PI;
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2*M_PI;

    // Residual and angle normalization
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2*M_PI;
    while (x_diff(1) > M_PI)
      z_diff(1) -= 2*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K(n_x_, n_z);
  K.fill(0.0);
  K = Tc * S.inverse();

  // Residual and angle normalization
  VectorXd z_diff = z - z_pred;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2*M_PI;
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS for radar
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
