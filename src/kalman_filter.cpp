#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Convert the last data from the cartesian to polar coordinates
  const double last_x = x_(0);
  const double last_y = x_(1);
  const double last_vx = x_(2);
  const double last_vy = x_(3);

  double rho = sqrt(last_x*last_x + last_y*last_y);
  const double theta = atan2(last_y, last_x);
  const double ro_dot = (last_x * last_vx + last_y * last_vy) / rho;
  VectorXd z_pred = VectorXd(3);
  z_pred << rho, theta, ro_dot;

  VectorXd y = z - z_pred;

  y(1) = NormalizeAngle(y(1));

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

double KalmanFilter::NormalizeAngle(const double angle) {
  if (angle > two_pi_) {
    return angle - two_pi_;
  } else if (angle < -two_pi_) {
    return angle + two_pi_;
  } else {
    return angle;
  }
}
