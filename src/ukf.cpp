#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
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
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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

  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  // System states, x coordinate, y coordinate, heading speed, yaw, yaw rate
  n_x_ = 5;
  // System noise, heading acceleration, yaw acceleration

  // System augmented states
  n_aug_ = 7;

  // Lambda initialization
  lambda_ = 3 - n_aug_;

  // weights for all sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  // weights initialization
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) 
  {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // Model covariance initialization
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // States sigma points initialization
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Augmented state sigma points matrix initialization
  Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  ///* Radar measurement dimension
  n_z_radar_ = 3;

  ///* Lidar measurement dimension
  n_z_lidar_ = 2;

  // Radar sigma point in measurement space
  MatrixXd Zsig_radar_ = MatrixXd::Zero(n_z_radar_, 2 * n_aug_ + 1);

  // Lidar sigma point in measurement space
  MatrixXd Zsig_lidar_ = MatrixXd::Zero(n_z_lidar_, 2 * n_aug_ + 1);

  // Measured mean states for radar
  VectorXd z_pred_radar_ = VectorXd::Zero(n_z_radar_);

  // Measured mean states for lidar
  VectorXd z_pred_lidar_ = VectorXd::Zero(n_z_lidar_);

  // Radar innovation covariance matrix
  MatrixXd S_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  S_radar_.fill(0.0);

  // Lidar innovation covariance matrix
  MatrixXd S_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  S_lidar_.fill(0.0);

  // time initialization
  previous_time = 0.0;
  delta_t = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  // Calculate delta time in second
  delta_t = (meas_package.timestamp_ - previous_time) / 1000000.0;
  previous_time = meas_package.timestamp_;

  // Initializing once
  if (!is_initialized_)
  {
  	// initializing from the first measurement
  	if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  	{
  		// Convert from polar to cartesian coordinates and initialize system states
  		double rho = meas_package.raw_measurements_[0];
  		double phi = meas_package.raw_measurements_[1];
  		double rho_dot = meas_package.raw_measurements_[2];
  		x_(0) = rho * cos(phi);
  		x_(1) = rho * sin(phi);
  		x_(2) = rho_dot;
  		x_(3) = phi;
  		x_(4) = 0.0;
  	}
  	else
  	{
  		// Initialize system states
  		x_(0) = meas_package.raw_measurements_[0];
  		x_(1) = meas_package.raw_measurements_[1];
  		x_(2) = 0.0;
  		x_(3) = 0.0;
  		x_(4) = 0.0;
  	}
  	is_initialized_ = true;
  	return;
  }
  // perform UKF prediction
  Prediction();
  // perform UKF update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  	UpdateRadar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  	UpdateLidar(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction() {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  // generate sigma point and perform augmentation, Xsig_aug_ will be updated
  AugmentedSigmaPoints();
  // perform prediction for augmented sigma points
  SigmaPointPrediction();
  // perform prediction for mean and covariance of sigma points
  // Xsig_pred_, x_, and P_ will be updated
  PredictMeanAndCovariance();
}

void UKF::AugmentedSigmaPoints() {
  /**
   * Generate sigma points for augmented states
   * @param augmented sigma points at k
   */
  //set augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  //set augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  //set sigma point placeholder
  Xsig_aug_.setZero();
  //create augmented mean states
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;
  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_ * std_a_, 0.0,
       0.0, std_yawdd_ * std_yawdd_;
  P_aug.bottomRightCorner(2,2) = Q;
  //create square root matrix
  MatrixXd S = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
  	Xsig_aug_.col(i+1) = x_aug + sqrt(n_aug_ + lambda_) * S.col(i);
  	Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(n_aug_ + lambda_) * S.col(i);
  }
}

void UKF::SigmaPointPrediction() {
  /**
   * Perform process prediction for augmented sigma point
   * @param augmented sigma point at k
   * @param delta time
   * @param predicted augmented sigma point after system process
   */
  //predict sigma points
  for (int i = 0; i < 2*n_aug_+1; i++)
  {
  	//extract values for better readability
  	double px = Xsig_aug_(0,i);
  	double py = Xsig_aug_(1,i);
  	double v = Xsig_aug_(2,i);
  	double yaw = Xsig_aug_(3,i);
  	double yawd = Xsig_aug_(4,i);
  	double nu_a = Xsig_aug_(5,i);
  	double nu_yawdd = Xsig_aug_(6,i);
  	//predicted state value
  	double px_p, py_p;
  	//avoid division by zero
  	if (fabs(yawd) > 0.001) 
  	{
  		px_p = px + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
  		py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
  	}
  	else
  	{
  		px_p = px + v * delta_t * cos(yaw);
  		py_p = py + v * delta_t * sin(yaw);
  	}
  	double v_p = v;
  	double yaw_p = yaw + yawd * delta_t;
  	double yawd_p = yawd;
  	//add noise
  	px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
  	py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;
    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {
  /**
   * Calculate predicted mean and covariance of states
   * @param predicted sigma points at k+1
   * @param predicted mean
   * @param predicted covariance
   */
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  	x_ = x_ + weights_(i) * Xsig_pred_.col(i);

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
  	//state difference
  	VectorXd x_diff = Xsig_pred_.col(i) - x_;
  	//angle normalization
  	while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
  	while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

  	P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //perform Lidar measurement prediction
  PredictLidarMeasurement();
  //obtain actual Lidar measurements
  double lx = meas_package.raw_measurements_[0];
  double ly = meas_package.raw_measurements_[1];
  // read in actual lidar measurement
  VectorXd z_in = VectorXd::Zero(n_z_lidar_);
  z_in << lx, ly;
  //update states and covariance from actual Lidar measurements
  UpdateStateLidar(&z_in);

}

void UKF::PredictLidarMeasurement() {

  /**
   * Perform the prediction of the mean and covariance of Lidar measurements
   */
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_lidar_, 2 * n_aug_ + 1);
  //transform sigma points into lidar measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
  	//extract values for better readibility
  	double p_x = Xsig_pred_(0,i);
  	double p_y = Xsig_pred_(1,i);
  	//measurement model
  	Zsig(0,i) = p_x;
  	Zsig(1,i) = p_y;
  }

  Zsig_lidar_ = Zsig;
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_lidar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  	z_pred = z_pred + weights_(i) * Zsig_lidar_.col(i);

  //innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z_lidar_, n_z_lidar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
  	//residual
  	VectorXd z_diff = Zsig_lidar_.col(i) - z_pred;
  	//angle normalization
  	while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  	while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  	S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_lidar_, n_z_lidar_);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  S = S + R;

  z_pred_lidar_ = z_pred;
  S_lidar_ = S;

}

void UKF::UpdateStateLidar(VectorXd* z_in) {

  /**
   * Perform the update of state and covariance of Lidar measurement
   */
  //create vector for incoming lidar measurement
  VectorXd z = *z_in;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
  	//residual 
  	VectorXd z_diff = Zsig_lidar_.col(i) - z_pred_lidar_;
  	//state difference
  	VectorXd x_diff = Xsig_pred_.col(i) - x_;
  	//angle normalization
  	while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
  	while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

  	Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //kalman gain K
  MatrixXd K = Tc * S_lidar_.inverse();
  //residual
  VectorXd z_diff = z - z_pred_lidar_;
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
  while (x_(3) < -M_PI) x_(3) += 2. * M_PI;

  P_ = P_ - K * S_lidar_ * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //perform Radar measurement prediction
  PredictRadarMeasurement();
  //obtain actual Radar measurements
  double rho = meas_package.raw_measurements_[0];
  double phi = meas_package.raw_measurements_[1];
  double rho_dot = meas_package.raw_measurements_[2];
  // read in radar actual measurement
  VectorXd z_in = VectorXd::Zero(n_z_radar_);
  z_in << rho, phi, rho_dot;
  //update states and covariance from actual Radar measurements
  UpdateStateRadar(&z_in);

}

void UKF::PredictRadarMeasurement() {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, 2 * n_aug_ + 1);
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    if(hypot(p_x, p_y) > 0.001)
    {
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
      Zsig(1,i) = atan2(p_y,p_x);                                 //phi
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
  }
  
  Zsig_radar_ = Zsig;
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred = z_pred + weights_(i) * Zsig_radar_.col(i);
  
  //innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    //residual
    VectorXd z_diff = Zsig_radar_.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);
  R <<    std_radr_ * std_radr_, 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0,std_radrd_ * std_radrd_;
  S = S + R;

  z_pred_radar_ = z_pred;
  S_radar_ = S;
}
void UKF::UpdateStateRadar(VectorXd* z_in) {

  //create  vector for incoming radar measurement
  VectorXd z = *z_in;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_radar_.col(i) - z_pred_radar_;
    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //Kalman gain
  MatrixXd K = Tc * S_radar_.inverse();
  //residual
  VectorXd z_diff = z - z_pred_radar_;
  //angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
  while (x_(3) < -M_PI) x_(3) += 2. * M_PI;

  P_ = P_ - K * S_radar_ * K.transpose();
}

