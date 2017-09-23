#include "Catch/catch.hpp"
#include "tools.h"

TEST_CASE( "Root-Mean-Square Error", "[tools]" ) {
    vector<VectorXd> estimations;
    vector<VectorXd> ground_truth;

    //the input list of estimations
    VectorXd e(4);
    e << 1, 1, 0.2, 0.1;
    estimations.push_back(e);
    e << 2, 2, 0.3, 0.2;
    estimations.push_back(e);
    e << 3, 3, 0.4, 0.3;
    estimations.push_back(e);

    //the corresponding list of ground truth values
    VectorXd g(4);
    g << 1.1, 1.1, 0.3, 0.2;
    ground_truth.push_back(g);
    g << 2.1, 2.1, 0.4, 0.3;
    ground_truth.push_back(g);
    g << 3.1, 3.1, 0.5, 0.4;
    ground_truth.push_back(g);

    Tools tools = Tools();
    VectorXd rmse = tools.CalculateRMSE(estimations, ground_truth);

    REQUIRE( rmse.size() == 4 );
    REQUIRE( rmse(0) == Approx(0.1).epsilon(0.00001) );
    REQUIRE( rmse(1) == Approx(0.1).epsilon(0.00001) );
    REQUIRE( rmse(2) == Approx(0.1).epsilon(0.00001) );
    REQUIRE( rmse(3) == Approx(0.1).epsilon(0.00001) );
}

TEST_CASE( "Jacobian", "[tools]" ) {
  VectorXd state(4);
  state << 1, 2, 0.2, 0.4;

  Tools tools = Tools();

  MatrixXd jacobian = tools.CalculateJacobian(state);
  REQUIRE( jacobian.rows() == 3 );
  REQUIRE( jacobian.cols() == 4 );

  REQUIRE( jacobian(0, 0) == Approx(0.447214).epsilon(0.000001) );
  REQUIRE( jacobian(0, 1) == Approx(0.894427).epsilon(0.000001) );
  REQUIRE( jacobian(0, 2) == Approx(0.0).epsilon(0.000001) );
  REQUIRE( jacobian(0, 3) == Approx(0.0).epsilon(0.000001) );

  REQUIRE( jacobian(1, 0) == Approx(-0.4).epsilon(0.000001) );
  REQUIRE( jacobian(1, 1) == Approx(0.2).epsilon(0.000001) );
  REQUIRE( jacobian(1, 2) == Approx(0.0).epsilon(0.000001) );
  REQUIRE( jacobian(1, 3) == Approx(0.0).epsilon(0.000001) );

  REQUIRE( jacobian(2, 0) == Approx(0.0).epsilon(0.000001) );
  REQUIRE( jacobian(2, 1) == Approx(0.0).epsilon(0.000001) );
  REQUIRE( jacobian(2, 2) == Approx(0.447214).epsilon(0.000001) );
  REQUIRE( jacobian(2, 3) == Approx(0.894427).epsilon(0.000001) );
}

