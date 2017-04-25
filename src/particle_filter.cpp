/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// We will have to assume that the number of particles was set beforehand, and not by us, as this is
	// not part of the constructor!

	particles.clear();
	particles.resize(num_particles);

	std::default_random_engine gen;

	// create distributions for all vars around first estimate
	std::normal_distribution<double> dist_x	(x, 	std[0]);
	std::normal_distribution<double> dist_y	(y, 	std[1]);
	std::normal_distribution<double> dist_psi(theta, std[2]);

	// and create all the particles, with uniform weights
	// Not using auto because we want to generate the index!
	for( int idx = 0; idx < num_particles; idx++ )
	{
		particles[idx].id		= idx;
		particles[idx].x 		= dist_x(gen);
		particles[idx].y 		= dist_y(gen);
		particles[idx].theta	= dist_psi(gen);
		particles[idx].weight   = 1.0;	// all particles start with the same weight, should it be so?

		std::cout 	<< "Particle: "
					<< particles[idx].id
					<< " "
					<< particles[idx].x
					<< " "
					<< particles[idx].y
					<< " "
					<< particles[idx].theta
					<< " "
					<< particles[idx].weight
					<< std::endl;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;

	// Yay C++11 !, if you are using the distributions this is needed!
	for ( auto particle : particles )
	{
		double x = particle.x;
		double y = particle.y;
		double theta = particle.theta;

		// create distributions for all vars around particle to add noise later on
		std::normal_distribution<double> dist_x	(x, 	 std_pos[0]);
		std::normal_distribution<double> dist_y	(y, 	 std_pos[1]);
		std::normal_distribution<double> dist_psi(theta, std_pos[2]);

		// Predict particle movement
		if ( abs(velocity) < 1e-3 )
		{
			particle.x += velocity * delta_t * cos(theta);
			particle.y += velocity * delta_t * sin(theta);
		}
		else
		{
			double vdivyawrate = velocity / yaw_rate;

			particle.x 		+= vdivyawrate * ( sin(theta + yaw_rate * delta_t) - sin(theta) );
			particle.y	 	+= vdivyawrate * ( cos(theta) - cos(theta + yaw_rate * delta_t) );
			particle.theta 	+= theta + yaw_rate;
		}

		// add random gaussian noise
		particle.x 		+= dist_x(gen);
		particle.y	 	+= dist_y(gen);
		particle.theta 	+= dist_psi(gen);

		std::cout 	<< "Particle "
					<< particle.id
					<< " "
					<< particle.x
					<< " "
					<< particle.y
					<< " "
					<< particle.theta
					<< " "
					<< particle.weight
					<< std::endl;

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// We will use NN over euclidian distances, this will be O(n^2) in complexity
	// We assume that it is the predicted landmarks (ultimately, from the map) that have ids
	// We assume that measurements do not have any assignment and we want to find it here
	// We assume that there is always at least a match?
	// We assume that we can modify the measurements by reference, signature suggest so too
	if ( predicted.size() == 0 || observations.size() == 0 ) return;

	for ( auto meas : observations )
	{
		std::vector<double> distances;
		for ( auto landmark : predicted )
		{
			double x_diff = meas.x - landmark.x;
			double y_diff = meas.y - landmark.y;
			// We dont even care about the square root
			double distance2 = x_diff * x_diff + y_diff * y_diff;
			distances.push_back(distance2);
		}
		// Find smallest distance by sorting
		auto min_dist = std::min_element(distances.begin(), distances.end());
		int  index    = std::distance(distances.begin(), min_dist);

		// And assign found best match predicted landmark to meas
		meas.id = predicted[index].id;

		// We could pop the landmark maybe? Complexity would be halved. I dunno.
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
