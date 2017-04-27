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

#include <cmath>

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if ( num_particles == 0) num_particles = 400;

	particles.clear();
	particles.resize(num_particles);

	weights.clear();
	weights.resize(num_particles);

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

		/*std::cout 	<< "Particle: "
					<< particles[idx].id
					<< " "
					<< particles[idx].x
					<< " "
					<< particles[idx].y
					<< " "
					<< particles[idx].theta
					<< " "
					<< particles[idx].weight
					<< std::endl;*/

		weights[idx] = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;

	// create distributions for all vars around particle to add noise later on
	std::normal_distribution<double> dist_x	 (0, std_pos[0]);
	std::normal_distribution<double> dist_y	 (0, std_pos[1]);
	std::normal_distribution<double> dist_psi(0, std_pos[2]);

	// Yay C++11 !, if you are using the distributions this is needed!
	int particle_size = particles.size();
	for ( int idx = 0; idx < particle_size; idx++ )
	{
		//double x = particle.x;
		//double y = particle.y;
		double theta = particles[idx].theta;

		// Predict particle movement
		if ( abs(yaw_rate) < 1e-3 )
		{
			particles[idx].x 		+= velocity * delta_t * cos(theta);
			particles[idx].y 		+= velocity * delta_t * sin(theta);
			particles[idx].theta 	+= yaw_rate * delta_t;
		}
		else
		{
			double vdivyawrate = velocity / yaw_rate;

			particles[idx].x 		+= vdivyawrate * ( sin(theta + yaw_rate * delta_t) - sin(theta) );
			particles[idx].y	 	+= vdivyawrate * ( cos(theta) - cos(theta + yaw_rate * delta_t) );
			particles[idx].theta 	+= yaw_rate * delta_t;
		}

		// add random gaussian noise
		particles[idx].x 		+= dist_x(gen);
		particles[idx].y	 	+= dist_y(gen);
		particles[idx].theta 	+= dist_psi(gen);

		/*std::cout 	<< "Particle "
					<< particle.id
					<< " "
					<< particle.x
					<< " "
					<< particle.y
					<< " "
					<< particle.theta
					<< " "
					<< particle.weight
					<< std::endl;*/

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

	int meas_size = observations.size();
	for ( int idx = 0; idx < meas_size; idx++ )
	{
		std::vector<double> distances;
		distances.clear();

		for ( auto landmark : predicted )
		{
			double x_diff = observations[idx].x - landmark.x;
			double y_diff = observations[idx].y - landmark.y;
			// We dont even care about the square root
			double distance = sqrt(std::pow(x_diff, 2) + std::pow(y_diff, 2));
			distances.push_back(distance);
		}
		// Find smallest distance by sorting
		auto min_dist = std::min_element(distances.begin(), distances.end());
		int  index    = std::distance(distances.begin(), min_dist);

		// And assign found best match predicted landmark to meas
		// Should we do this? Or just return the index on the likely match?
		//meas.id = predicted[index].id;
		observations[idx].id = index;

		// We could pop the landmark maybe? Complexity would be halved. I dunno.
	}
}

/*
* @param sensor_range Range [m] of sensor
* @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
*   standard deviation of bearing [rad]]
* @param observations Vector of landmark observations
* @param map Map class containing map landmarks
*/
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

	// I do not like the map coordsys, I would rather work always in the veh coord system, if only because, by
	// doing so, I can transform directly to the expected range and bearing, and so get the errors directly
	// To do all this, we will also need to associate the landmarks in the map (the predicted landmarks, as
	// suggested above, means that we want this association to be done in the veh sys) with the observations
	// in the veh sys.

	// So, at the beginning, ALL particles reside in the map system?
	// And all landmarks also reside in the map system?
	// And each vehicle has its own system given by x, y, psi

	// Get all landmarks within sensor range, for each particle

	for ( int idx = 0; idx < num_particles; idx++ )
	{
		Particle particle = particles[idx];

		double x_p  = particle.x;
		double y_p  = particle.y;
		double a_p  = particle.theta;
		//double id_p = particle.id;

		std::vector<LandmarkObs> range_landmarks;

		int size_map = map_landmarks.landmark_list.size();
		for ( auto landmark : map_landmarks.landmark_list )
		{
			double x_l = landmark.x_f;
			double y_l = landmark.y_f;
			int   id_l = landmark.id_i;

			// check if in range and push
			double distance = std::sqrt( std::pow(x_p - x_l, 2) + std::pow(y_p - y_l, 2) );
			if (distance < sensor_range)
			{
				LandmarkObs range_landmark = {id_l, x_l, y_l};
				range_landmarks.push_back(range_landmark);
			}
		}

		// Now that we have our list of landmarks in range for this particle, we must transform them all
		// to the system of each vehicle, as observations are in that system, and we will be able to associate
		// in there and get the weight. We suppose landmarks have 0 range

		// We dont have eigen?! Really?
		int size_ranged = range_landmarks.size();
		for ( int idx = 0; idx < size_ranged; idx++ )
		{
			// Transform all landmarks to vehicle pose, this will probably need adjustment
			double x_l_m = range_landmarks[idx].x;
			double y_l_m = range_landmarks[idx].y;
			// We have the vehicle pose up at the beginning
			// please note the sign inversions due to translation representation
			// TODO: check all signs, check this conversion!!!
			range_landmarks[idx].x = (x_l_m - x_p) * cos(a_p) + (y_l_m - y_p) * sin(a_p);
			range_landmarks[idx].y = (y_l_m - y_p) * cos(a_p) - (x_l_m - x_p) * sin(a_p);
		}

		// Now all our landmarks in the expected range are in the system of the particle (vehicle)
		// All observations are also in this system
		// Since we are using NN, as long as we have a map and observations all items will be
		// matched, maybe very badly!
		// This will set the id in observations to the matched index of the landmarks, so observations
		// will be modified by reference
		dataAssociation(range_landmarks, observations);

		// Calculate the error now
		double new_weight = 1;

		int size_obs = observations.size();
		for ( auto observation : observations )
		{
			// Observations and landmarks are still in x,y form. We will have to transform to
			// expected range and bearing here

			// First get the best landmark for the association, the index of the best landmark
			// match should already be contained in the observation
			// Note that we share types for the predicted landmarks, but this is only for
			// convenience
			auto matched_landmark = range_landmarks[observation.id];

			double observed_range    = std::sqrt( std::pow(observation.x, 2) + std::pow(observation.y, 2) );
			double observed_bearing  = atan2(observation.y, observation.x);
			double predicted_range   = std::sqrt( std::pow(matched_landmark.x, 2) + std::pow(matched_landmark.y, 2) );
			double predicted_bearing = atan2(matched_landmark.y, matched_landmark.x);

			double std_range   = std_landmark[0];
			double std_bearing = std_landmark[1];

			// Ah, I miss eigen!
			double diff_range = observed_range - predicted_range;
			double diff_bearing = observed_bearing - predicted_bearing;

			while (diff_bearing >  M_PI) diff_bearing -= 2 * M_PI;
			while (diff_bearing < -M_PI) diff_bearing += 2 * M_PI;

			// Calculate the value then
			double a   = - 0.5 * ( std::pow(diff_range, 2) / (std::pow(std_range, 2)) + std::pow(diff_bearing, 2) / (std::pow(std_bearing, 2)) );
			double b   = std::sqrt( 2 * M_PI * (std_range * std_bearing ) );
			double w_i = exp(a) / b;

			new_weight *= (w_i);
		}

		// Assign new weight to particle
		particles[idx].weight = (new_weight);
		weights[idx] = (new_weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
	std::vector<Particle> new_particles;
	new_particles.clear();
	std::discrete_distribution<> sampler(weights.begin(), weights.end());

	for (int idx = 0; idx < num_particles; idx++)
	{
		int chosen_index = sampler(gen);
		new_particles.push_back(particles[chosen_index]);
	}

	particles = new_particles;
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
