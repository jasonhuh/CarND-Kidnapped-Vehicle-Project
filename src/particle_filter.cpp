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
#include <random>

#include "particle_filter.h"

ParticleNoise ParticleNoiseGenerator::GenerateNoise() {
	return {N_x_init_(gen_), N_y_init_(gen_), N_theta_init_(gen_)};
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	particle_id_gen = 1;
	Particle p = {particle_id_gen, x, y, theta, 1.0};
	particles.emplace_back(p);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Calculate prediction
	Particle p0 = particles.back(), p;
	double vy = velocity/yaw_rate, yd = yaw_rate * delta_t;
	p.id = ++particle_id_gen;
	p.x = p0.x + vy * (sin(p.theta + yd) - sin(p0.x));
	p.y = p0.y + vy * (cos(p.theta) - cos(p.theta + yd));
	p.theta = p0.theta + yd;

	// Add noise
	ParticleNoise noise = noise_gen_.GenerateNoise();
	p.x *= noise.x;
	p.y *= noise.y;
	p.theta *= noise.theta;

	particles.emplace_back(p);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    //TODO: Improve the algorithm as the current implementation is O(M*N)
	for (auto ob : observations) {
		auto min_idx = 0;
		auto min_dist = std::numeric_limits<double>::max();
		for (auto i = 0; i < predicted.size(); ++i) {
			auto d = dist(ob.x, ob.y, predicted[i].x, predicted[i].y);
			if (d < min_dist) {
				min_dist = d;
				min_idx = i;
			}
		}
		ob.id = min_idx;
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
