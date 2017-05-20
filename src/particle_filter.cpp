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

const static int NUM_PARTICLES = 100;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    //TODO: Refactor
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = NUM_PARTICLES;
    particles.resize(num_particles);
    weights.resize(num_particles, 1.0);
    for(auto i = 0; i < num_particles; ++i) {
        Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
        particles[i] = p;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	// Calculate prediction
    const double yaw_dt = yaw_rate * delta_t, v_dt = velocity * delta_t;

    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    for (auto& p : particles) {
        if(yaw_rate > 1e-5) {
            const double v_over_yaw = velocity / yaw_rate;
            p.x += v_over_yaw * (sin(p.theta + yaw_dt) - sin(p.theta)) + dist_x(gen);
            p.y += v_over_yaw * (cos(p.theta) - cos(p.theta + yaw_dt)) + dist_y(gen);
        } else {
            p.x += v_dt * cos(p.theta) + dist_x(gen);
            p.y += v_dt * sin(p.theta) + dist_y(gen);
        }
        p.theta += yaw_dt + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for (auto& ob : observations) {
		auto min_idx = -1;
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
    weights.clear();
    for (auto& p : particles) {
        std::vector<LandmarkObs> predictions;

        // Find landmarks that particle can reach
        for (auto lm : map_landmarks.landmark_list) {
            if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range) {
                LandmarkObs ob_pred = {lm.id_i, lm.x_f, lm.y_f};
                predictions.emplace_back(ob_pred);
            }
        }

        // Transform from vehicle frame to map frame
        std::vector<LandmarkObs> obvervations_f;
        for (auto ob : observations) {
            auto x_final = ob.x * cos(p.theta) - ob.y * sin(p.theta) + p.x;
            auto y_final = ob.x * sin(p.theta) + ob.y * cos(p.theta) + p.y;
            LandmarkObs ob_final = {ob.id, x_final, y_final};
            obvervations_f.emplace_back(ob_final);
        }

        dataAssociation(predictions, obvervations_f);

        // Calculate weights
        const auto gaussian = 0.5 / (M_PI * std_landmark[0] * std_landmark[1]);
        auto weight = 1.0;
        // Multivariate-Guassian probability
        for (const auto ob : obvervations_f) {
            auto error_x = ob.x - predictions[ob.id].x;
            auto error_y = ob.y - predictions[ob.id].y;
            auto x_nom = error_x*error_x/(std_landmark[0]*std_landmark[0]);
            auto y_nom = error_y*error_y/(std_landmark[1]*std_landmark[1]);
            weight *= gaussian * exp(-0.5 * (x_nom + y_nom));;
        }

        // Update weight
        p.weight = weight;
        weights.emplace_back(weight);
    }
}

void ParticleFilter::resample() {
    std::default_random_engine gen;
    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    std::vector<Particle> res;
    for (auto i = 0; i < num_particles; ++i) {
        auto idx = distribution(gen);
        res.emplace_back(particles[idx]);
    }
    particles = res;
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
