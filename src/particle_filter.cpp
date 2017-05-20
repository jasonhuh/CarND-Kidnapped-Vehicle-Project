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

const static int NUM_PARTICLES = 300;

/**
 * Generate random noise based on particle's position
 */
class ParticleNoiseGenerator {
private:
    std::default_random_engine gen_;
    std::normal_distribution<double> dist_x_;
    std::normal_distribution<double> dist_y_;
    std::normal_distribution<double> dist_theta_;
public:
    struct ParticleNoise {
        double x, y, theta;
    };

    ParticleNoiseGenerator(double std_pos[], double x, double y, double theta) {
        dist_x_ = std::normal_distribution<double>(x, std_pos[0]);
        dist_y_ = std::normal_distribution<double>(y, std_pos[1]);
        dist_theta_ = std::normal_distribution<double>(theta, std_pos[2]);
    }

    ParticleNoise GenerateNoise() {
        return {dist_x_(gen_), dist_y_(gen_), dist_theta_(gen_)};
    }
};


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = NUM_PARTICLES;
    particles.resize(num_particles);
    ParticleNoiseGenerator gen(std, x, y, theta);
    weights.resize(num_particles);
    for(auto i = 0; i < num_particles; ++i) {
        auto noise = gen.GenerateNoise();
        Particle p = {i, noise.x, noise.y, noise.theta, 1.0};
        particles[i] = p;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    ParticleNoiseGenerator gen(std_pos, 0.0, 0.0, 0.0);
    const double yaw_dt = yaw_rate * delta_t, v_dt = velocity * delta_t;
    for (auto& p : particles) {
        auto noise = gen.GenerateNoise();
        if(yaw_rate > 1e-5) { // Turning
            const double v_over_yaw = velocity / yaw_rate;
            p.x += v_over_yaw * (sin(p.theta + yaw_dt) - sin(p.theta)) + noise.x;
            p.y += v_over_yaw * (cos(p.theta) - cos(p.theta + yaw_dt)) + noise.y;
        } else { // Going straight
            p.x += v_dt * cos(p.theta) + noise.x;
            p.y += v_dt * sin(p.theta) + noise.y;
        }
        p.theta += yaw_dt + noise.theta;
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
    auto weight_idx = 0;
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
        weights[weight_idx++] = weight;
    }
}

void ParticleFilter::resample() {
    std::default_random_engine gen;
    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    std::vector<Particle> res(particles.size());
    for (auto i = 0; i < particles.size(); ++i) {
        auto idx = distribution(gen);
        res[i] = particles[idx];
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
