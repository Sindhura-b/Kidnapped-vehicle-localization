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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	default_random_engine gen;

	num_particles=100;
	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);
	weights.resize(num_particles);
	particles.resize(num_particles);	
	
	for(int i=0;i<num_particles;i++){
	   particles[i].id=i;
	   particles[i].x=dist_x(gen);
	   particles[i].y=dist_y(gen);
	   particles[i].theta=dist_theta(gen);
	   particles[i].weight=1.0;
        }
        is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen;	
	normal_distribution<double> dist_x(0.0,std_pos[0]);
	normal_distribution<double> dist_y(0.0,std_pos[1]);
	normal_distribution<double> dist_theta(0.0,std_pos[2]);

	for(int i=0;i<num_particles;i++){
        if(yaw_rate!=0){
	    particles[i].x+=(velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            particles[i].y+=(velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
	    particles[i].theta+=yaw_rate*delta_t;
	}
	else{
	    particles[i].x+=velocity*cos(particles[i].theta)*delta_t;
            particles[i].y+=velocity*sin(particles[i].theta)*delta_t;
            particles[i].theta+=yaw_rate*delta_t;
	}
	    particles[i].x+=dist_x(gen);
	    particles[i].y+=dist_y(gen);
	    particles[i].theta+=dist_theta(gen);
        }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for(int i=0;i<num_particles;i++){
	    std::vector<LandmarkObs> obs_trans;
	    double Mult_gaussian=1.0;
	    for(int j=0;j<observations.size();j++){
		LandmarkObs obs_t;
	        obs_t.x=particles[i].x+(cos(particles[i].theta)*observations[j].x-sin(particles[i].theta)*observations[j].y);
	        obs_t.y=particles[i].y+(sin(particles[i].theta)*observations[j].x+cos(particles[i].theta)*observations[j].y);
                obs_trans.push_back(obs_t);
            
	        vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
	        vector<double> dist_est;
		double dst=0.0;
                for(int k=0;k<landmarks.size();k++){
		   double landmark_dist_range=dist(landmarks[k].x_f,landmarks[k].y_f,particles[i].x,particles[i].y);
		   if(landmark_dist_range<=sensor_range){
		       dst=dist(landmarks[k].x_f,landmarks[k].y_f,obs_t.x,obs_t.y);
		   }
		   else{
		       dst=999999.0;			
		   }
		   dist_est.push_back(dst);
	        }
	        auto result=min_element(begin(dist_est),end(dist_est));
	        int pos=distance(begin(dist_est),result);
	        obs_t.id=landmarks[pos].id_i;
                double lanmark_x=landmarks[pos].x_f;
		double lanmark_y= landmarks[pos].y_f;
                double x_diff=lanmark_x-obs_t.x;
        	double y_diff=lanmark_y-obs_t.y;
		double power=((x_diff*x_diff)/(2*pow(std_landmark[0],2))+((y_diff*y_diff)/(2*pow(std_landmark[0],2))));
                Mult_gaussian*=exp(-power)/(2*M_PI*std_landmark[0]*std_landmark[1]);
	    }
	    particles[i].weight=Mult_gaussian;
	    weights[i]=particles[i].weight;
	}
}

void ParticleFilter::resample() {

	random_device rd;
	mt19937 gen(rd());	
	discrete_distribution<int> d(weights.begin(),weights.end());
	vector<Particle> particles_new;
	for(int i=0;i<num_particles;i++){
	    particles_new.push_back(particles[d(gen)]);
	}
	particles=particles_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
