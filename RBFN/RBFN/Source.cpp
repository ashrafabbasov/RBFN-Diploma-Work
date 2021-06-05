#include<iostream>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iomanip>
#include<fstream>
#include<sstream>
using namespace std;

int N = 10;
int num_of_particles = 15;
int num_of_iterations = 1000;
double W = 1;
double c1 = 2;
double c2 = 2;
double weight_max =100;
double weight_min = -100;
double c_min = -5;
double c_max = 5;
double sigma_max = 1.2;
double sigma_min = 0.05;
const double PI = 3.141592653589793238463;
vector<vector<double>> X_train;
vector<vector<double>> X_test;
vector<double> Y_actual;
vector<double> Y_predicted;
vector<double> Y_test;
vector<double> error;

double distance(vector<double> x, vector<double> y) {
	double sum = 0;
	for (size_t i = 0; i < x.size(); i++)
	{
		sum = sum + pow(x[i] - y[i], 2);
	}
	return sqrt(sum);
}

template<typename T>
void print_array(vector<T> arr)
{
	for (size_t i = 0; i < arr.size(); i++) {
		cout << arr[i] << ' ';
	}

}

template<typename T>
vector<T> sumOfTwoArrays(vector<T> arr1, vector<T> arr2) {
	if (arr1.size() != arr2.size())
	{
		cout << "Arrays are not the same size";
	}
	vector<T> arr(arr1.size(),0);
	for (size_t i = 0; i < arr1.size(); i++) {
		arr[i] = arr1[i] + arr2[i];
	}
	return arr;
}

void getYout(vector<double> position_vector, vector<vector<double>>  X_in, int n_data, int n_rbf) {

	for (size_t i = 0; i < n_data; i++)
	{
	
		double y = 0;
		for (size_t j = 0; j < n_rbf; j++)
		{
			vector<double> a(position_vector.begin() + 10*j, position_vector.begin() + 10*j+8);
			double dist = distance(X_in[i], a);
			y = y + position_vector[10 * j + 9] * exp(-1 * pow(dist, 2) / (2 * pow(position_vector[10 * j + 8], 2)));
			//cout << y<<" ";
		}
		Y_predicted.push_back(y);
		//cout << endl;
		
	}
	
}

double RbfFitness(vector<double> position_vector,  vector<vector<double>>  X_in, vector<double> Y_actual, int n_data, int n_rbf) {
	double err = 0;
	for (size_t i = 0; i < n_data; i++)
	{
		double y = 0;
		for (size_t j = 0; j < n_rbf; j++)
		{
			vector<double> a(position_vector.begin() + 10 * j, position_vector.begin() + 10 * j + 8);
			
			double dist = distance(X_in[i], a);
			
			y = y + position_vector[10 * j + 9] * exp(-1 * pow(dist, 2) / (2 * pow(position_vector[10 * j + 8], 2)));
			
		}
		err = err + pow((Y_actual[i] - y), 2);
	}
	return sqrt(err/n_data);
}

vector<double> ApplyConstraints(vector<double> pos)
{
	vector<double> a = pos;
	for (size_t i = 0; i < N; i++)
	{
		if (a[10*i] > c_max) 
		{
			a[10*i] = c_max;
		}
		if (a[10*i] < c_min)
		{
			a[10*i] = c_min;
		}
		if (a[10 * i + 1] > c_max)
		{
			a[10 * i + 1 ] = c_max;
		}
		if (a[10 * i + 1] < c_min)
		{
			a[10 * i + 1] = c_min;
		}
		if (a[10 * i + 2] > c_max)
		{
			a[10 * i + 2] = c_max;
		}
		if (a[10 * i + 2] < c_min)
		{
			a[10 * i + 2] = c_min;
		}
		if (a[10 * i + 3] > c_max)
		{
			a[10 * i + 3] = c_max;
		}
		if (a[10 * i + 3] < c_min)
		{
			a[10 * i + 3] = c_min;
		}
		if (a[10 * i + 4] > c_max)
		{
			a[10 * i + 4] = c_max;
		}
		if (a[10 * i + 4] < c_min)
		{
			a[10 * i + 4] = c_min;
		}
		if (a[10 * i + 5] > c_max)
		{
			a[10 * i + 5] = c_max;
		}
		if (a[10 * i + 5] < c_min)
		{
			a[10 * i + 5] = c_min;
		}
		if (a[10 * i + 6] > c_max)
		{
			a[10 * i + 6] = c_max;
		}
		if (a[10 * i + 6] < c_min)
		{
			a[10 * i + 6] = c_min;
		}
		if (a[10 * i + 7] > c_max)
		{
			a[10 * i + 7] = c_max;
		}
		if (a[10 * i + 7] < c_min)
		{
			a[10 * i + 7] = c_min;
		}
		if (a[10 * i + 8] > sigma_max)
		{
			a[10 * i + 8] = sigma_max;
		}
		if (a[10 * i + 8] < sigma_min)
		{
			a[10 * i + 8] = sigma_min;
		}
		if (a[10 * i + 9] > weight_max)
		{
			a[10 * i + 9] = weight_max;
		}
		if (a[10 * i + 9] < weight_min)
		{
			a[10 * i + 9] = weight_min;
		}
	}
	return a;
}

class Particle {
public:
	vector<double> position;
	vector<double> pbest_position;
	double pbest_error = 1e6;
	vector<double> velocity;
	Particle() {
		for (size_t i = 0; i < N; i++)
		{
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back((double)rand() / RAND_MAX);
			position.push_back(sigma_min + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (sigma_max - sigma_min))));
			position.push_back(rand()%11 - 5.0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
			velocity.push_back(0);
		}
		pbest_position = position;
	}	
	void move() {
		position = sumOfTwoArrays(position, velocity);
	}
};

class Space {
public:
	double target;
	vector<Particle> particles;
	double gbest_error = 1e6;
	vector<double> gbest_position;
	Space(double targ) {
		target = targ;
	}
	
	double fitness(Particle a) {
		return RbfFitness(a.position, X_train, Y_actual, Y_actual.size(), N);
	}

	void set_pbest() {
		for (size_t i = 0; i < particles.size(); i++)
		{
			double fitness_candidate = fitness(particles[i]);
			
			if (particles[i].pbest_error > fitness_candidate)
			{
				
				particles[i].pbest_error = fitness_candidate;
				particles[i].pbest_position = particles[i].position;
			}
		}
	}

	void set_gbest() {
		for (size_t i = 0; i < particles.size(); i++)
		{
			if (gbest_error >  particles[i].pbest_error)
			{
				
				gbest_error =  particles[i].pbest_error;
				gbest_position = particles[i].position;
			}
		}
	}

	void move_particles() {
		for (size_t i = 0; i < particles.size(); i++)
		{
			
			for (size_t j = 0; j < 100; j++)
			{
			
				
				double r1 = (double)rand() / RAND_MAX;
				double r2 = (double)rand() / RAND_MAX;
				particles[i].velocity[j] = W*particles[i].velocity[j] + c1 * r1 * (particles[i].pbest_position[j] - particles[i].position[j]) + c2 * r2 * (gbest_position[j]-particles[i].position[j]);
			}
			
			particles[i].move();
			particles[i].position=ApplyConstraints(particles[i].position);
		}
		
	}
	
};

int main() {
	time_t start, end;
	time(&start);
	ios_base::sync_with_stdio(false);
	srand(time(NULL));

	string x1, x2, x3, x4, x5, x6, x7, x8, y;
	string file;
	cout << "Enter the filename: ";
	cin >> file;
	ifstream training(file); 
	if (training.is_open()) 
	{
		
		string line;
		getline(training, line);
		while (!training.eof()) 
		{
			getline(training, x1, ',');
			getline(training, x2, ',');
			getline(training, x3, ',');
			getline(training, x4, ',');
			getline(training, x5, ',');
			getline(training, x6, ',');
			getline(training, x7, ',');
			getline(training, x8, ',');
			X_train.push_back({ stod(x1), stod(x2),stod(x3),stod(x4),stod(x5),stod(x6),stod(x7), stod(x8) });
			getline(training, y, '\n');
			Y_actual.push_back(stod(y));	
		}
		training.close(); 
	}
	else cout << "Unable to open file";

	ifstream test("test_data.txt");
	if (test.is_open()) 
	{
		string line;
		getline(test, line);

		while (!test.eof()) 
		{
			getline(test, x1, ',');
			getline(test, x2, ',');
			getline(test, x3, ',');
			getline(test, x4, ',');
			getline(test, x5, ',');
			getline(test, x6, ',');
			getline(test, x7, ',');
			getline(test, x8, ',');

			X_test.push_back({ stod(x1), stod(x2),stod(x3),stod(x4),stod(x5),stod(x6),stod(x7),stod(x8) });
			getline(test, y, '\n');
			Y_test.push_back(stod(y));
		}
		training.close(); 
	}
	else cout << "Unable to open file"; 


	
	Space search_space(0);
	vector<Particle> particles_vector;
	for (size_t i = 0; i < 15; i++)
	{
		Particle a;
		particles_vector.push_back(a);
	}
	search_space.particles = particles_vector;
	
	int iteration = 0;
	while (iteration < num_of_iterations)
	{
		
		search_space.set_pbest();
		search_space.set_gbest();
		
		W = 0.4 + (1 - 0.4) * exp(-6 * iteration / num_of_iterations);
		error.push_back(search_space.gbest_error);
		if (search_space.gbest_error <= search_space.target)
		{
			break;
		}

		search_space.move_particles();
		
		iteration++;
		
		cout << " " << search_space.gbest_error << endl;
		
	}
	
	print_array(search_space.gbest_position);
	cout << endl;
	cout << "in " << iteration << " iterations"<<endl;
	cout << "error: " << search_space.gbest_error;
	getYout(search_space.gbest_position, X_test, X_test.size(), N);
	cout << endl;
	
	
	ofstream outfile ("predicted.txt");
	if (outfile.is_open())
	{
		outfile << "y_predicted\n";
		for (size_t i = 0; i < Y_predicted.size(); i++)
		{
			outfile << Y_predicted[i];
			outfile << "\n";
		}
	}
	ofstream errorfile ("error.txt");
	if (errorfile.is_open())
	{
		errorfile << "error\n";
		for (size_t i = 0; i < error.size(); i++)
		{
			errorfile << error[i];
			errorfile << "\n";
		}
	}
	// Time taken by program to execute
	time(&end);
	double time_taken = double(end - start);
	cout << "Time taken by program is : " << fixed
		<< time_taken << setprecision(5);
	cout << " sec " << endl;
	
	return 0;
}
