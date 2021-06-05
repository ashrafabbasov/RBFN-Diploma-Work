#include<iostream>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iomanip>
#include<fstream>
#include<sstream>
#include<algorithm>
using namespace std;

int N = 3;
double W = 1;
double c1 = 2;
double c2 = 2;
double weight_max = 100;
double weight_min = -100;
double c_min = -5;
double c_max = 5;
double sigma_max = 1.2;
double sigma_min = 0.05;
double target_error = 1;
const double PI = 3.141592653589793238463;
vector<vector<double>> X_train;
vector<vector<double>> X_test;
vector<double> Y_actual;
vector<double> Y_predicted;
vector<double> Y_test;
vector<double> growing_parameters;
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
vector<T> sumOfTwoArrays(vector<T> arr1, vector<T> arr2) {
	if (arr1.size() != arr2.size())
	{
		cout << "Arrays are not the same size";
	}
	vector<T> arr(arr1.size(), 0);
	for (size_t i = 0; i < arr1.size(); i++) {
		arr[i] = arr1[i] + arr2[i];
	}
	return arr;
}

template<typename T>
void print_array(vector<T> arr)
{
	for (size_t i = 0; i < arr.size(); i++) {
		cout << arr[i] << ' ';
	}

}




void getYout(vector<double> position_vector, vector<vector<double>>  X_in, int n_data, int n_rbf) {

	for (size_t i = 0; i < n_data; i++)
	{
		double y = 0;
		for (size_t j = 0; j < n_rbf; j++)
		{
			vector<double> a(position_vector.begin() + 10 * j, position_vector.begin() + 10 * j + 8);
			double dist = distance(X_in[i], a);
			y = y + position_vector[10 * j + 9] * exp(-1 * pow(dist, 2) / (2 * pow(position_vector[10 * j + 8], 2)));
			//cout << y<<" ";
		}
		//cout << endl;
		Y_predicted.push_back(y);

	}

}

double RbfFitness(vector<double> position_vector, vector<vector<double>>  X_in, vector<double> Y_actual, int n_data, int n_rbf) {
	double err = 0;
	vector<double> all_params = growing_parameters;
	all_params.insert(all_params.end(), position_vector.begin(), position_vector.end());
	for (size_t i = 0; i < n_data; i++)
	{
		double y = 0;
		for (size_t j = 0; j < n_rbf; j++)
		{
			vector<double> a(all_params.begin() + 10 * j, all_params.begin() + 10 * j + 8);

			double dist = distance(X_in[i], a);

			y = y + all_params[10 * j + 9] * exp(-1 * pow(dist, 2) / (2 * pow(all_params[10 * j + 8], 2)));

		}
	/*	cout << y << "  ";
		cout << endl;*/
		err = err + pow((Y_actual[i] - y), 2);
	}
	return sqrt(err/n_data);
}

vector<double> ApplyConstraints(vector<double> pos)
{
	vector<double> a = pos;
	for (size_t i = 0; i < 1; i++)
	{
		if (a[10 * i] > c_max)
		{
			a[10 * i] = c_max;
		}
		if (a[10 * i] < c_min)
		{
			a[10 * i] = c_min;
		}
		if (a[10 * i + 1] > c_max)
		{
			a[10 * i + 1] = c_max;
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
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back((double)rand() / RAND_MAX);
		position.push_back(0.05 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (1.2 - 0.05))));
		position.push_back(rand() % 11 - 5.0);
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
		pbest_position = position;
	
	}
	void print_particle() {
		cout << "I am at ";
		print_array(position);
		cout << endl;
		cout << "my pbest is ";
		print_array(pbest_position);
		cout << endl;
	}
	void move() {
		position = sumOfTwoArrays(position, velocity);
	}
};

class Space {
public:
	double target;
	
	int n_particles;
	vector<Particle> particles;
	double gbest_error = 1e6;
	vector<double> gbest_position;
	Space(double targ,int n_part) {
		target = targ;
		n_particles = n_part;
		/*	for (size_t i = 0; i < N; i++)
			{
				gbest_position.push_back(rand()%3 - 1.0);
				gbest_position.push_back(rand()%3 - 1.0);
				gbest_position.push_back(0.05 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (1.2 - 0.05))));
				gbest_position.push_back(rand()%41 - 20.0);

			}*/
	}
	void print_particles() {
		for (size_t i = 0; i < particles.size(); i++)
		{
			particles[i].print_particle();
		}
	}
	double fitness(Particle a) {
		/*return pow(a.position[0], 2) + pow(a.position[1], 2)+1; pow(a.position[0], 2) - 10 * cos(2 * PI * a.position[0]) + pow(a.position[1], 2) - 10 * cos(2 * PI * a.position[1]) + 20;*/
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
			double best_fitness_candidate = fitness(particles[i]);
			if (gbest_error > best_fitness_candidate)
			{
				gbest_error = best_fitness_candidate;
				gbest_position = particles[i].position;
			}
		}
	}

	void move_particles() {
		for (size_t i = 0; i < particles.size(); i++)
		{
			for (size_t j = 0; j < 10; j++)
			{
				double r1 = (double)rand() / RAND_MAX;
				double r2 = (double)rand() / RAND_MAX;
				particles[i].velocity[j] = W * particles[i].velocity[j] +
					(c1 * r1 * (particles[i].pbest_position[j] - particles[i].position[j])) +
					(c2 * r2 * (gbest_position[j] - particles[i].position[j]));
			}
			
			particles[i].move();
			particles[i].position = ApplyConstraints(particles[i].position);
		}
	}
};

int main() {

	//Initializing time
	time_t start, end;
	time(&start);
	ios_base::sync_with_stdio(false);
	srand(time(NULL));


	string x1, x2, x3, x4, x5, x6, x7, x8, y; //variables from file are here
	//input filename
	string file;
	cout << "Enter the filename: ";
	cin >> file;

	ifstream training(file); //opening the file which contains training data.
	if (training.is_open()) //if the file is open
	{
		string line;
		getline(training, line);

		while (!training.eof()) //while the end of file is NOT reached
		{
			//8 getlines because of 8 features
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
		training.close(); //closing the file
	}
	else cout << "Unable to open file"; //if the file is not open output

	ifstream test("test_data.txt"); //opening the file which contains test data.
	if (test.is_open()) //if the file is open
	{
		//ignore first line
		string line;
		getline(test, line);

		while (!test.eof()) //while the end of file is NOT reached
		{
			//8 getlines because of 8 features
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
		training.close(); //closing the file
	}
	else cout << "Unable to open file"; //if the file is not open output



	
	for (int i = 0; i < 10; i++)
	{
		Space search_space(0, 15);
		vector<Particle> particles_vector;
		for (int j = 0; j < 15; j++)
		{
			Particle a;
			particles_vector.push_back(a);
		}
		search_space.particles = particles_vector;
		int iteration = 0;
		while (iteration < 100)
		{		
			search_space.set_pbest();
			search_space.set_gbest();	
			error.push_back(search_space.gbest_error);
			W = 0.3 + (1 - 0.4) * exp(-6 * iteration / 100);
			if (search_space.gbest_error <= target_error)
			{
				break;
			}	
			search_space.move_particles();
			iteration++;	
		}
		growing_parameters.insert(growing_parameters.end(), search_space.gbest_position.begin(), search_space.gbest_position.end());
		N++;
		// print_array(growing_parameters);
		// cout << endl;
		// cout << "in " << iteration << " iterations" << endl;
		 cout << "error: " << search_space.gbest_error<<endl;
	}
	
	
	getYout(growing_parameters, X_test, X_test.size(), N-1);
	cout << endl;
	print_array(growing_parameters);

	ofstream outfile("predicted.txt");
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
