#include <glm/glm.hpp>
#include <ctime>
#include <vector>

using namespace glm;

constexpr int res = 128 * 4;
constexpr int node_res = 1;
constexpr int elem_func_size_1D = 5;
constexpr int elem_func_size = elem_func_size_1D* elem_func_size_1D;

#define SQR(x) ((x)*(x))
std::vector<std::vector<vec2>> velocities(res, std::vector<vec2>(res, glm::vec2(0)));
std::vector<std::vector<vec2>> node_velocities(res, std::vector<vec2>(res, glm::vec2(0)));

float fit_step_size = 6e-7f;
float sim_step_size = 1e-3f;

typedef struct struct_coeff
{
	float a1, b1, c1, d1, e1, f1;
	float a2, b2, c2, d2, e2, f2;
	struct_coeff(float _a1, float _b1, float _c1, float _d1, float _e1, float _f1,
		float _a2, float _b2, float _c2, float _d2, float _e2, float _f2)
	{
		a1 = _a1, b1 = _b1, c1 = _c1, d1 = _d1, e1 = _e1, f1 = _f1;
		a2 = _a2, b2 = _b2, c2 = _c2, d2 = _d2, e2 = _e2, f2 = _f2;
	};
} Coeff;

// a*sin(bx+cy+d)
std::vector<std::vector<Coeff>> node_coeffs(node_res, std::vector<Coeff>(elem_func_size, Coeff(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
std::vector<std::vector<Coeff>> node_coeffs_momentum(node_res, std::vector<Coeff>(elem_func_size, Coeff(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));


std::vector<std::vector<vec2>>& getVelocities()
{
	return velocities;
}

std::vector<std::vector<vec2>>& getNodeVelocities()
{
#pragma omp parallel for
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
		{
			float x = 2.0f * i / res - 1.0f;
			float y = 2.0f * j / res - 1.0f;

			float vel_x = 0.0f;
			float vel_y = 0.0f;

			float xx = x * x;
			float yy = y * y;
			for (int n = 0; n < elem_func_size; n++)
			{
				Coeff& nd = node_coeffs[0][n];

				vel_x += nd.a1 * exp(-nd.b1 * nd.b1 * xx - nd.c1 * nd.c1 * yy) * sin(nd.d1 * x + nd.e1 * y + nd.f1);
				vel_y += nd.a2 * exp(-nd.b2 * nd.b2 * xx - nd.c2 * nd.c2 * yy) * sin(nd.d2 * x + nd.e2 * y + nd.f2);
			}
			node_velocities[i][j].x = vel_x;
			node_velocities[i][j].y = vel_y;
		}
	return node_velocities;
}

void init()
{
	srand(time(0));

	// insert source fluid
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
		{
			float x = 2.0f * i / res - 1.0f;
			float y = 2.0f * j / res - 1.0f;
			velocities[i][j] = glm::vec2(
				cos(x) * cos(y) + 3.0f * sin(x) * sin(y) +
				0.1f * cos(2 * x + 1) * cos(2 * y + 1) + 0.5f * sin(3 * x + 1) * sin(3 * y + 1),
				sin(x) * sin(y) + 3.0f * cos(x) * cos(y) +
				0.1f * sin(2 * x + 1) * sin(2 * y + 1) + 0.5f * cos(3 * x + 1) * cos(3 * y + 1)
				//x, -y
				//cos(x) * cos(y), sin(x) * sin(y)
				//cos(x), sin(y)
			);
		}

	for (int n = 0; n < elem_func_size; n++)
	{
		Coeff& nd = node_coeffs[0][n];

		nd.a1 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.a2 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.b1 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.b2 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.c1 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.c2 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.d1 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.d2 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.e1 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.e2 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.f1 = 2.0f * rand() / RAND_MAX - 1.0f;
		nd.f2 = 2.0f * rand() / RAND_MAX - 1.0f;

	}
}

bool is_fit = false;
bool fit()
{

	static int counter = 0;
	static std::vector<std::vector<float>> common_factor_x(res, std::vector<float>(res, 0));
	static std::vector<std::vector<float>> common_factor_y(res, std::vector<float>(res, 0));

	float error_x = 0.0;
	float error_y = 0.0f;
#pragma omp parallel for reduction(+:error_x, error_y)
	for (int i = 0; i < res; i++)
	{
		float local_error_x = 0.0f;
		float local_error_y = 0.0f;
		for (int j = 0; j < res; j++)
		{
			float x = 2.0f * i / res - 1.0f;
			float y = 2.0f * j / res - 1.0f;

			float cf_x = 0.0f;
			float cf_y = 0.0f;
			float xx = x * x;
			float yy = y * y;
			for (int n = 0; n < elem_func_size; n++)
			{
				Coeff& nd = node_coeffs[0][n];

				cf_x += nd.a1 * exp(-nd.b1 * nd.b1 * xx - nd.c1 * nd.c1 * yy) * sin(nd.d1 * x + nd.e1 * y + nd.f1);
				cf_y += nd.a2 * exp(-nd.b2 * nd.b2 * xx - nd.c2 * nd.c2 * yy) * sin(nd.d2 * x + nd.e2 * y + nd.f2);
			}
			common_factor_x[i][j] = cf_x;
			common_factor_y[i][j] = cf_y;

			local_error_x += pow(velocities[i][j].x - cf_x, 2);
			local_error_y += pow(velocities[i][j].y - cf_y, 2);
		}
		error_x += local_error_x;
		error_y += local_error_y;
	}
	if (counter % 10 == 0)
	{
		//Coeff& nd = node_coeffs[0][0];
		printf("err_x %3.3f\terr_y %3.3f \n", 100.0f * error_x / res, 100.0f * error_y / res);
		//printf(" * %3.3f %3.3f %3.3f %3.3f %3.3f %3.3f \n", nd.a1, nd.b1, nd.c1, nd.d1, nd.e1, nd.f1);
		//printf(" * %3.3f %3.3f %3.3f %3.3f %3.3f %3.3f \n", nd.a2, nd.b2, nd.c2, nd.d2, nd.e2, nd.f2);
	}
	if (error_x / res < 10.0f && error_y / res < 10.0f)
	{
		printf("fit!\t\t\r");
		return is_fit = true;
	}

	// fit the functions to the grid
#pragma omp parallel for
	for (int n = 0; n < elem_func_size; n++)
	{
		Coeff& nd = node_coeffs[0][n];

		float& a1 = nd.a1;
		float& a2 = nd.a2;
		float& b1 = nd.b1;
		float& b2 = nd.b2;
		float& c1 = nd.c1;
		float& c2 = nd.c2;
		float& d1 = nd.d1;
		float& d2 = nd.d2;
		float& e1 = nd.e1;
		float& e2 = nd.e2;
		float& f1 = nd.f1;
		float& f2 = nd.f2;

		float temp_a1 = 0.0f;
		float temp_a2 = 0.0f;
		float temp_b1 = 0.0f;
		float temp_b2 = 0.0f;
		float temp_c1 = 0.0f;
		float temp_c2 = 0.0f;
		float temp_d1 = 0.0f;
		float temp_d2 = 0.0f;
		float temp_e1 = 0.0f;
		float temp_e2 = 0.0f;
		float temp_f1 = 0.0f;
		float temp_f2 = 0.0f;

		constexpr int stride = 1;
		for (int i = stride/2; i < res- stride/2; i+=stride)
		{
			for (int j = stride/2; j < res- stride/2; j+=stride)
			{
				float x = 2.0f * i / res - 1.0f;
				float y = 2.0f * j / res - 1.0f;

				float& vel_x = velocities[i][j].x;
				float& vel_y = velocities[i][j].y;

				float x_common_coeff = exp(-b1*b1*x*x -c1*c1*y*y);
				float y_common_coeff = exp(-b2*b2*x*x -c2*c2*y*y);
				float x_sin = sin(d1*x + e1*y + f1);
				float x_cos = cos(d1*x + e1*y + f1);
				float y_sin = sin(d2*x + e2*y + f2);
				float y_cos = cos(d2*x + e2*y + f2);

				float delta_a1 = x_common_coeff * x_sin * (vel_x - common_factor_x[i][j]);
				float delta_b1 = -2.0f * a1 * b1 * x * x * delta_a1;
				float delta_c1 = -2.0f * a1 * c1 * y * y * delta_a1;
				float delta_f1 = x_common_coeff * x_cos * (vel_x - common_factor_x[i][j]);
				float delta_d1 = x * delta_f1;
				float delta_e1 = y * delta_f1;

				float delta_a2 = y_common_coeff * y_sin * (vel_y - common_factor_y[i][j]);
				float delta_b2 = -2.0f * a2 * b2 * x * x * delta_a2;
				float delta_c2 = -2.0f * a2 * c2 * y * y * delta_a2;
				float delta_f2 = y_common_coeff * y_cos * (vel_y - common_factor_y[i][j]);
				float delta_d2 = x * delta_f2;
				float delta_e2 = y * delta_f2;

				temp_a1 += delta_a1;
				temp_a2 += delta_a2;
				temp_b1 += delta_b1;
				temp_b2 += delta_b2;
				temp_c1 += delta_c1;
				temp_c2 += delta_c2;
				temp_d1 += delta_d1;
				temp_d2 += delta_d2;
				temp_e1 += delta_e1;
				temp_e2 += delta_e2;
				temp_f1 += delta_f1;
				temp_f2 += delta_f2;
			}
		}

		Coeff& nd_mmt = node_coeffs_momentum[0][n];
		nd_mmt.a1 = 0.1f * nd_mmt.a1 + 0.9f * temp_a1;
		nd_mmt.a2 = 0.1f * nd_mmt.a2 + 0.9f * temp_a2;
		nd_mmt.b1 = 0.1f * nd_mmt.b1 + 0.9f * temp_b1;
		nd_mmt.b2 = 0.1f * nd_mmt.b2 + 0.9f * temp_b2;
		nd_mmt.c1 = 0.1f * nd_mmt.c1 + 0.9f * temp_c1;
		nd_mmt.c2 = 0.1f * nd_mmt.c2 + 0.9f * temp_c2;
		nd_mmt.d1 = 0.1f * nd_mmt.d1 + 0.9f * temp_d1;
		nd_mmt.d2 = 0.1f * nd_mmt.d2 + 0.9f * temp_d2;
		nd_mmt.e1 = 0.1f * nd_mmt.e1 + 0.9f * temp_e1;
		nd_mmt.e2 = 0.1f * nd_mmt.e2 + 0.9f * temp_e2;
		nd_mmt.f1 = 0.1f * nd_mmt.f1 + 0.9f * temp_f1;
		nd_mmt.f2 = 0.1f * nd_mmt.f2 + 0.9f * temp_f2;

		a1 += nd_mmt.a1 * fit_step_size;
		a2 += nd_mmt.a2 * fit_step_size;
		b1 += nd_mmt.b1 * fit_step_size;
		b2 += nd_mmt.b2 * fit_step_size;
		c1 += nd_mmt.c1 * fit_step_size;
		c2 += nd_mmt.c2 * fit_step_size;
		d1 += nd_mmt.d1 * fit_step_size;
		d2 += nd_mmt.d2 * fit_step_size;
		e1 += nd_mmt.e1 * fit_step_size;
		e2 += nd_mmt.e2 * fit_step_size;
		f1 += nd_mmt.f1 * fit_step_size;
		f2 += nd_mmt.f2 * fit_step_size;
	}

	
	counter++;
	return is_fit;
}

void step()
{
	static int counter = 0;
	if (counter++ % 50 == 0)
		fit_step_size *= 1.1f;

	is_fit = false;
#pragma omp parallel for
	for (int i = 0; i < res; i++)
	{
		float x = 2.0f * i / res - 1.0f;
		for (int j = 0; j < res; j++)
		{
			float y = 2.0f * j / res - 1.0f;

			float u_x = 0.0f;
			float u_y = 0.0f;
			float grad_x_u_x = 0.0f;
			float grad_x_u_y = 0.0f;
			float grad_y_u_x = 0.0f;
			float grad_y_u_y = 0.0f;
			for (int n = 0; n < elem_func_size; n++)
			{
				Coeff& nd = node_coeffs[0][n];

				float& a1 = nd.a1;
				float& a2 = nd.a2;
				float& b1 = nd.b1;
				float& b2 = nd.b2;
				float& c1 = nd.c1;
				float& c2 = nd.c2;
				float& d1 = nd.d1;
				float& d2 = nd.d2;
				float& e1 = nd.e1;
				float& e2 = nd.e2;
				float& f1 = nd.f1;
				float& f2 = nd.f2;

				float x_common_coeff = exp(-b1 * b1 * x * x - c1 * c1 * y * y);
				float y_common_coeff = exp(-b2 * b2 * x * x - c2 * c2 * y * y);
				float x_sin = sin(d1 * x + e1 * y + f1);
				float x_cos = cos(d1 * x + e1 * y + f1);
				float y_sin = sin(d2 * x + e2 * y + f2);
				float y_cos = cos(d2 * x + e2 * y + f2);

				u_x += a1 * x_common_coeff * x_sin;
				u_y += a2 * y_common_coeff * y_sin;

				grad_x_u_x += a1 * x_common_coeff * (d1 * x_cos + -2.0f * b1 * b1 * x * x_sin);
				grad_x_u_y += a1 * x_common_coeff * (e1 * x_cos + -2.0f * c1 * c1 * y * x_sin);

				grad_y_u_x += a2 * y_common_coeff * (d2 * y_cos + -2.0f * b2 * b2 * x * y_sin);
				grad_y_u_y += a2 * y_common_coeff * (e2 * y_cos + -2.0f * c2 * c2 * y * y_sin);
			}
			velocities[i][j].x += (u_x * grad_x_u_x + u_y * grad_x_u_y) * sim_step_size;
			velocities[i][j].y += (u_x * grad_y_u_x + u_y * grad_y_u_y) * sim_step_size;
		}
	}


	// fit the functions to the grid
#pragma omp parallel for
	for (int n = 0; n < elem_func_size; n++)
	{
		Coeff& nd = node_coeffs[0][n];

		float& a1 = nd.a1;
		float& a2 = nd.a2;
		float& b1 = nd.b1;
		float& b2 = nd.b2;
		float& c1 = nd.c1;
		float& c2 = nd.c2;
		float& d1 = nd.d1;
		float& d2 = nd.d2;
		float& e1 = nd.e1;
		float& e2 = nd.e2;
		float& f1 = nd.f1;
		float& f2 = nd.f2;

		float temp_a1 = 0.0f;
		float temp_a2 = 0.0f;
		float temp_b1 = 0.0f;
		float temp_b2 = 0.0f;
		float temp_c1 = 0.0f;
		float temp_c2 = 0.0f;
		float temp_d1 = 0.0f;
		float temp_d2 = 0.0f;
		float temp_e1 = 0.0f;
		float temp_e2 = 0.0f;
		float temp_f1 = 0.0f;
		float temp_f2 = 0.0f;

		constexpr int stride = 1;
		for (int i = stride / 2; i < res - stride / 2; i += stride)
		{
			for (int j = stride / 2; j < res - stride / 2; j += stride)
			{
				float x = 2.0f * i / res - 1.0f;
				float y = 2.0f * j / res - 1.0f;

				float delta_a1 = 0;
				float delta_b1 = -4.0f * b1 * (b1 * b1 - b2 * b2);
				float delta_c1 = -4.0f * c1 * (c1 * c1 - c2 * c2);
				float delta_d1 = -2.0f * d1 * (d1 - d2);
				float delta_e1 = -2.0f * e1 * (e1 - e2);
				float delta_f1 = -2.0f * f1 * (f1 - f2);

				float delta_a2 = 0;
				float delta_b2 = -4.0f * b2 * (b2 * b2 - b1 * b1);
				float delta_c2 = -4.0f * c2 * (c2 * c2 - c1 * c1);
				float delta_d2 = -2.0f * d2 * (d2 - d1);
				float delta_e2 = -2.0f * e2 * (e2 - e1);
				float delta_f2 = -2.0f * f2 * (f2 - f1);

				temp_a1 += delta_a1;
				temp_a2 += delta_a2;
				temp_b1 += delta_b1;
				temp_b2 += delta_b2;
				temp_c1 += delta_c1;
				temp_c2 += delta_c2;
				temp_d1 += delta_d1;
				temp_d2 += delta_d2;
				temp_e1 += delta_e1;
				temp_e2 += delta_e2;
				temp_f1 += delta_f1;
				temp_f2 += delta_f2;
			}
		}

		a1 += temp_a1 * fit_step_size * 0.1f;
		a2 += temp_a2 * fit_step_size * 0.1f;
		b1 += temp_b1 * fit_step_size * 0.1f;
		b2 += temp_b2 * fit_step_size * 0.1f;
		c1 += temp_c1 * fit_step_size * 0.1f;
		c2 += temp_c2 * fit_step_size * 0.1f;
		d1 += temp_d1 * fit_step_size * 0.1f;
		d2 += temp_d2 * fit_step_size * 0.1f;
		e1 += temp_e1 * fit_step_size * 0.1f;
		e2 += temp_e2 * fit_step_size * 0.1f;
		f1 += temp_f1 * fit_step_size * 0.1f;
		f2 += temp_f2 * fit_step_size * 0.1f;
	}
	printf("step!\t\t\r");
}