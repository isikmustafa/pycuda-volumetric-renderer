#include "glm/glm/glm.hpp"
#include "geometry.h"

extern "C"
{
	__global__ void renderDeltaTracking(float* output, int output_width, int output_height, float* camera_position)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		const int j = blockDim.y * blockIdx.y + threadIdx.y;

		Bbox bbox;
		bbox.min = -glm::vec3(0.5f);
		bbox.max = glm::vec3(0.5f);

		//Ray generation in view space.
		auto aspect_ratio = output_height / static_cast<float>(output_width);
		auto x = 0.5f - (i + 0.5f) / output_width; //Fixed to [-0.5, 0.5]
		auto y = aspect_ratio * 0.5f - (j + 0.5f) / output_width; //Fixed to [-aspect_ratio / 2, aspect_ratio / 2]
		auto z = 0.865f; //Fixed to horizontal FoV of 60 degrees.

		//Ray direction from view space to world space.
		Ray ray;
		ray.origin = { camera_position[0], camera_position[1], camera_position[2] };
		ray.direction = glm::normalize(glm::vec3(x, y, z)); //TODO: get 3x3 matrix and multiply with ray.direction to get ray direction in world coordinates.

		//Ray-bbox intersection.
		auto result = bbox.intersect(ray);

		//Write the output.
		auto vec_output = reinterpret_cast<glm::vec3*>(output);
		if (result.y < 0.0f)
		{
			vec_output[j * output_width + i] = { 0.0f, 0.0f , 0.0f };
		}
		else
		{
			vec_output[j * output_width + i] = { result.x, result.x, result.x };
		}
	}
}