#include "glm/glm/glm.hpp"
#include "geometry.h"

extern "C"
{
	texture<float, cudaTextureType3D, cudaReadModeElementType> volume_tex;

	__global__ void render(float* output, int output_width, int output_height, float* bbox_size, float* camera_position, float* camera_to_world_matrix, float majorant)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		const int j = blockDim.y * blockIdx.y + threadIdx.y;

		if (i >= output_width || j >= output_height)
		{
			return;
		}

		//Bbox of volume.
		Bbox bbox;
		bbox.min = glm::vec3(0.0f);
		bbox.max = *reinterpret_cast<glm::vec3*>(bbox_size);

		//Ray generation in view space.
		auto aspect_ratio = output_height / static_cast<float>(output_width);
		auto x = 0.5f - (i + 0.5f) / output_width; //Fixed to [-0.5, 0.5]
		auto y = aspect_ratio * 0.5f - (j + 0.5f) / output_width; //Fixed to [-aspect_ratio / 2, aspect_ratio / 2]
		auto z = 0.865f; //Fixed to horizontal FoV of 60 degrees.

		//Ray direction from view space to world space.
		Ray ray;
		ray.origin = *reinterpret_cast<glm::vec3*>(camera_position);
		ray.direction = (*reinterpret_cast<glm::mat3*>(camera_to_world_matrix)) * glm::normalize(glm::vec3(x, y, z));

		//Ray-bbox intersection.
		auto result = bbox.intersect(ray);

		//Write the output.
		auto vec_output = reinterpret_cast<glm::vec3*>(output);
		if (result.y < 0.0f)
		{
			//From Peter Shirley's "Ray Tracing in One Weekend"
			auto t = 0.5f * (ray.direction + 1.0f);
			vec_output[j * output_width + i] = (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
		}
		else
		{
			auto distance = fmaxf(0.0f, result.x);
			auto total = 0.0f;
			for (int k = 0; k < 1000; ++k)
			{
				auto xyz = ray.getPosition(distance) / bbox.max;
				total += tex3D<float>(volume_tex, xyz.x, xyz.y, xyz.z);
				distance += 0.001f;
			}

			vec_output[j * output_width + i] = glm::vec3(expf(-total * 0.01f));
		}
	}
}