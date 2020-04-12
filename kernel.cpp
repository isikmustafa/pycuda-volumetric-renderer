#include "glm/glm/glm.hpp"
#include "geometry.h"
#include <curand_kernel.h>

extern "C"
{
	texture<float, cudaTextureType3D, cudaReadModeElementType> gVolumeTexture;

	__device__ curandStateMRG32k3a gRandNumStates[%(NUMBER_OF_GENERATORS)s];
	
	__global__ void initRandNumGenerators()
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i < %(NUMBER_OF_GENERATORS)s)
		{
			curand_init(48754595238, i, 0, &gRandNumStates[i]);
		}
	}

	__device__ glm::vec3 getBackgroundRadiance(const Ray& ray)
	{
		//From Peter Shirley's "Ray Tracing in One Weekend"
		auto t = 0.5f * (ray.direction + 1.0f);
		return glm::vec3(1.0f - t) + t * glm::vec3(0.5f, 0.7f, 1.0f);
	}

	__global__ void render(float* output, int output_width, int output_height, float* bbox_size, float* camera_position, float* camera_to_world_matrix,
		float absorption_factor, float scattering_factor)
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

		//Get pixel index and corresponding random number generator state.
		auto pixel_index = j * output_width + i;
		auto& rand_state = gRandNumStates[pixel_index];

		//Ray generation in view space.
		auto aspect_ratio = output_height / static_cast<float>(output_width);
		auto x = 0.5f - (i + curand_uniform(&rand_state)) / output_width; //Fixed to [-0.5, 0.5]
		auto y = aspect_ratio * 0.5f - (j + curand_uniform(&rand_state)) / output_width; //Fixed to [-aspect_ratio / 2, aspect_ratio / 2]
		auto z = 0.865f; //Fixed to horizontal FoV of 60 degrees.

		//Ray direction from view space to world space.
		Ray ray;
		ray.origin = *reinterpret_cast<glm::vec3*>(camera_position);
		ray.direction = (*reinterpret_cast<glm::mat3*>(camera_to_world_matrix)) * glm::normalize(glm::vec3(x, y, z));

		auto vec_output = reinterpret_cast<glm::vec3*>(output);
		glm::vec3 throughput(1.0f);

		//Delta tracking.
		//majorant = [largest extinction coefficient] = [largest density] * ([absorption_factor] + [scattering_factor]) = 1.0 * ([absorption_factor] + [scattering_factor])
		auto majorant = absorption_factor + scattering_factor;
		auto scattering_throughput = scattering_factor / (absorption_factor + scattering_factor);
		auto result = bbox.intersect(ray);
		auto distance = fmaxf(0.0f, result.x);
		while (true)
		{
			distance -= logf(curand_uniform(&rand_state)) / majorant;
			if (distance >= result.y)
			{
				break;
			}

			auto position = ray.getPosition(distance);
			auto vol_position = position / bbox.max;
			auto density = tex3D<float>(gVolumeTexture, vol_position.x, vol_position.y, vol_position.z);

			//Real collision occurs. Select new direction based on isotropic phase function.
			if (curand_uniform(&rand_state) < density)
			{
				ray.origin = position;

				//Uniformly sample a direction from unit sphere.
				glm::vec3 new_direction;
				while (true)
				{
					new_direction = glm::vec3(curand_normal(&rand_state), curand_normal(&rand_state), curand_normal(&rand_state));
					auto length = glm::length(new_direction);

					if (length > 0.00001f)
					{
						new_direction /= length;
						break;
					}
				}
				ray.direction = new_direction;
				throughput *= scattering_throughput;

				result = bbox.intersect(ray);
				distance = fmaxf(0.0f, result.x);
			}

			//Null collision occurs, otherwise. So, move on.
		}

		vec_output[pixel_index] += throughput * getBackgroundRadiance(ray);
	}
}