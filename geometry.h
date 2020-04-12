struct Ray
{
	glm::vec3 origin;
	glm::vec3 direction;

	__device__ glm::vec3 getPosition(float distance) const
	{
		return origin + distance * direction;
	}
};

struct Bbox
{
	glm::vec3 min;
	glm::vec3 max;

	__device__ glm::vec2 intersect(const Ray& ray) const
	{
		auto inv_dir = 1.0f / ray.direction;

		auto t0 = (min - ray.origin) * inv_dir;
		auto t1 = (max - ray.origin) * inv_dir;

		auto min = glm::min(t0, t1);
		auto max = glm::max(t0, t1);
		auto tm_min = glm::max(min.x, glm::max(min.y, min.z));
		auto tm_max = glm::min(max.x, glm::min(max.y, max.z));

		return tm_max < tm_min ? glm::vec2(-1.0f) : glm::vec2(tm_min, tm_max);
	}
};