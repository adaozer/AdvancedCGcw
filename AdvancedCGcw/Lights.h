#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"
#include <algorithm>;

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const Vec3& wi)
	{
		if (Dot(wi, triangle->gNormal()) < 0)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
	}
	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi); // very easy small changes (cosine hemisphere sample and pdf)
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light
{
public:
	Colour emission;
	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		return emission;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env;
	float sumTotal = 0.f;
	std::vector<float> marginalPDF;
	std::vector<float> conditionalPDF;
	std::vector<float> rowCDF;
	std::vector<float> columnCDF;
	EnvironmentMap(Texture* _env)
	{
		env = _env;
		computeCDF();
	}

	void computeCDF() { // this is for computing the actual tables for the tabular distribution
		//you need sum of rows (for each row), total sum. then we build pdfs (marginal and conditional) which we then use to make row and column cdfs. 
		// this is all based off of the formulas in the slides
		sumTotal = 0.f;
		std::vector<float> sumRows(env->height);
		float WH = 1.f / (env->height * env->width);
		marginalPDF.resize(env->height);
		conditionalPDF.resize(env->height * env->width);
		for (int y = 0; y < env->height; y++) {
				float v = (y + 0.5f) / env->height;
				float theta = M_PI * v;
			for (int x = 0; x < env->width; x++) {
				sumTotal += env->texels[y * env->width + x].Lum() * sinf(theta);
				sumRows[y] += (1.f / env->width) * env->texels[y * env->width + x].Lum() * sinf(theta);
			}
		}
		float denom = WH * sumTotal;
		for (int i = 0; i < sumRows.size(); i++) {
			marginalPDF[i] = sumRows[i] / denom;
		}
		for (int y = 0; y < env->height; y++) {
			float v = (y + 0.5f) / env->height;
			float theta = M_PI * v;
			for (int x = 0; x < env->width; x++) {
				conditionalPDF[y * env->width + x] = ((env->texels[y * env->width + x].Lum()  * sinf(theta)) / denom) / marginalPDF[y];
			}
		}
		rowCDF.resize(env->height * (env->width + 1));
		columnCDF.resize(env->height + 1);
		for (int i = 0; i < env->height; i++) {
			columnCDF[i+1] = columnCDF[i] + marginalPDF[i]; 
		}
		for (int y = 0; y < env->height; y++) {
			for (int x = 0; x < env->width; x++) {
			int pos = y * (env -> width + 1) + x;
			rowCDF[pos + 1] = rowCDF[pos] + conditionalPDF[y * env->width + x];
			}
		}
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		float r1 = sampler->next(); // this is sampling for environment lights, convert from u,v space to theta, phi space
		int row = (int)(std::upper_bound(columnCDF.begin(), columnCDF.end(), r1 * env->height) - columnCDF.begin() - 1);
		row = std::clamp(row, 0, env->height - 1);
		float r2 = sampler->next();
		auto rowBegin = rowCDF.begin() + row * (env->width + 1);
		auto rowEnd = rowBegin + env->width + 1;
		int column = (int)(std::upper_bound(rowBegin, rowEnd, r2 * env->width) - rowBegin - 1); 
		column = std::clamp(column, 0, env->width - 1);
		// you sample 2 numbers, then you find out which row and column they belong to
		float puv = marginalPDF[row] * conditionalPDF[row * env->width + column]; // pdf kinda
		float u = (column + 0.5f) / env->width;
		float v = (row + 0.5f) / env->height;
		float theta = M_PI * v;
		float phi = 2 * M_PI * u;
		pdf = puv / (2 * M_PI * M_PI * sinf(theta)); // this is where we go from u,v space to theta,phi space
		Vec3 wi(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
		reflectedColour = evaluate(wi);
		return wi;
	}

	Colour evaluate(const Vec3& wi)
	{
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		float v = acosf(std::max(-1.0f, std::min(1.0f, wi.y))) / M_PI;
		return env->sample(u, v);
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		float theta = acosf(wi.y);
		float v = theta / M_PI;
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		int row = v * env->height;
		int column = u * env->width;
		float puv = marginalPDF[row] * conditionalPDF[row * env->width + column];
		return puv / (2 * M_PI * M_PI * sinf(theta)); // basically just rewriting pdf in sample
	}

	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		float total = 0;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.0f / (4 * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		float r1 = sampler->next();
		int row = (int)(std::upper_bound(columnCDF.begin(), columnCDF.end(), r1) - columnCDF.begin() - 1);
		float r2 = sampler->next();
		auto rowBegin = rowCDF.begin() + row * (env->width + 1);
		auto rowEnd = rowBegin + env->width + 1;
		int column = (int)(std::upper_bound(rowBegin, rowEnd, r2) - rowBegin - 1);
		float puv = marginalPDF[row] * conditionalPDF[row * env->width + column];
		float u = (column + 0.5f) / env->width;
		float v = (row + 0.5f) / env->height;
		float theta = M_PI * v;
		float phi = 2 * M_PI * u;
		pdf = puv / (2 * M_PI * M_PI * sinf(theta));
		Vec3 wi(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
		return wi;
	}
};