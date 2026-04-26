#pragma once

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning( disable : 4244)
#pragma warning( disable : 4305) // Double to float

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt)
	{
		float n = iorExt / iorInt;
		if (cosTheta < 0.f) {
			n = 1.f / n;
			cosTheta = -cosTheta;
		}
		
		float sin_i = sqrtf(1.f - cosTheta * cosTheta);
		float sin_t = n * sin_i;

		float cosTheta_t = sqrtf(1.f - sin_t * sin_t);
		float parallel = (cosTheta - n * cosTheta_t) / (cosTheta + n * cosTheta_t);
		float perpendicular = (n * cosTheta - cosTheta_t) / (n * cosTheta + cosTheta_t);
		float average = (parallel * parallel + perpendicular * perpendicular) / 2.f;
		return average;
	}

	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		float cos2 = cosTheta * cosTheta;
		float sin2 = 1.f - cos2;
		Colour iorK = ior * ior + k * k;

		Colour parallel = (iorK * cos2 - ior * (2.f * cosTheta) + Colour(sin2, sin2, sin2)) / ((iorK)*cos2 + ior * (2.f * cosTheta) + Colour(sin2, sin2, sin2));
		Colour perpendicular = (iorK - ior * (2.f * cosTheta) + Colour(cos2, cos2, cos2)) / (iorK + ior * (2.f * cosTheta) + Colour(cos2, cos2, cos2));
		Colour average = (parallel + perpendicular) / 2.f;
		return average;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		if (wi.z <= 0.f) return 1e10f;
		float cos2 = wi.z * wi.z;
		float sin2 = 1.f - cos2;
		float tan2 = sin2 / cos2;
		return (sqrtf(1.f + alpha * alpha * tan2) - 1.f) / 2.f;
	}

	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		float wiGGX = 1.f / (1.f + lambdaGGX(wi, alpha));
		float woGGX = 1.f / (1.f + lambdaGGX(wo, alpha));
		return wiGGX * woGGX;
	}

	static float Dggx(Vec3 h, float alpha)
	{
		float alpha2 = alpha * alpha;
		float cos2 = h.z * h.z;
		float denom = cos2 * (alpha2 - 1) + 1;
		return alpha2 / (M_PI * denom * denom);
	}
};

class BSDF
{
public:
	Colour emission;
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
};


class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi);
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
	Texture* albedo;
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woWorld = shadingData.frame.toLocal(shadingData.wo);
		woWorld.x = -woWorld.x;
		woWorld.y = -woWorld.y;
		Vec3 wi = shadingData.frame.toWorld(woWorld);
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / Dot(wi, shadingData.sNormal);
		pdf = 1;
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / Dot(wi, shadingData.sNormal);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 0;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};


class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = std::max(1e-3f, 1.62142f * sqrtf(roughness));
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		float r1 = sampler->next();
		float r2 = sampler->next(); // sample 2 numbers
		float cosTheta_m = sqrtf((1.f - r1) / (r1 * (alpha * alpha - 1.f) + 1.f));
		float theta_m = acosf(cosTheta_m);
		float phi_m = 2 * M_PI * r2;
		float sinTheta_m = sinf(theta_m); // this is all to get wm
		Vec3 wm = Vec3(sinTheta_m * cosf(phi_m), sinTheta_m * sinf(phi_m), cosTheta_m); // to calcualte wi
		Vec3 wi = wm * (2.f * Dot(wm, woLocal)) - woLocal; // this is what we will return
		if (wi.z <= 0.f) {
			pdf = 0.f;
			reflectedColour = Colour(0.f, 0.f, 0.f);
			return shadingData.frame.toWorld(wi);
		}
		float D = ShadingHelper::Dggx(wm, alpha); // D, G, F for reflected colour
		float G = ShadingHelper::Gggx(wi, woLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(Dot(woLocal, wm), eta, k);
		pdf = (D * cosTheta_m) / (4.f * Dot(woLocal, wm)); // simple formula (all in the slides)
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F * (G * D) / (4.f * woLocal.z * wi.z);
		return shadingData.frame.toWorld(wi);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z <= 0.f || woLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);
		Vec3 wm = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(wm, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(Dot(woLocal, wm), eta, k);
		return albedo->sample(shadingData.tu, shadingData.tv) * (F * G * D) / (4.f * woLocal.z * wiLocal.z);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wm = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(wm, alpha);
		return (D * wm.z) / (4.f * Dot(woLocal, wm));
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GlassBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	GlassBSDF() = default;
	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		float n = extIOR / intIOR;
		float cosTheta = woLocal.z;
		if (cosTheta < 0.f) {
			n = 1.f / n;
			cosTheta = -cosTheta;
		}
		float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
		float sin_t = n * sinTheta; // this is all stuff we will pass into fresnel to see if f < r 

		if (sin_t >= 1.f) {
			pdf = 1.f;
			reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / cosTheta;
			Vec3 wiLocal(-woLocal.x, -woLocal.y, woLocal.z);
			return shadingData.frame.toWorld(wiLocal);
		}

		float cosTheta_t = sqrtf(1.f - sin_t * sin_t);

		float fres = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		float r = sampler->next();
		if (r < fres) { // reflection
			pdf = fres;
			reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * fres / cosTheta;
			Vec3 wiLocal(-woLocal.x, -woLocal.y, woLocal.z);
			Vec3 woWorld = shadingData.frame.toWorld(wiLocal);
			return woWorld;
		} //refraction
		pdf = 1 - fres;
		reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) * (1 - fres) * n * n) / cosTheta_t;
		if (woLocal.z > 0.f) {
			cosTheta_t = -cosTheta_t;
		} // plug in formulas based on if reflection or reflaction
		Vec3 wiLocal(-n * woLocal.x, -n * woLocal.y, cosTheta_t);
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		return wiWorld;
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return Colour(0.f, 0.f, 0.f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 0;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Dielectric sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	float alphaToPhongExponent()
	{
		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Plastic sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};