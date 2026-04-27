#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <OpenImageDenoise/oidn.h>
#include <atomic>

struct VPL {
	ShadingData shadingData;
	Colour Le;
};

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	std::atomic<int> tiles;
	int tileSize = 16;
	Colour* albedoBuffer = nullptr;
	Colour* normalBuffer = nullptr;
	std::vector<VPL> vpls;
	int N = 8; // VPL number

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new GaussianFilter(1.f, 2.f));
		albedoBuffer = new Colour[scene->camera.width * scene->camera.height];
		normalBuffer = new Colour[scene->camera.width * scene->camera.height];
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		clear();
	}
	void clear()
	{
		film->clear();
		memset(albedoBuffer, 0, film->width * film->height * sizeof(Colour));
    	memset(normalBuffer, 0, film->width * film->height * sizeof(Colour));
	}

	Colour vplSum(ShadingData shadingData) {
		// this is the formula in the slides, sigma 0 -> VPLn where you do emitted colour x * emitted colour vpl * geometry term * Le
		Colour sum(0.f, 0.f, 0.f);
		for (auto& vpl : vpls) {
			Vec3 wi = vpl.shadingData.x - shadingData.x;
			float r2 = wi.lengthSq();
			Vec3 dir = wi.normalize();
			float cosX = Dot(dir, shadingData.sNormal);
			float cosV = -Dot(dir, vpl.shadingData.sNormal);
			if (scene->visible(shadingData.x, vpl.shadingData.x)) {
				float gTerm = (cosX * cosV) / r2;
				if (gTerm <= 0.f) continue;
				Colour BSDFx = shadingData.bsdf->evaluate(shadingData, dir);
				Colour BSDFv = vpl.shadingData.bsdf->evaluate(vpl.shadingData, -dir);
				sum = sum + BSDFx * gTerm * BSDFv * vpl.Le;
			}
		}
		return sum / N;
	}

	void vplTrace(Sampler* sampler) { // this is literally light trace but I added the for loop
		vpls.clear();
		for (int i = 0; i < N; i++) {
			Colour emittedColour;
			float pmf, pdfPosition, pdfDirection;
			Light* light = scene->sampleLight(sampler, pmf);
			if (light->isArea()) { 
				Vec3 p = light->samplePositionFromLight(sampler, pdfPosition);
				Vec3 wi = light->sampleDirectionFromLight(sampler, pdfDirection);
				Colour Le = light->evaluate(-wi);
				Colour col = Le / pdfPosition; 
				Ray r; 
				r.init(p + wi * EPSILON, wi); 
				vplTracePath(r, col / pmf, Le, sampler, 0);
			}
	}
	}

	void vplTracePath(Ray&r, Colour pathThroughput, Colour Le, Sampler* sampler, int depth) {
		// this is literally light trace path but I added the VPL lines
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r); 
		if (shadingData.t == FLT_MAX) return;
		if (shadingData.bsdf->isLight()) return; 
		VPL vpl;
		vpl.Le = pathThroughput;
		vpl.shadingData = shadingData;
		vpls.push_back(vpl);
		if (depth > 10) return; 
		if (depth > 3) { 
			float rrp = std::min(pathThroughput.Lum(), 1.f);
			if (sampler->next() > rrp) return;
			pathThroughput = pathThroughput / rrp;
		}
		Colour reflectedColour;
		float pdf; 
		Vec3 wi2 = shadingData.bsdf->sample(shadingData, sampler, reflectedColour, pdf);
		pathThroughput = pathThroughput * reflectedColour;
		if (pathThroughput.Lum() > 10.f) return; 
		Ray r2;
		r2.init(shadingData.x + wi2 * EPSILON, wi2); 
		vplTracePath(r2, pathThroughput, Le, sampler, depth + 1);
	}

	// connect to camera function for light tracing
	void connectToCamera(Vec3 p, Vec3 n, Colour col) {
		float x, y;
		if (!scene->camera.projectOntoCamera(p, x, y)) return;
		Vec3 camNormal = scene->camera.viewDirection;
		Vec3 camPos = scene->camera.origin; // get normal and pos of camera
		Vec3 directionTo = (camPos - p); // cam pos - position for cam direction
		float r2 = directionTo.lengthSq(); // r2 for gterm 
		directionTo = directionTo.normalize();
		float Afilm = scene->camera.Afilm; // afilm (which we apparently need later)
		float cosTheta = Dot(camNormal, directionTo); // for we and gterm
		float We = 1.f / (Afilm * powf(cosTheta, 4)); // We calculate
		float gTerm = -cosTheta * Dot(directionTo, n) / r2; // g term classic
		if (gTerm <= 0.f) return;
		Ray shadow;
		shadow.init(p + directionTo * EPSILON, directionTo); // for visibility check
		if (scene->traverse(shadow).t < (sqrtf(r2) - EPSILON)) return;
		film->splat(x, y, col * We * gTerm); //splat using gterm and we
	}

	void lightTrace(Sampler* sampler) {
		Colour emittedColour;
		float pmf, pdfPosition, pdfDirection;
		Light* light = scene->sampleLight(sampler, pmf); // sample light
		if (light->isArea()) { // only for area light
			Vec3 p = light->samplePositionFromLight(sampler, pdfPosition);
			Vec3 wi = light->sampleDirectionFromLight(sampler, pdfDirection); // sample dir and pos from light
			Colour Le = light->evaluate(-wi); // this is le (equation in slides)
			Colour col = Le / pdfPosition; // calculate col
			Ray r; 
			r.init(p + wi * EPSILON, wi); // will feed all this to other functions
			connectToCamera(p, light->normal(ShadingData(), p), col); // use col here, no normal so we just spawn one in
 			lightTracePath(r, col/pmf, Le, sampler, 0); // light trace path (where real stuff happens)
		}
	}

	void lightTracePath(Ray&r, Colour pathThroughput, Colour Le, Sampler* sampler, int depth) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r); // similar to path tracing
		if (shadingData.t == FLT_MAX) return;
		if (shadingData.bsdf->isLight()) return; // checks so we dont calculate stuff for no reason
		Vec3 wi = (scene->camera.origin - shadingData.x).normalize();
		Colour col = pathThroughput * shadingData.bsdf->evaluate(shadingData, wi) * Le;
		connectToCamera(shadingData.x, shadingData.sNormal, col); // wi and col so we can connect to camera
		if (depth > 10) return; // so it doesnt crash
		if (depth > 3) { // russian roulette
			float rrp = std::min(pathThroughput.Lum(), 1.f);
			if (sampler->next() > rrp) return;
			pathThroughput = pathThroughput / rrp;
		}
		Colour reflectedColour;
		float pdf; // similar to path tracing
		Vec3 wi2 = shadingData.bsdf->sample(shadingData, sampler, reflectedColour, pdf);
		pathThroughput = pathThroughput * reflectedColour;
		if (pathThroughput.Lum() > 10.f) return; // it becomes full white if i dont have htis
		Ray r2;
		r2.init(shadingData.x + wi2 * EPSILON, wi2); // we pass it all back (depth + 1)
		lightTracePath(r2, pathThroughput, Le, sampler, depth + 1);
	}

	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		if (shadingData.bsdf->isPureSpecular()) return Colour(0.f, 0.f, 0.f);

		float pmf;
		Light* light = scene->sampleLight(sampler, pmf); // sample light
		Colour emittedColour;
		float pdf;
		Vec3 samp = light->sample(shadingData, sampler, emittedColour, pdf); //sample
		if (light->isArea()) { // for area lights
			Vec3 wi1 = samp - shadingData.x;
			Vec3 wi = wi1.normalize(); // get wi
			float gTerm = (Dot(wi, shadingData.sNormal) * -Dot(wi, light->normal(shadingData, wi))) / wi1.lengthSq(); // gterm, wi * n , -wi * n' / r2 
			if (gTerm <= 0.f) return Colour(0.f,0.f,0.f); 
			if (scene->visible(shadingData.x, samp)) { // Visiblity for gterm
				float cosine = Dot(wi, shadingData.sNormal); // cosTheta for MIS
				Colour bsdf = shadingData.bsdf->evaluate(shadingData, wi); // BSDF colour from evaluate
				float pOmega = shadingData.bsdf->PDF(shadingData, wi); // BSDF's PDF for MIS
				return emittedColour * bsdf * gTerm / ((pdf*pmf) + ((pOmega / cosine) * gTerm)); // MIS added
			}
		} else { //env lighting
		float cosine = Dot(samp, shadingData.sNormal); 
		if (cosine <= 0.f) return Colour(0.f, 0.f, 0.f);
		Ray shadow;
		shadow.init(shadingData.x + samp * EPSILON, samp); //for visibility check
		if (scene->traverse(shadow).t == FLT_MAX) {
			Colour bsdf = shadingData.bsdf->evaluate(shadingData, samp);
			return emittedColour * bsdf * cosine / (pdf * pmf); // return based off formula
		}
	}
		return Colour(0.f, 0.f, 0.f);
	}

Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) {
				return (depth == 0) ? shadingData.bsdf->emit(shadingData, shadingData.wo) : Colour(0.f, 0.f, 0.f);

			}
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);
			// direct = direct + vplSum(shadingData); // enable this for instant radiosity!
			if (depth > 10) return direct;
			Colour indirect;
			float pdf;	
			Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, indirect, pdf);
			if (pdf == 0.f) return direct;
			r.init(shadingData.x + (wi * EPSILON), wi);
			float cosine = fabsf(Dot(wi, shadingData.sNormal));
			pathThroughput = pathThroughput * indirect * cosine / pdf;

			if (depth > 3) {
				float rrp = std::min(pathThroughput.Lum(), 1.f);
				if (sampler->next() < rrp) {
					pathThroughput = pathThroughput / rrp;
					return direct + pathTrace(r, pathThroughput, depth + 1, sampler);
				}
				else return direct;
			}
			return direct + pathTrace(r, pathThroughput, depth + 1, sampler);
		}
		return scene->background->evaluate(r.dir) * pathThroughput;
		}

	Colour direct(Ray& r, Sampler* sampler)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return scene->background->evaluate(r.dir);
	}

	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(r.dir);
	}
	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	// multithreading (this is used to calculate the tile bounds)
	void renderTile(int threadID)
	{
		int tilesX = (film->width + tileSize - 1) / tileSize; // little trick bceause of floor division, make sure that even if its not divisible by 16 we dont miss pixels
		int tilesY = (film->height + tileSize - 1) / tileSize;
		int tilesSum = tilesX * tilesY; // dynamic (each scene could have different proportions)

		while (1)
		{
			int tileID = tiles.fetch_add(1); // atomic += 1
			if (tileID >= tilesSum) break; // break if atomic = amount of tiles needed

			int tileX = tileID % tilesX;
			int tileY = tileID / tilesX; // this is all to set up the bounds, startX, endX etc.

			unsigned int startX = tileX * tileSize;
			unsigned int startY = tileY * tileSize;
			unsigned int endX = startX + tileSize;
			unsigned int endY = startY + tileSize;

			if (endX > film->width) endX = film->width;
			if (endY > film->height) endY = film->height; // in case we overflow (not divisible by 16 check)

			render2(startY, endY, startX, endX, threadID); //actual render
		}
	}

	void render()
	{ 
		film->incrementSPP(); // this must happen once, if it happens every thread it breaks
		tiles.store(0); // atomic = 0
		// vplTrace(&samplers[0]); // enable this for instant radiosity!
		for (unsigned int i = 0; i < numProcs; i++) {
			threads[i] = new std::thread(&RayTracer::renderTile, this, i); // spawn in threads = to num of processors
		}
		for (unsigned int i = 0; i < numProcs; i++) {
			threads[i]->join();
			delete threads[i]; // join and delete so it doesnt crash/burn memory
		}
	}
	
	void render2(unsigned int startY = 0, unsigned int endY = 0,
		unsigned int startX = 0, unsigned int endX = 0, unsigned int threadID = 0)
	{
		if (endY == 0) endY = film->height;
		if (endX == 0) endX = film->width; // in case this gets passed before rendertile for whatever reason
		for (unsigned int y = startY; y < endY; y++) // startY endY because we need bounds for threads
		{
			for (unsigned int x = startX; x < endX; x++)
			{
				float px = x + samplers[threadID].next();
				float py = y + samplers[threadID].next(); // sampler.next() because we need to
				Ray ray = scene->camera.generateRay(px, py);
				normalBuffer[y * film->width + x] = normalBuffer[y * film->width + x] + viewNormals(ray);
				albedoBuffer[y * film->width + x] = albedoBuffer[y * film->width + x] + albedo(ray); // for denoiser (feed normal + albedo info)
				Colour throughput(1.0f, 1.0f, 1.0f);
				Colour col = pathTrace(ray, throughput, 0, &samplers[threadID]);
				lightTrace(&samplers[threadID]); // enable this for light tracing!
				film->splat(px, py, col); 
				unsigned char r = (unsigned char)(col.r * 255);
				unsigned char g = (unsigned char)(col.g * 255);
				unsigned char b = (unsigned char)(col.b * 255);
				film->tonemap(x, y, r, g, b);
				canvas->draw(x, y, r, g, b);
			}
		}
	}

	void denoise() { // intel oid
		int width = film->width;
		int height = film->height;

		OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
		oidnCommitDevice(device);

		OIDNBuffer colorBuf  = oidnNewBuffer(device, width * height * 3 * sizeof(float));
		OIDNBuffer normalBuf  = oidnNewBuffer(device, width * height * 3 * sizeof(float));
		OIDNBuffer albedoBuf = oidnNewBuffer(device, width * height * 3 * sizeof(float));
		OIDNBuffer outputBuf = oidnNewBuffer(device, width * height * 3 * sizeof(float));

		OIDNFilter filter = oidnNewFilter(device, "RT"); 

		oidnSetFilterImage(filter, "color",  colorBuf, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
		oidnSetFilterImage(filter, "albedo", albedoBuf, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
		oidnSetFilterImage(filter, "normal", normalBuf, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
		oidnSetFilterImage(filter, "output", outputBuf, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
		oidnSetFilterBool(filter, "hdr", true);
		oidnCommitFilter(filter);
 // this is all boilerplate on their documentation
		float* colourPtr = (float*)oidnGetBufferData(colorBuf);
		float* normalPtr = (float*)oidnGetBufferData(normalBuf);
		float* albedoPtr = (float*)oidnGetBufferData(albedoBuf);


		// this is the only part i wrote, feed the information in the film + albedo buffer + normal buffer into the pointers
//		divide by SPP to get average at that pixel
		for (int y = 0; y < height; y++) 
			for (int x = 0; x < width; x++) {
				int i = (y * width + x) * 3;
				Colour c = film->film[y * width + x] / (float)film->SPP;
				colourPtr[i] = c.r;
				colourPtr[i+1] = c.g;
				colourPtr[i+2] = c.b; 
				Colour a = albedoBuffer[y * width + x] / (float)film->SPP;
				albedoPtr[i] = a.r;
				albedoPtr[i+1] = a.g;
				albedoPtr[i+2] = a.b;

        		Colour n = normalBuffer[y * width + x] / (float)film->SPP;
				normalPtr[i] = n.r;
				normalPtr[i+1] = n.g;
				normalPtr[i+2] = n.b;
			}

		oidnExecuteFilter(filter);
		const char* errorMessage;
		if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
		printf("Error: %s\n", errorMessage);
// feed the information in the output pointer into the film
		float* outputPtr = (float*)oidnGetBufferData(outputBuf);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int i = (y * width + x) * 3;
				film->film[y * width + x] = Colour(outputPtr[i], outputPtr[i+1], outputPtr[i+2]) * (float)film->SPP;
			}
		}

		oidnReleaseBuffer(colorBuf);
		oidnReleaseBuffer(normalBuf);
		oidnReleaseBuffer(albedoBuf);
		oidnReleaseBuffer(outputBuf);
		oidnReleaseFilter(filter);
		oidnReleaseDevice(device);
	}

	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};