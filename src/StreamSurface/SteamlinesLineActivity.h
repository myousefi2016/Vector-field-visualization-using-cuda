#pragma once
#include "Activity.h"
#include "CudaHelper.h"
#include "Utils.h"
#include "CudaMathHelper.h"

namespace mf {

	template<typename Sorry = int>
	class TStreamlinesLineActivity : public Activity {

	protected:
		VectorFieldInfo m_vfInfo;
		bool m_allocated;
		bool m_useRk4;
		double m_dt;
		uint m_linesCount;
		uint m_linePairsCount;
		float3 m_startPt;
		float3 m_endPt;
		uint m_maxSamples;
		bool m_useTubes;
		float m_tubeRadius;
		uint m_geometrySampling;
		uint m_maxGeometrySamples;
                double m_sphereRadius;
		bool m_streamSurface;
		bool m_streamSurfaceLines;
		uint m_stremasurfaceSlicesCount;
		bool m_wingLine;

		static const uint m_verticesPerTubeSlice = 5;
		static const uint m_facesPerTubeSegment = 5 * 2;


		GLuint m_verticesVbo;
		cudaGraphicsResource* m_cudaVerticesVboResource;

		GLuint m_normalsVbo;
		cudaGraphicsResource* m_cudaNormalsVboResource;

		GLuint m_indicesVbo;
		cudaGraphicsResource* m_cudaIndicesVboResource;

		GLuint m_colorsVbo;
		cudaGraphicsResource* m_cudaColorsVboResource;

		GLuint m_surfaceIndicesVbo;
		cudaGraphicsResource* m_cudaSurfaceIndicesVboResource;

		GLuint m_surfaceNormalsVbo;
		cudaGraphicsResource* m_cudaSurfaceNormalsVboResource;

		uint* m_d_surfaceFacesCounts;
		uint* m_h_surfaceFacesCounts;

		uint* m_d_ptCounts;
		uint* m_h_ptCounts;

		uint m_seedCount;
		float3* m_d_seeds;
		float3* m_h_seeds;

		uint2* h_linePairs;
		uint2* d_inLinePairs;
		uint2* d_outLinePairs;

		uint* d_outLinePairsIndex;
		uint* d_outLinesIndex;


		float3* d_vertices;
		uint3* d_indices;
		float3* d_normals;
		float3* d_colors;

		uint3* d_surfaceIndices;
		float3* d_surfaceNormals;


	public:
		TStreamlinesLineActivity(const VectorFieldInfo& vfInfo)
				: m_vfInfo(vfInfo)
				, m_allocated(false)
				, m_useRk4(true)
				, m_useTubes(false)
				, m_dt(1.0 / 256.0)
				, m_linesCount(256)
				, m_linePairsCount(0)
				, m_tubeRadius(1.0f)
				, m_geometrySampling(4)
				, m_streamSurface(false)
				, m_streamSurfaceLines(true)
				, m_stremasurfaceSlicesCount(0)
				, m_startPt(make_float3(0, 0, 0))
				, m_endPt(make_float3(0, vfInfo.realSize.y, 0))
				, m_maxSamples(8192)
				, m_maxGeometrySamples((uint)-1)  // will be properly initialized in ctor body
                                , m_sphereRadius(1.0f)
				, m_verticesVbo(NULL)
				, m_cudaVerticesVboResource(nullptr)
				, m_normalsVbo(NULL)
				, m_cudaNormalsVboResource(nullptr)
				, m_indicesVbo(NULL)
				, m_cudaIndicesVboResource(nullptr)
				, m_colorsVbo(NULL)
				, m_cudaColorsVboResource(nullptr)
				, m_surfaceIndicesVbo(NULL)
				, m_cudaSurfaceIndicesVboResource(nullptr)
				, m_surfaceNormalsVbo(NULL)
				, m_cudaSurfaceNormalsVboResource(nullptr)
				, m_d_surfaceFacesCounts(nullptr)
				, m_h_surfaceFacesCounts(nullptr)
				, m_d_ptCounts(nullptr)
				, m_h_ptCounts(nullptr)
				, m_seedCount(0)
				, m_d_seeds(nullptr)
				, m_h_seeds(nullptr)
				, h_linePairs(nullptr)
				, d_inLinePairs(nullptr)
				, d_outLinePairs(nullptr)
				, d_outLinePairsIndex(nullptr)
				, d_outLinesIndex(nullptr)
				, d_vertices(nullptr)
				, d_indices(nullptr)
				, d_normals(nullptr)
				, d_colors(nullptr)
				, d_surfaceIndices(nullptr)
				, d_surfaceNormals(nullptr)
		{
			applyGeometrySampling();
			allocate();
			recompute();
		}

		virtual ~TStreamlinesLineActivity() {
			free();
		}



		virtual void recompute() {
			assert(m_allocated);

			auto t1 = std::chrono::high_resolution_clock::now();

			size_t bytesCount;
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVerticesVboResource));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &bytesCount, m_cudaVerticesVboResource));

			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaColorsVboResource));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytesCount, m_cudaColorsVboResource));

			seedAsSphere(m_linesCount, m_sphereRadius, m_startPt);

			if (m_useTubes) {
				checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaIndicesVboResource));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_indices, &bytesCount, m_cudaIndicesVboResource));

				checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaNormalsVboResource));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_normals, &bytesCount, m_cudaNormalsVboResource));
			}

			bool doComputeSurface = m_streamSurface && !m_useTubes;

		        runKernel(m_h_seeds, m_linesCount);
			 if (doComputeSurface) {
			  m_linePairsCount = m_linesCount - 1;
			   for (uint i = 0; i < m_linePairsCount; ++i) {
			    h_linePairs[i].x = i;
			    h_linePairs[i].y = i + 1;
			   }
			 checkCudaErrors(cudaMemcpy(d_inLinePairs, h_linePairs, sizeof(uint2) * m_linePairsCount, cudaMemcpyHostToDevice));
			}

			if (doComputeSurface) {
				computeSurface();
			}

			if (m_useTubes) {
				checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaNormalsVboResource));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaIndicesVboResource));
			}


			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaColorsVboResource));
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVerticesVboResource));

			d_vertices = nullptr;
			d_indices = nullptr;
			d_normals = nullptr;
			d_colors = nullptr;

			auto t2 = std::chrono::high_resolution_clock::now();
			m_lastDuration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
		}

		virtual void drawControls(float& i, float incI) {

			std::stringstream ss;

			ss.str("");
			ss << "[+] [-] Time step: " << m_dt;
			drawString(10, ++i * incI, 0, ss.str());

                        ss.str("");
                        ss << "[z] [x] Sphere radius: " << m_sphereRadius;
                        drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "[q] [w] Lines count: " << m_linesCount;
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "[a] [s] Max samples: " << m_maxSamples;
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "[d] [f] Geometry sampling: " << m_geometrySampling;
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "  [e]   Use RK4: " << (m_useRk4 ? "true" : "false");
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "  [t]   Use tubes: " << (m_useTubes ? "true" : "false");
			drawString(10, ++i * incI, 0, ss.str());

			if (m_useTubes) {
				ss.str("");
				ss << "[c] [v] Tube radius: " << m_tubeRadius;
				drawString(10, ++i * incI, 0, ss.str());
			}

			ss.str("");
			ss << "  [m]   Toggle stream surface";
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "  [,]   Toggle stream surface lines";
			drawString(10, ++i * incI, 0, ss.str());


		}

		virtual bool keyboardCallback(unsigned char key) {

			switch (key) {
				case '+':
					m_dt *= 2;
					break;
				case '-':
					m_dt /= 2;
					break;
				case 'q':
					m_linesCount *= 2;
					allocate();
					break;
				case 'w':
					if (m_linesCount > 2) {
						m_linesCount /= 2;
						allocate();
					}
					break;
				case 'a':
					m_maxSamples *= 2;
					applyGeometrySampling();
					allocate();
					break;
				case 's':
					if (m_maxSamples > 2) {
						m_maxSamples /= 2;
						applyGeometrySampling();
						allocate();
					}
					break;
				case 'c':
					m_tubeRadius *= 1.25;
					break;
				case 'v':
					m_tubeRadius /= 1.25;
					break;
                                case 'z':
                                        m_sphereRadius *= 1.25;
                                        break;
                                case 'x':
                                        m_sphereRadius /= 1.25;
                                        break;
				case 'd':
					++m_geometrySampling;
					applyGeometrySampling();
					allocate();
					break;
				case 'f':
					if (m_geometrySampling > 1) {
						--m_geometrySampling;
						applyGeometrySampling();
						allocate();
					}
					break;
				case 'e':
					m_useRk4 ^= true;
					break;
				case 't':
					m_useTubes ^= true;
					allocate();
					break;
				case 'm':
					m_streamSurface ^= true;
					allocate();
					break;
				case ',':
					m_streamSurfaceLines ^= true;
					break;
				default:
					return false;
			}

			recompute();
			return true;
		}


		virtual bool motionCallback(int /*x*/, int y, int /*dx*/, int /*dy*/, int /*screenWidth*/, int screenHeight, int mouseButtonsState) {

                        
			if (mouseButtonsState == (1 << GLUT_RIGHT_BUTTON)) {
				float ptX = (1.0f - ((float)y / screenHeight)) * m_vfInfo.realSize.x;
				m_startPt.x = ptX;
				m_endPt.x = ptX;
				recompute();
				return true;
			}
			else if (mouseButtonsState == ((1 << GLUT_LEFT_BUTTON) | (1 << GLUT_RIGHT_BUTTON))) {
			        float ptY = ((float)y / screenHeight) * m_vfInfo.realSize.y;
				m_startPt.y = ptY;
				m_endPt.y = ptY;
				recompute();
				return true;
			}

			return false;
		}

		virtual void displayCallback() {
			assert(m_allocated);

			if (m_linesCount == 0) {
				return;
			}

			glBindBuffer(GL_ARRAY_BUFFER, m_verticesVbo);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, NULL);

			glBindBuffer(GL_ARRAY_BUFFER, m_colorsVbo);
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, 0, NULL);

			//std::cout << "============================" << std::endl;

			if (m_useTubes) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesVbo);

				glBindBuffer(GL_ARRAY_BUFFER, m_normalsVbo);
				glEnableClientState(GL_NORMAL_ARRAY);
				glNormalPointer(GL_FLOAT, 0, NULL);

				glEnable(GL_COLOR_MATERIAL);
				glEnable(GL_LIGHTING);

				uint indicesPerTube = m_facesPerTubeSegment * (m_maxGeometrySamples - 1);
				for (size_t i = 0; i < m_linesCount; ++i) {
					if (m_h_ptCounts[i] >= 2) {
						assert(m_h_ptCounts[i] <= m_maxGeometrySamples);
						//std::cout << "Drawing tubes " << m_h_ptCounts[i] << std::endl;
						glDrawElements(GL_TRIANGLES, 3 * m_facesPerTubeSegment * (m_h_ptCounts[i] - 1), GL_UNSIGNED_INT, (void*)(i * indicesPerTube * 3 * sizeof(GLuint)));
					}
				}

				glDisable(GL_COLOR_MATERIAL);
				glDisable(GL_LIGHTING);

				glDisableClientState(GL_NORMAL_ARRAY);
			}
			else {

				if (m_streamSurface) {
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIndicesVbo);

					glBindBuffer(GL_ARRAY_BUFFER, m_surfaceNormalsVbo);
					glEnableClientState(GL_NORMAL_ARRAY);
					glNormalPointer(GL_FLOAT, 0, NULL);

					glEnable(GL_COLOR_MATERIAL);
					glEnable(GL_LIGHTING);

					uint facesPerLine = 2 * m_maxGeometrySamples - 2;
					for (uint i = 0; i < m_stremasurfaceSlicesCount; ++i) {
						assert(m_h_surfaceFacesCounts[i] <= 2 * m_maxGeometrySamples - 2);
						if (m_h_surfaceFacesCounts[i] > 0) {
							glDrawElements(GL_TRIANGLES, 3 * m_h_surfaceFacesCounts[i], GL_UNSIGNED_INT, (void*)(i * facesPerLine * 3 * sizeof(GLuint)));
						}
					}

					glDisable(GL_COLOR_MATERIAL);
					glDisable(GL_LIGHTING);

					glDisableClientState(GL_NORMAL_ARRAY);
					//std::cout << "Drawing " << m_h_surfaceFacesCounts[0] << " faces" << std::endl;
					//glDrawElements(GL_TRIANGLES, 3 * m_h_surfaceFacesCounts[0], GL_UNSIGNED_INT, NULL/*(void*)(facesPerLine * 3 * sizeof(GLuint))*/);
				}

				if (!m_streamSurface || m_streamSurfaceLines) {
					for (uint i = 0; i < m_linesCount; ++i) {
						if (m_h_ptCounts[i] >= 2) {
							assert(m_h_ptCounts[i] <= m_maxGeometrySamples);
							//std::cout << "Drawing lines " << m_h_ptCounts[i] << std::endl;
							glDrawArrays(GL_LINE_STRIP, i * m_maxGeometrySamples, m_h_ptCounts[i]);
						}
					}
				}
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}


	protected:

		void applyGeometrySampling() {
			m_maxGeometrySamples = m_maxSamples / m_geometrySampling + 1;
		}

		void seedAsSphere(uint count, float radius, float3 origin) {
                         
                        float3 randomSeed;
			for (uint i = 0; i < count; ++i) {
                                randomSeed = make_float3(origin.x + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/radius)), 
                                                         origin.y + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/radius)), 
                                                         origin.z + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/radius)));
				m_h_seeds[i] = randomSeed;
			}

		}


		void runKernel(float3* h_seeds, uint seedsCount, uint offsetLines = 0) {
			if (h_seeds != nullptr) {
				checkCudaErrors(cudaMemcpy(m_d_seeds + offsetLines, h_seeds + offsetLines, seedsCount * sizeof(float3), cudaMemcpyHostToDevice));
			}

			if (m_useTubes) {
				uint verticesOffset = offsetLines * m_verticesPerTubeSlice * m_maxGeometrySamples;
				uint facesOffset = offsetLines * m_facesPerTubeSegment * (m_maxGeometrySamples - 1);
				runStreamtubesLineKernel(m_d_seeds + offsetLines, seedsCount, m_dt, m_maxSamples, m_vfInfo.cudaDataSize, m_vfInfo.volumeCoordSpaceMult, m_tubeRadius, m_useRk4, m_geometrySampling,
					d_vertices + verticesOffset, m_d_ptCounts + offsetLines, d_indices + facesOffset, d_normals + verticesOffset, d_colors + verticesOffset);
			}
			else {
				uint verticesOffset = offsetLines * m_maxGeometrySamples;
				runStreamlinesLineKernel(m_d_seeds + offsetLines, seedsCount, m_dt, m_maxSamples, m_vfInfo.cudaDataSize, m_vfInfo.volumeCoordSpaceMult, m_useRk4, m_geometrySampling,
					d_vertices + verticesOffset, m_d_ptCounts + offsetLines, d_colors + verticesOffset);
			}

			checkCudaErrors(cudaMemcpy(m_h_ptCounts + offsetLines, m_d_ptCounts + offsetLines, seedsCount * sizeof(uint), cudaMemcpyDeviceToHost));
		}

		void computeSurface() {

			for (size_t i = 0; i < m_linesCount; ++i) {
				m_h_surfaceFacesCounts[i] = 0;
			}

			size_t bytesCount;
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaSurfaceIndicesVboResource));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_surfaceIndices, &bytesCount, m_cudaSurfaceIndicesVboResource));

			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaSurfaceNormalsVboResource));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_surfaceNormals, &bytesCount, m_cudaSurfaceNormalsVboResource));

			uint verticesPerSample = m_useTubes ? m_verticesPerTubeSlice : 1;
			uint verticesPerLine = verticesPerSample * m_maxGeometrySamples;

			//std::cout << "Running surface for " << pairsCount << " pairs" << std::endl;
			runLineStreamSurfaceKernel(d_inLinePairs, m_linePairsCount, d_vertices, verticesPerLine, m_d_ptCounts, d_surfaceIndices, m_d_surfaceFacesCounts, d_surfaceNormals);
			m_stremasurfaceSlicesCount = m_linePairsCount;

			checkCudaErrors(cudaMemcpy(m_h_surfaceFacesCounts, m_d_surfaceFacesCounts, m_linePairsCount * sizeof(uint), cudaMemcpyDeviceToHost));

			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaSurfaceNormalsVboResource));
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaSurfaceIndicesVboResource));

		}

		void allocate() {
			if (m_allocated) {
				free();
			}

			m_allocated = true;

			if (m_useTubes) {
				uint verticesCount = m_verticesPerTubeSlice * m_maxGeometrySamples * m_linesCount;
				checkCudaErrors(createCudaSharedVbo(&m_verticesVbo, GL_ARRAY_BUFFER, sizeof(float3) * verticesCount, &m_cudaVerticesVboResource));
				checkCudaErrors(createCudaSharedVbo(&m_colorsVbo, GL_ARRAY_BUFFER, sizeof(float3) * verticesCount, &m_cudaColorsVboResource));
				checkCudaErrors(createCudaSharedVbo(&m_normalsVbo, GL_ARRAY_BUFFER, sizeof(float3) * verticesCount, &m_cudaNormalsVboResource));
				uint facesCount = m_facesPerTubeSegment * (m_maxGeometrySamples - 1) * m_linesCount;
				checkCudaErrors(createCudaSharedVbo(&m_indicesVbo, GL_ELEMENT_ARRAY_BUFFER, facesCount * sizeof(uint3), &m_cudaIndicesVboResource));
			}
			else {
				checkCudaErrors(createCudaSharedVbo(&m_verticesVbo, GL_ARRAY_BUFFER, m_linesCount * m_maxGeometrySamples * sizeof(float3), &m_cudaVerticesVboResource));
				checkCudaErrors(createCudaSharedVbo(&m_colorsVbo, GL_ARRAY_BUFFER, m_linesCount * m_maxGeometrySamples * sizeof(float3), &m_cudaColorsVboResource));
			}

			if (m_streamSurface) {
				checkCudaErrors(cudaMalloc((void**)&d_inLinePairs, sizeof(uint2) * m_linesCount));
			}

			if (m_streamSurface) {
				h_linePairs = new uint2[m_linesCount];

				checkCudaErrors(createCudaSharedVbo(&m_surfaceIndicesVbo, GL_ELEMENT_ARRAY_BUFFER, m_linesCount * (2 * m_maxGeometrySamples - 2) * sizeof(uint3), &m_cudaSurfaceIndicesVboResource));
				checkCudaErrors(createCudaSharedVbo(&m_surfaceNormalsVbo, GL_ELEMENT_ARRAY_BUFFER, m_linesCount * (2 * m_maxGeometrySamples - 2) * sizeof(float3), &m_cudaSurfaceNormalsVboResource));

				checkCudaErrors(cudaMalloc((void**)&m_d_surfaceFacesCounts, sizeof(uint) * m_linesCount));
				m_h_surfaceFacesCounts = new uint[m_linesCount];
				for (size_t i = 0; i < m_linesCount; ++i) {
					m_h_surfaceFacesCounts[i] = 0;
				}

				m_stremasurfaceSlicesCount = 0;
			}

			checkCudaErrors(cudaMalloc((void**)&m_d_ptCounts, sizeof(uint) * m_linesCount));
			m_h_ptCounts = new uint[m_linesCount];
			for (size_t i = 0; i < m_linesCount; ++i) {
				m_h_ptCounts[i] = 0;
			}

			checkCudaErrors(cudaMalloc((void**)&m_d_seeds, sizeof(float3) * m_linesCount));
			m_h_seeds = new float3[m_linesCount];
		}

		void free() {
			if (!m_allocated) {
				return;
			}

			checkCudaErrors(deleteCudaSharedVbo(&m_verticesVbo, m_cudaVerticesVboResource));
			m_verticesVbo = NULL;
			m_cudaVerticesVboResource = nullptr;

			checkCudaErrors(deleteCudaSharedVbo(&m_colorsVbo, m_cudaColorsVboResource));
			m_colorsVbo = NULL;
			m_cudaColorsVboResource = nullptr;

			if (m_cudaNormalsVboResource != nullptr) {
				checkCudaErrors(deleteCudaSharedVbo(&m_normalsVbo, m_cudaNormalsVboResource));
				m_normalsVbo = NULL;
				m_cudaNormalsVboResource = nullptr;
			}

			if (m_cudaIndicesVboResource != nullptr) {
				checkCudaErrors(deleteCudaSharedVbo(&m_indicesVbo, m_cudaIndicesVboResource));
				m_indicesVbo = NULL;
				m_cudaIndicesVboResource = nullptr;
			}

			if (m_cudaSurfaceIndicesVboResource != nullptr) {
				checkCudaErrors(deleteCudaSharedVbo(&m_surfaceIndicesVbo, m_cudaSurfaceIndicesVboResource));
				m_surfaceIndicesVbo = NULL;
				m_cudaSurfaceIndicesVboResource = nullptr;
			}

			if (m_cudaSurfaceNormalsVboResource != nullptr) {
				checkCudaErrors(deleteCudaSharedVbo(&m_surfaceNormalsVbo, m_cudaSurfaceNormalsVboResource));
				m_surfaceNormalsVbo = NULL;
				m_cudaSurfaceNormalsVboResource = nullptr;
			}

			if (h_linePairs != nullptr) {
				delete[] h_linePairs;
				h_linePairs = nullptr;
			}

			if (d_inLinePairs != nullptr) {
				checkCudaErrors(cudaFree((void*)d_inLinePairs));
				d_inLinePairs = nullptr;
			}

			if (d_outLinePairs != nullptr) {
				checkCudaErrors(cudaFree((void*)d_outLinePairs));
				d_outLinePairs = nullptr;
			}

			if (d_outLinePairsIndex != nullptr) {
				checkCudaErrors(cudaFree((void*)d_outLinePairsIndex));
				d_outLinePairsIndex = nullptr;
			}

			if (d_outLinesIndex != nullptr) {
				checkCudaErrors(cudaFree((void*)d_outLinesIndex));
				d_outLinesIndex = nullptr;
			}

			if (m_d_surfaceFacesCounts) {
				checkCudaErrors(cudaFree((void*)m_d_surfaceFacesCounts));
				m_d_surfaceFacesCounts = nullptr;
				delete[] m_h_surfaceFacesCounts;
				m_h_surfaceFacesCounts = nullptr;
			}

			checkCudaErrors(cudaFree((void*)m_d_ptCounts));
			m_d_ptCounts = nullptr;
			delete[] m_h_ptCounts;
			m_h_ptCounts = nullptr;

			checkCudaErrors(cudaFree((void*)m_d_seeds));
			m_d_seeds = nullptr;
			delete[] m_h_seeds;
			m_h_seeds = nullptr;

			m_allocated = false;
		}

	};

	typedef TStreamlinesLineActivity<> StreamlinesLineActivity;

}
