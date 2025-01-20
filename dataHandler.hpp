#pragma once

#include <Eigen/Core>
using Eigen::Index;

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

class IOManager {
	private:
		fs::path    m_dataRootDir;
		std::string m_customHeader = "";
		std::string m_shWithStr    = "";

	public:
		IOManager() = default;
		IOManager(std::string rootDir) : m_dataRootDir{rootDir} {}

		fs::path const& rootDir() const { return m_dataRootDir; };
		fs::path&       rootDir() { return m_dataRootDir; };

		std::string const& shWith() const { return m_shWithStr; };
		std::string&       shWith() { return m_shWithStr; };

		template<class... Args>
		fs::path     generate_filepath(Args...) const;
		std::string& custom_header() { return m_customHeader; }
		std::string  header(std::string shWithStr, double energyRange, double gE,
		                    double maxDiff) const {
            std::stringstream headers("");
            headers << std::setprecision(6) << std::scientific << std::showpos;
            headers << m_customHeader << "\n"
                    << "#shWidth = " << shWithStr << "\n"
                    << "#energyRange = " << energyRange << "\n"
                    << "#gE          = " << gE << "\n"
                    << "#maxDiff     = " << maxDiff << "\n\n"
                    << "# 1.(Normalized energy) 2.(shellDim) 3.4. ...(quasi ETH measures)\n";
            return headers.str();
		};

		template<class Vector_, class Matrix_, class iVector_>
		bool save_ResultsToFile(fs::path outFilePath, Vector_ const& eigVals,
		                        Matrix_ const& ETHmeasure, iVector_ const& dimShell,
		                        double const maxDiff) const {
			auto const gE          = *std::min_element(eigVals.begin(), eigVals.end());
			auto const energyRange = *std::max_element(eigVals.begin(), eigVals.end()) - gE;

			fs::create_directories(outFilePath.parent_path());
			if(!fs::exists(outFilePath.parent_path())) {
				std::cerr << "Error: Failed to create a directory " << outFilePath.parent_path()
				          << std::endl;
				std::exit(EXIT_FAILURE);
			}
			std::ofstream resFile(outFilePath);
			if(!resFile) {
				std::cerr << "Error: Can't open a file (" << outFilePath << ")" << std::endl;
				std::exit(EXIT_FAILURE);
			}

			resFile << this->header(m_shWithStr, energyRange, gE, maxDiff);
			resFile << std::setprecision(18) << std::scientific << std::showpos;
			for(auto j = 0; j < eigVals.size(); ++j) {
				resFile << (eigVals[j] - gE) / energyRange << " " << dimShell[j] << " ";
				for(Index m = 1; m < ETHmeasure.cols(); ++m) { resFile << ETHmeasure(j, m) << " "; }
				resFile << "\n";
			}
			resFile.close();

			return true;
		};

		template<class Vector_, class Matrix_, class iVector_>
		bool save_ResultsToFile(Vector_ const& eigVals, Matrix_ const& ETHmeasure, Index sysSize,
		                        iVector_ const& dimShell, double const maxDiff) const {
			fs::path outFilePath = this->generate_filepath(sysSize);
			return this->save_ResultsToFile(outFilePath, eigVals, ETHmeasure, dimShell, maxDiff);
		};

		template<class Vector_, class Matrix_, class iVector_>
		bool save_ResultsToFile(Vector_ const& eigVals, Matrix_ const& ETHmeasure, Index sysSize,
		                        Index N, iVector_ const& dimShell, double const maxDiff) const {
			fs::path outFilePath = this->generate_filepath(sysSize, N);
			return this->save_ResultsToFile(outFilePath, eigVals, ETHmeasure, dimShell, maxDiff);
		};

		template<class Vector_, class Matrix_>
		bool load_Data(fs::path inFilePath, Vector_& eigVals, Matrix_& ETHmeasure, Index N) const {
			(void)N;
			std::ifstream inFile(inFilePath);
			if(!inFile) {
				std::cout << "### File \"" << inFilePath << "\" does not exist. Create a new one."
				          << std::endl;
				return true;
			}

			// std::cout << "# Reading from file: " << inFilePath << std::endl;
			auto        fpos = inFile.tellg();
			std::string str;
			while(std::getline(inFile, str)) {
				if(str.size() == 0 || str.at(0) == '#') {
					fpos = inFile.tellg();
					continue;
				}
				inFile.seekg(fpos);
				break;
			}
			// std::cout << "# Point 1 " << std::endl;

			Index dim  = 0;
			Index cols = -1, colsPrev = 0;
			while(std::getline(inFile, str)) {
				str.erase(str.find_last_not_of("\t\n\v\f\r ") + 1);  // Remove trailing whitespaces
				if(str.size() == 0 || str.at(0) == '#') continue;
				cols = -1;
				std::stringstream ss(str);
				while(!ss.eof()) {
					ss >> str;
					// std::cout << str << "\n";
					++cols;
				}
				if(colsPrev != 0) assert(cols == colsPrev);
				colsPrev = cols;
				++dim;
			}
			inFile.close();
			inFile.open(inFilePath);
			inFile.seekg(fpos);
			// std::cout << "# Point 2: dim = " << dim << ", cols = " << cols << std::endl;

			eigVals     = Vector_::Constant(dim, std::nan(""));
			ETHmeasure  = Matrix_::Constant(dim, cols, std::nan(""));
			Index alpha = 0;
			while(std::getline(inFile, str)) {
				str.erase(str.find_last_not_of("\t\n\v\f\r ") + 1);  // Remove trailing whitespaces
				if(str.size() == 0 || str.at(0) == '#') continue;
				std::stringstream ss(str);
				double            energy;
				Index             shDim;

				std::string word;
				ss >> word;
				if(word.find("nan") == std::string::npos) {
					// std::cout << __FILE__ << ":" << __LINE__ << " word = " << word << "(END)"
					//           << std::endl;
					energy = std::stod(word);
				}
				else { energy = std::nan(""); }

				ss >> shDim;
				eigVals(alpha) = energy;
				// std::cout << "# str = " << str << std::endl;
				// std::cout << "#\t energy = " << energy << ", shDim = " << shDim << std::endl;
				for(auto m = 1; m < ETHmeasure.cols(); ++m) {
					std::string word;
					ss >> word;
					// std::cout << word << "\n";
					if(word.find("nan") == std::string::npos) {
						// std::cout << __FILE__ << ":" << __LINE__ << ", m = " << m
						//           << " word = " << word << "(END)" << std::endl;
						ETHmeasure(alpha, m) = std::stod(word);
					}
					else { ETHmeasure(alpha, m) = std::nan(""); }
				}
				// std::cout << std::endl;
				++alpha;
			}
			inFile.close();

			// std::cout << "#(END) " << __PRETTY_FUNCTION__ << std::endl;
			return true;
		};

		template<class Vector_, class Matrix_>
		bool load_Data(Vector_& eigVals, Matrix_& ETHmeasure, Index const sysSize) const {
			fs::path inFilePath = this->generate_filepath(sysSize);
			return this->load_Data(inFilePath, eigVals, ETHmeasure, sysSize);
		};
		template<class Vector_, class Matrix_>
		bool load_Data(Vector_& eigVals, Matrix_& ETHmeasure, Index const sysSize,
		               Index const N) const {
			fs::path inFilePath = this->generate_filepath(sysSize, N);
			return this->load_Data(inFilePath, eigVals, ETHmeasure, N);
		};

		template<class Matrix_>
		bool load_Data(Matrix_& ETHmeasure, Index const sysSize) const {
			Eigen::ArrayXd eigVals;
			return this->load_Data(eigVals, ETHmeasure, sysSize);
		}
		template<class Matrix_>
		bool load_Data(Matrix_& ETHmeasure, Index const sysSize, Index const N) const {
			Eigen::ArrayXd eigVals;
			return this->load_Data(eigVals, ETHmeasure, sysSize, N);
		}
};