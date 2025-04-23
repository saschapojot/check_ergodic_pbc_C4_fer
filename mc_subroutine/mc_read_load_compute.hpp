//
// Created by adada on 22/4/2025.
//

#ifndef MC_READ_LOAD_COMPUTE_HPP
#define MC_READ_LOAD_COMPUTE_HPP
#include <armadillo>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cfenv> // for floating-point exceptions
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = boost::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;

constexpr double PI = M_PI;

class mc_computation
{
public:
    mc_computation(const std::string& cppInParamsFileName): e2(std::random_device{}()), distUnif01(0.0, 1.0)
    {
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open())
        {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty())
            {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            //read T
            if (paramCounter == 0)
            {
                iss >> T;
                if (T <= 0)
                {
                    std::cerr << "T must be >0" << std::endl;
                    std::exit(1);
                } //end if
                std::cout << "T=" << T << std::endl;
                this->beta = 1.0 / T;
                std::cout << "beta=" << beta << std::endl;
                paramCounter++;
                continue;
            } //end T
            // read a
            if (paramCounter == 1)
            {
                iss >> a;
                if (a <= 0)
                {
                    std::cerr << "a must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "a=" << a << std::endl;
                paramCounter++;
                continue;
            } //end a

            //read J
            if (paramCounter == 2)
            {
                iss >> J;
                std::cout << "J=" << J << std::endl;
                paramCounter++;
                continue;
            } //end J

            //read N
            if (paramCounter == 3)
            {
                iss >> N0;
                N1 = N0;
                if (N0 <= 0)
                {
                    std::cerr << "N must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "N0=N1=" << N0 << std::endl;
                paramCounter++;
                continue;
            } //end N
            //read q
            if (paramCounter == 4)
            {
                iss >> q;
                if (q <= 0)
                {
                    std::cerr << "q must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "q=" << q << std::endl;
                paramCounter++;
                continue;
            } //end q
            //read alpha1
            if (paramCounter == 5)
            {
                iss >> alpha1;
                std::cout << "alpha1=" << alpha1 << std::endl;
                paramCounter++;
                continue;
            } //end alpha1

            //read alpha2
            if (paramCounter == 6)
            {
                iss >> alpha2;
                std::cout << "alpha2=" << alpha2 << std::endl;
                paramCounter++;
                continue;
            } //end alpha2

            //read alpha3
            if (paramCounter == 7)
            {
                iss >> alpha3;
                std::cout << "alpha3=" << alpha3 << std::endl;
                paramCounter++;
                continue;
            } //end alpha3

            //read sweepToWrite
            if (paramCounter==8)
            {
                iss >> sweepToWrite;
                if (sweepToWrite <= 0)
                {
                    std::cerr << "sweepToWrite must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "sweepToWrite=" << sweepToWrite << std::endl;
                paramCounter++;
                continue;
            }//end sweepToWrite
            //read newFlushNum
            if (paramCounter == 9)
            {
                iss >> newFlushNum;
                if (newFlushNum <= 0)
                {
                    std::cerr << "newFlushNum must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "newFlushNum=" << newFlushNum << std::endl;
                paramCounter++;
                continue;
            } //end newFlushNum
            //read flushLastFile
            if (paramCounter == 10)
            {
                iss >> flushLastFile;
                std::cout << "flushLastFile=" << flushLastFile << std::endl;
                paramCounter++;
                continue;
            } //end flushLastFile
            //read TDirRoot
            if (paramCounter == 11)
            {
                iss >> TDirRoot;
                std::cout << "TDirRoot=" << TDirRoot << std::endl;
                paramCounter++;
                continue;
            } //end TDirRoot
            //read U_dipole_dataDir
            if (paramCounter == 12)
            {
                iss >> U_dipole_dataDir;
                std::cout << "U_dipole_dataDir=" << U_dipole_dataDir << std::endl;
                paramCounter++;
                continue;
            } //end U_dipole_dataDir
            //read h
            if (paramCounter == 13)
            {
                iss >> h;
                if (h <= 0)
                {
                    std::cerr << "h must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "h=" << h << std::endl;
                paramCounter++;
                continue;
            } //end h
            //read sweep_multiple
            if (paramCounter == 14)
            {
                iss >> sweep_multiple;
                if (sweep_multiple <= 0)
                {
                    std::cerr << "sweep_multiple must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "sweep_multiple=" << sweep_multiple << std::endl;
                paramCounter++;
                continue;
            } //end sweep_multiple

            //read N_half_side
            if (paramCounter == 15)
            {
                iss >>Nx;
                Ny=Nx;
                if (Nx <= 0)
                {
                    std::cerr << "Nx must be >0" << std::endl;
                    std::exit(1);
                }

                std::cout << "Nx=" << Nx << std::endl;
                std::cout << "Ny=" << Ny << std::endl;
                paramCounter++;
                continue;
            }
            //read init_path
            if (paramCounter == 16)
            {
                iss>>init_path;
                std::cout<<"init_path="<<init_path<<std::endl;
                paramCounter++;
                continue;
            }//end init_path
        }//end while
        //allocate memory for data
        try
        {
            this->U_data_all_ptr = std::shared_ptr<double[]>(new double[sweepToWrite],
                                                             std::default_delete<double[]>());
            this->Px_all_ptr = std::shared_ptr<double[]>(new double[sweepToWrite * N0 * N1],
                                                         std::default_delete<double[]>());

            this->Py_all_ptr = std::shared_ptr<double[]>(new double[sweepToWrite * N0 * N1],
                                                         std::default_delete<double[]>());
            this->Px_init = std::shared_ptr<double[]>(new double[N0 * N1],
                                                      std::default_delete<double[]>());

            this->Py_init = std::shared_ptr<double[]>(new double[N0 * N1],
                                                      std::default_delete<double[]>());
        }
        catch (const std::bad_alloc& e)
        {
            std::cerr << "Memory allocation error: " << e.what() << std::endl;
            std::exit(2);
        } catch (const std::exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            std::exit(2);
        }

        this->out_U_path = this->U_dipole_dataDir + "/U/";
        if (!fs::is_directory(out_U_path) || !fs::exists(out_U_path))
        {
            fs::create_directories(out_U_path);
        }
        this->out_Px_path = this->U_dipole_dataDir + "/Px/";
        if (!fs::is_directory(out_Px_path) || !fs::exists(out_Px_path))
        {
            fs::create_directories(out_Px_path);
        }
        this->out_Py_path = this->U_dipole_dataDir + "/Py/";
        if (!fs::is_directory(out_Py_path) || !fs::exists(out_Py_path))
        {
            fs::create_directories(out_Py_path);
        }
        this->unif_in_0_N0N1 = std::uniform_int_distribution<int>(0, N0 * N1-1);

        this->dipole_upper_bound = 1.0 / 4.0  * a * q;
        this->dipole_lower_bound = -dipole_upper_bound;
        std::cout << "dipole_upper_bound=" << dipole_upper_bound << std::endl;
        std::cout << "dipole_lower_bound=" << dipole_lower_bound << std::endl;

        this->a_squared = std::pow(a, 2.0);
      std::cout<<"PI="<<PI<<std::endl;
        this->theta=PI/2.0;
        this->U4={{std::cos(theta),-std::sin(theta)},{std::sin(theta),std::cos(theta)}};
        U4.print("U4:");
    }//end constructor

public:
    void init_and_run();
    void execute_mc(const std::shared_ptr<double[]>& Px_vec,
                    const std::shared_ptr<double[]>& Py_vec,const int& flushNum);
    void execute_mc_one_sweep(arma::dvec& Px_arma_vec_curr,
                              arma::dvec& Py_arma_vec_curr,
                              double& U_base_value,
                              arma::dvec& Px_arma_vec_next,
                              arma::dvec& Py_arma_vec_next);
    //local update of Py
    void H_update_Py(const int &flattened_ind,const arma::dvec &Px_arma_vec_curr,
        const arma::dvec& Py_arma_vec_curr,
       const arma::dvec& Py_arma_vec_next, double& UCurr, double& UNext);
    //local update of Px
    void H_update_Px(const int &flattened_ind,const arma::dvec &Px_arma_vec_curr,
        const arma::dvec& Py_arma_vec_curr,const arma::dvec& Px_arma_vec_next,
        double& UCurr, double& UNext);
    ///
    /// @param Px_arma_vec
    /// @param Py_arma_vec
    /// @param rotated_Px_arma_vec Px after C4 rotation
    /// @param rotated_Py_arma_vec Py after C4 rotation
    void rotated_Px_Py(const arma::dvec &Px_arma_vec,const arma::dvec& Py_arma_vec,
                       arma::dvec& rotated_Px_arma_vec, arma::dvec& rotated_Py_arma_vec);
    arma::dvec p_2_p_tilde(const int &n0,const int & n1,const arma::dvec &Px_arma_vec,const arma::dvec& Py_arma_vec);

    void C_next(const int &n0,const int &n1, int &n0_next,int &n1_next);
    void C_prev(const int &n0,const int &n1, int &n0_next,int &n1_next);
    void init_flattened_ind_and_neighbors();//neighbors around (0,0)
    double H_tot(const arma::dvec& Px_arma_vec, const arma::dvec& Py_arma_vec);
    ///
    /// @param flattened_ind_center (flattened) index of dipole to be updated
    /// @param ind_neighbor  index of dipole around the center dipole (0..7)
    /// @param Px_arma_vec
    /// @param Py_arma_vec
    /// @return interaction energy
    double H2(const int& flattened_ind_center,const int& ind_neighbor, const arma::dvec& Px_arma_vec, const arma::dvec& Py_arma_vec );
    void construct_neighbors_1_point();
    int mod_direction0(const int&m0);
    int mod_direction1(const int&m1);
    void init_Px_Py();
    ///
    /// @param n0
    /// @param n1
    /// @return flatenned index
    int double_ind_to_flat_ind(const int& n0, const int& n1);
    void load_pickle_data(const std::string& filename, std::shared_ptr<double[]>& data_ptr, std::size_t size);

    void save_array_to_pickle(const std::shared_ptr<double[]>& ptr, int size, const std::string& filename);

    void proposal_uni(const arma::dvec& arma_vec_curr, arma::dvec& arma_vec_next,
                            const int& flattened_ind);
    ///
    /// @param x
    /// @param leftEnd
    /// @param rightEnd
    /// @param eps
    /// @return return a value within distance eps from x, on the open interval (leftEnd, rightEnd)
    double generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                      const double& eps);
    ///
    /// @param flattened_ind
    /// @param Px_arma_vec
    /// @param Py_arma_vec
    /// @return
    double H1(const int& flattened_ind, const arma::dvec& Px_arma_vec, const arma::dvec& Py_arma_vec);

    double acceptanceRatio_uni(const arma::dvec& arma_vec_curr,
                                  const arma::dvec& arma_vec_next, const int& flattened_ind,
                                  const double& UCurr, const double& UNext);
    ///
    /// @param x proposed value
    /// @param y current value
    /// @param a left end of interval
    /// @param b right end of interval
    /// @param epsilon half length
    /// @return proposal probability S(x|y)
    double S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon);
    template <class T>
    void print_shared_ptr(const std::shared_ptr<T>& ptr, const int& size)
    {
        if (!ptr)
        {
            std::cout << "Pointer is null." << std::endl;
            return;
        }

        for (int i = 0; i < size; i++)
        {
            if (i < size - 1)
            {
                std::cout << ptr[i] << ",";
            }
            else
            {
                std::cout << ptr[i] << std::endl;
            }
        }
    } //end print_shared_ptr
    // Template function to print the contents of a std::vector<T>
    template <typename T>
    void print_vector(const std::vector<T>& vec)
    {
        // Check if the vector is empty
        if (vec.empty())
        {
            std::cout << "Vector is empty." << std::endl;
            return;
        }

        // Print each element with a comma between them
        for (size_t i = 0; i < vec.size(); ++i)
        {
            // Print a comma before all elements except the first one
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << vec[i];
        }
        std::cout << std::endl;
    }
public:
    double T; // temperature
    double beta;
    int init_path;
    double a;
    double a_squared;
    double J;
    // double J_over_a_squared;
    int N0;
    int N1;
    int Nx;
    int Ny;

    double h; // step size
    int sweepToWrite;
    int newFlushNum;
    int flushLastFile;
    std::string TDirRoot;
    std::string U_dipole_dataDir;
    std::ranlux24_base e2;
    double q;
    double alpha1;
    double alpha2;
    double alpha3;
    std::uniform_real_distribution<> distUnif01;
    int sweep_multiple;
    std::string out_U_path;
    std::string out_Px_path;

    std::string out_Py_path;
    //data in 1 flush
    std::shared_ptr<double[]> U_data_all_ptr; //all U data
    std::shared_ptr<double[]> Px_all_ptr; //all Px data
    std::shared_ptr<double[]> Py_all_ptr; //all Py data
    //initial value
    std::shared_ptr<double[]> Px_init;
    std::shared_ptr<double[]> Py_init;
    std::uniform_int_distribution<int> unif_in_0_N0N1;

    double dipole_lower_bound;
    double dipole_upper_bound;
    arma::dmat U4;
    double theta;
std::vector<std::vector<int>> neigbors;//around (0,0)
    std::vector<std::vector<double>>neighbors_r;//around (0,0)
    std::vector<double> neighbors_r2;//around (0,0)
    std::vector<double> neighbors_r4;//around (0,0)

    std::vector<std::vector<int>> flattened_ind_neighbors;// a point (flattened index) and its neighbors(also flattened ind)
    std::vector<std::vector<std::vector<double>>>flattened_ind_neighbors_r;// a point (flattened index) and its vectors to neighbors
    std::vector<std::vector<double>> flattened_ind_neighbors_r2; //a point (flatttend index) 's distance squared to neigbors
    std::vector<std::vector<double>> flattened_ind_neighbors_r4; //a point (flatttend index) 's distance quartic to neighbors

};


#endif //MC_READ_LOAD_COMPUTE_HPP
