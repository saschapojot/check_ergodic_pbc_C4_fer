//
// Created by adada on 22/4/2025.
//

#include "mc_read_load_compute.hpp"


///
/// @param x proposed value
/// @param y current value
/// @param a left end of interval
/// @param b right end of interval
/// @param epsilon half length
/// @return proposal probability S(x|y)
double mc_computation::S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon)
{
    if (a < y and y < a + epsilon)
    {
        return 1.0 / (y - a + epsilon);
    }
    else if (a + epsilon <= y and y < b - epsilon)
    {
        return 1.0 / (2.0 * epsilon);
    }
    else if (b - epsilon <= y and y < b)
    {
        return 1.0 / (b - y + epsilon);
    }
    else
    {
        std::cerr << "value out of range." << std::endl;
        std::exit(10);
    }
}


double mc_computation::acceptanceRatio_uni(const arma::dvec& arma_vec_curr,
                                           const arma::dvec& arma_vec_next, const int& flattened_ind,
                                           const double& UCurr, const double& UNext)
{
    double numerator = -this->beta * UNext;
    double denominator = -this->beta * UCurr;
    double R = std::exp(numerator - denominator);

    double S_curr_next = S_uni(arma_vec_curr(flattened_ind), arma_vec_next(flattened_ind),
                               dipole_lower_bound, dipole_upper_bound, h);

    double S_next_curr = S_uni(arma_vec_next(flattened_ind), arma_vec_curr(flattened_ind),
                               dipole_lower_bound, dipole_upper_bound, h);

    double ratio = S_curr_next / S_next_curr;

    if (std::fetestexcept(FE_DIVBYZERO))
    {
        std::cout << "Division by zero exception caught." << std::endl;
        std::exit(15);
    }
    if (std::isnan(ratio))
    {
        std::cout << "The result is NaN." << std::endl;
        std::exit(15);
    }
    R *= ratio;

    return std::min(1.0, R);
}


///
/// @param flattened_ind
/// @param Px_arma_vec
/// @param Py_arma_vec
/// @return
double mc_computation::H1(const int& flattened_ind, const arma::dvec& Px_arma_vec, const arma::dvec& Py_arma_vec)
{
    double px_n0n1 = Px_arma_vec(flattened_ind);

    double py_n0n1 = Py_arma_vec(flattened_ind);


    double squared_px_n0n1 = std::pow(px_n0n1, 2.0);
    double squared_py_n0n1 = std::pow(py_n0n1, 2.0);

    double val1=alpha1*
        (std::pow(squared_px_n0n1-squared_py_n0n1,2.0)-4.0*squared_px_n0n1*squared_py_n0n1);

    double val2=alpha2*px_n0n1*py_n0n1
                *(squared_px_n0n1-squared_py_n0n1);

    double val3=alpha3*(squared_px_n0n1+squared_py_n0n1);
    return val1+val2+val3;
}


///
/// @param x
/// @param leftEnd
/// @param rightEnd
/// @param eps
/// @return return a value within distance eps from x, on the open interval (leftEnd, rightEnd)
double mc_computation::generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                                  const double& eps)
{
    double xMinusEps = x - eps;
    double xPlusEps = x + eps;

    double unif_left_end = xMinusEps < leftEnd ? leftEnd : xMinusEps;
    double unif_right_end = xPlusEps > rightEnd ? rightEnd : xPlusEps;

    //    std::random_device rd;
    //    std::ranlux24_base e2(rd());

    double unif_left_end_double_on_the_right = std::nextafter(unif_left_end, std::numeric_limits<double>::infinity());


    std::uniform_real_distribution<> distUnif(unif_left_end_double_on_the_right, unif_right_end);
    //(unif_left_end_double_on_the_right, unif_right_end)

    double xNext = distUnif(e2);
    return xNext;
}

void mc_computation::proposal_uni(const arma::dvec& arma_vec_curr, arma::dvec& arma_vec_next,
                                  const int& flattened_ind)
{
    double dp_val_new = this->generate_uni_open_interval(arma_vec_curr(flattened_ind), dipole_lower_bound,
                                                         dipole_upper_bound, h);
    arma_vec_next = arma_vec_curr;
    arma_vec_next(flattened_ind) = dp_val_new;


}


void mc_computation::save_array_to_pickle(const std::shared_ptr<double[]>& ptr, int size, const std::string& filename)
{
    using namespace boost::python;
    namespace np = boost::python::numpy;

    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }
        np::initialize(); // Initialize NumPy
    }

    try
    {
        // Import the pickle module
        object pickle = import("pickle");
        object pickle_dumps = pickle.attr("dumps");

        // Convert C++ array to NumPy array using shared_ptr
        np::ndarray numpy_array = np::from_data(
            ptr.get(), // Use shared_ptr's raw pointer
            np::dtype::get_builtin<double>(), // NumPy data type (double)
            boost::python::make_tuple(size), // Shape of the array (1D array)
            boost::python::make_tuple(sizeof(double)), // Strides
            object() // Optional base object
        );

        // Serialize the NumPy array using pickle.dumps
        object serialized_array = pickle_dumps(numpy_array);

        // Extract the serialized data as a string
        std::string serialized_str = extract<std::string>(serialized_array);

        // Write the serialized data to a file
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Debug output (optional)
        // std::cout << "Array serialized and written to file successfully." << std::endl;
    }
    catch (const error_already_set&)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred." << std::endl;
    } catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}


void mc_computation::load_pickle_data(const std::string& filename, std::shared_ptr<double[]>& data_ptr,
                                      std::size_t size)
{
    // Initialize Python and NumPy
    Py_Initialize();
    np::initialize();


    try
    {
        // Use Python's 'io' module to open the file directly in binary mode
        py::object io_module = py::import("io");
        py::object file = io_module.attr("open")(filename, "rb"); // Open file in binary mode

        // Import the 'pickle' module
        py::object pickle_module = py::import("pickle");

        // Use pickle.load to deserialize from the Python file object
        py::object loaded_data = pickle_module.attr("load")(file);

        // Close the file
        file.attr("close")();

        // Check if the loaded object is a NumPy array
        if (py::extract<np::ndarray>(loaded_data).check())
        {
            np::ndarray np_array = py::extract<np::ndarray>(loaded_data);

            // Convert the NumPy array to a Python list using tolist()
            py::object py_list = np_array.attr("tolist")();

            // Ensure the list size matches the expected size
            ssize_t list_size = py::len(py_list);
            if (static_cast<std::size_t>(list_size) > size)
            {
                throw std::runtime_error("The provided shared_ptr array size is smaller than the list size.");
            }

            // Copy the data from the Python list to the shared_ptr array
            for (ssize_t i = 0; i < list_size; ++i)
            {
                data_ptr[i] = py::extract<double>(py_list[i]);
            }
        }
        else
        {
            throw std::runtime_error("Loaded data is not a NumPy array.");
        }
    }
    catch (py::error_already_set&)
    {
        PyErr_Print();
        throw std::runtime_error("Python error occurred.");
    }
}


///
/// @param n0
/// @param n1
/// @return flatenned index
int mc_computation::double_ind_to_flat_ind(const int& n0, const int& n1)
{
    return n0 * N1 + n1;
}
void mc_computation::init_Px_Py()
{
    std::string name;

    std::string Px_inFileName, Py_inFileName;
    if (this->flushLastFile == -1)
    {
        name = "init";

        Px_inFileName = out_Px_path + "/Px_" + name + ".pkl";

        Py_inFileName = out_Py_path + "/Py_" + name + ".pkl";
        this->load_pickle_data(Px_inFileName, Px_init, N0 * N1);
        this->load_pickle_data(Py_inFileName, Py_init, N0 * N1);
    }//end flushLastFile==-1
    else
    {
        name="flushEnd"+std::to_string(this->flushLastFile);
        Px_inFileName=out_Px_path+"/"+name+".Px.pkl";

        Py_inFileName=out_Py_path+"/"+name+".Py.pkl";
        //load Px
        this->load_pickle_data(Px_inFileName,Px_all_ptr,sweepToWrite * N0 * N1);
        //copy last N0*N1 elements of to Px_init
        std::memcpy(Px_init.get(),Px_all_ptr.get()+N0*N1*(sweepToWrite-1),
            N0*N1*sizeof(double));
        //load Py
        this->load_pickle_data(Py_inFileName,Py_all_ptr,sweepToWrite * N0 * N1);
        //copy last N0*N1 elements of to Py_init
        std::memcpy(Py_init.get(),Py_all_ptr.get()+N0*N1*(sweepToWrite-1),
            N0*N1*sizeof(double));
    }
}

int mc_computation::mod_direction0(const int&m0)
{

    return ((m0 % N0) + N0) % N0;

}

int mc_computation::mod_direction1(const int&m1)
{return ((m1 % N1) + N1) % N1;
}


///
/// @param flattened_ind_center (flattened) index of dipole to be updated
/// @param ind_neighbor  index of dipole around the center dipole (0..7)
/// @param Px_arma_vec
/// @param Py_arma_vec
/// @return interaction energy
double mc_computation::H2(const int& flattened_ind_center,const int& ind_neighbor, const arma::dvec& Px_arma_vec, const arma::dvec& Py_arma_vec )
{
    // std::cout<<"flattened_ind_center="<<flattened_ind_center<<std::endl;
    // std::cout<<Px_arma_vec.n_elem<<std::endl;
    // Px_arma_vec.print("Px_arma_vec");
    double px_n0n1=Px_arma_vec(flattened_ind_center);
    double py_n0n1=Py_arma_vec(flattened_ind_center);

    // std::cout<<"px_n0n1="<<px_n0n1<<", py_n0n1="<<py_n0n1<<std::endl;


    int flattened_ind_one_neighbor=this->flattened_ind_neighbors[flattened_ind_center][ind_neighbor];
    double px_m0m1=Px_arma_vec(flattened_ind_one_neighbor);
    double py_m0m1=Py_arma_vec(flattened_ind_one_neighbor);

    // std::cout<<"flattened_ind_center="<<flattened_ind_center
    // <<", ind_neighbor="<<ind_neighbor
    // <<", flattened_ind_one_neighbor="<<flattened_ind_one_neighbor<<std::endl;
    // std::cout<<"px_m0m1="<<px_m0m1<<", py_m0m1="<<py_m0m1<<std::endl;
    double rx_n0n1m0m1=flattened_ind_neighbors_r[flattened_ind_center][ind_neighbor][0];
    double ry_n0n1m0m1=flattened_ind_neighbors_r[flattened_ind_center][ind_neighbor][1];

    double r2_tmp=flattened_ind_neighbors_r2[flattened_ind_center][ind_neighbor];
    double r4_tmp=flattened_ind_neighbors_r4[flattened_ind_center][ind_neighbor];
    // std::cout<<"rx_n0n1m0m1="<<rx_n0n1m0m1<<", ry_n0n1m0m1="<<ry_n0n1m0m1<<std::endl;
    // std::cout<<"r2_tmp="<<r2_tmp<<", r4_tmp="<<r4_tmp<<std::endl;
    double val1=J/r2_tmp*(px_n0n1*px_m0m1+py_n0n1*py_m0m1);
    double val2=-2.0*J/r4_tmp*(px_n0n1*rx_n0n1m0m1+py_n0n1*ry_n0n1m0m1)
    *(px_m0m1*rx_n0n1m0m1+py_m0m1*ry_n0n1m0m1);
    // std::cout<<"val1="<<val1<<", val2="<<val2<<std::endl;
    return val1+val2;
}


void mc_computation::init_flattened_ind_and_neighbors()
{
//this function initializes each point's neigboring indices(flattened), neigboring vectors, distance^2, distance^4
    this->flattened_ind_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_neighbors_r=std::vector<std::vector<std::vector<double>>>(N0*N1,std::vector<std::vector<double>>());
this->flattened_ind_neighbors_r2=std::vector<std::vector<double>>(N0*N1,std::vector<double>());
this->flattened_ind_neighbors_r4=std::vector<std::vector<double>>(N0*N1,std::vector<double>());

for (int n0=0;n0<N0;n0++)
{
    for (int n1=0;n1<N1;n1++)
    {   std::vector<std::vector<double>> neighors_r_tmp;
        // std::cout<<"======================="<<std::endl;
        int point_curr_flattened=this->double_ind_to_flat_ind(n0,n1);
        for (const auto&vec_nghbrs:this->neigbors)
        {
            int diff_direc0=vec_nghbrs[0];
            int diff_direc1=vec_nghbrs[1];
            neighors_r_tmp.push_back({diff_direc0*a,diff_direc1*a});
            double r2Tmp=std::pow(diff_direc0*a,2.0)+std::pow(diff_direc1*a,2.0);
            double r4Tmp=std::pow(r2Tmp,2.0);
            flattened_ind_neighbors_r2[point_curr_flattened].push_back(r2Tmp);
            flattened_ind_neighbors_r4[point_curr_flattened].push_back(r4Tmp);
            int m0=n0+diff_direc0;
            int m1=n1+diff_direc1;
            // std::cout<<"n0="<<n0<<", n1="<<n1<<", m0="<<m0<<", m1="<<m1<<std::endl;

            int m0_mod=mod_direction0(m0);
            int m1_mod=mod_direction1(m1);
            int flattened_ngb=double_ind_to_flat_ind(m0_mod,m1_mod);
            flattened_ind_neighbors[point_curr_flattened].push_back(flattened_ngb);
            // std::cout<<"point_curr_flattened="<<point_curr_flattened
            // <<", flattened_ngb="<<flattened_ngb<<std::endl;
            // std::cout<<"..."<<std::endl;
        }//end neighbors
        // std::cout<<"point_curr_flattened="<<point_curr_flattened<<std::endl;
        // print_vector(flattened_ind_neighbors[point_curr_flattened]);
        flattened_ind_neighbors_r[point_curr_flattened]=neighors_r_tmp;
        // std::cout<<"neigbnoring r:"<<std::endl;
        // for (const auto &vec:flattened_ind_neighbors_r[point_curr_flattened])
        // {
        //     print_vector(vec);
        // }
        // std::cout<<"current r2:"<<std::endl;
        // print_vector(flattened_ind_neighbors_r2[point_curr_flattened]);
        // std::cout<<"current r4:"<<std::endl;
        // print_vector(flattened_ind_neighbors_r4[point_curr_flattened]);
        // std::cout<<"xxxx"<<std::endl;
    }//end for n1
}//end for n0

}
void mc_computation::construct_neighbors_1_point()
{
    //constrcuting neighbors
    for (int n0=-Nx;n0<Nx+1;n0++)
    {
        for (int n1=-Ny;n1<Ny+1;n1++)
        {
            if (n0==0 and n1==0)
            {
                continue;
            }else
            {
                this->neigbors.push_back({n0,n1});
            }
        }//end for n1
    }//end for n0
    //print neighbors
    std::cout<<"print neighbors:"<<std::endl;
    for (const auto & vec:neigbors)
    {
        print_vector(vec);
        double tmp=std::pow(static_cast<double>(vec[0]),2.0)
        +std::pow(static_cast<double>(vec[1]),2.0);
        this->neighbors_r.push_back({static_cast<double>(vec[0])*a,static_cast<double>(vec[1])*a});
        this->neighbors_r2.push_back(tmp);
        this->neighbors_r4.push_back(std::pow(tmp,2.0));
    }
    std::cout<<"print neighbors_r2:"<<std::endl;
    print_vector(neighbors_r2);
    std::cout<<"print neighbors_r4:"<<std::endl;
    print_vector(neighbors_r4);

    std::cout<<"print neighbors_r:"<<std::endl;
    for (const auto& vec:neighbors_r)
    {
        print_vector(vec);
    }
}



void mc_computation::
init_and_run()
{
    this->init_Px_Py();
    this->construct_neighbors_1_point();
    this->init_flattened_ind_and_neighbors();
    this->execute_mc(Px_init,Py_init,newFlushNum);
    // arma::dvec Px_arma_vec_tmp(Px_init.get(),N0*N1);
    // arma::dvec Py_arma_vec_tmp(Py_init.get(),N0*N1);
    // double e=H_tot(Px_arma_vec_tmp,Py_arma_vec_tmp);
    // std::cout<<"e="<<e<<std::endl;

    // arma::dvec Px_arma_vec_tmp_rot(N0*N1, arma::fill::zeros);
    // arma::dvec Py_arma_vec_tmp_rot(N0*N1, arma::fill::zeros);
    // rotated_Px_Py(Px_arma_vec_tmp,Py_arma_vec_tmp,
        // Px_arma_vec_tmp_rot,Py_arma_vec_tmp_rot);
    // double e_rot=H_tot(Px_arma_vec_tmp_rot,Py_arma_vec_tmp_rot);
    // std::cout<<"e_rot="<<e_rot<<std::endl;

    // int flat_ind=17;
    // double UCurr=0;
    // double UNext=0;
    // this->H_update_Py(flat_ind,Px_arma_vec_tmp,Py_arma_vec_tmp,Px_arma_vec_tmp_rot,UCurr,UNext);
}


double mc_computation::H_tot(const arma::dvec& Px_arma_vec, const arma::dvec& Py_arma_vec)
{
// std::cout<<"flattened_ind_neighbors:"<<std::endl;
//     for (const auto& vec:flattened_ind_neighbors)
//     {
//         print_vector(vec);
//     }
    double part1=0;
    for (int n0=0;n0<N0;n0++)
    {
        for (int n1=0;n1<N1;n1++)
        {
            int flattened_ind=double_ind_to_flat_ind(n0,n1);
            part1+=H1(flattened_ind,Px_arma_vec,Py_arma_vec);
        }//end for n1
    }//end n0

    double part2=0;
    for (int n0=0;n0<N0;n0++)
    {
        for (int n1=0;n1<N1;n1++)
        {
            int flattened_ind=double_ind_to_flat_ind(n0,n1);
            const auto& vec_ind_neighbor_tmp=flattened_ind_neighbors[flattened_ind];
            // std::cout<<"flattened_ind="<<flattened_ind<<", vec_ind_neighbor_tmp:"<<std::endl;
            // print_vector(vec_ind_neighbor_tmp);
            // std::cout<<"vec_ind_neighbor_tmp.size()="<<vec_ind_neighbor_tmp.size()<<std::endl;
            for (int j=0;j<vec_ind_neighbor_tmp.size();j++)
            {
                part2+=H2(flattened_ind,j,Px_arma_vec,Py_arma_vec);
            }//end j
        }//end n1
    }//end n0
    // std::cout<<"part1="<<part1<<std::endl;
    // std::cout<<"part2="<<part2<<std::endl;

    return part1+part2*0.5;
}

void mc_computation::C_next(const int &n0,const int &n1, int &n0_next,int &n1_next)
{

    n0_next=1-n1;
    n1_next=n0;
}


void mc_computation::C_prev(const int &n0,const int &n1, int &n0_next,int &n1_next)
{
    n0_next=n1;
    n1_next=1-n0;
}


///
/// @param Px_arma_vec
/// @param Py_arma_vec
/// @param rotated_Px_arma_vec Px after C4 rotation
/// @param rotated_Py_arma_vec Py after C4 rotation
void mc_computation::rotated_Px_Py(const arma::dvec &Px_arma_vec,const arma::dvec& Py_arma_vec,
                   arma::dvec& rotated_Px_arma_vec, arma::dvec& rotated_Py_arma_vec)
{
for (int n0=0;n0<N0;n0++)
{
    for (int n1=0;n1<N1;n1++)
    {
        arma::dvec p_tmp=p_2_p_tilde(n0,n1,Px_arma_vec,Py_arma_vec);
        int flat_ind=double_ind_to_flat_ind(n0,n1);
        rotated_Px_arma_vec[flat_ind]=p_tmp(0);
        rotated_Py_arma_vec[flat_ind]=p_tmp(1);
    }//end n1
}//end for n0
}

///
/// @param n0 x ind of dipole to be rotated
/// @param n1 y ind of dipole to be rotated
/// @return
arma::dvec mc_computation::p_2_p_tilde(const int &n0,const int & n1,
    const arma::dvec &Px_arma_vec,const arma::dvec& Py_arma_vec)
{

    // int flat_ind=double_ind_to_flat_ind(n0,n1);

    int n0_prev=0;
    int n1_prev=0;
    C_prev(n0,n1,n0_prev,n1_prev);

    int m0=mod_direction0(n0_prev);
    int m1=mod_direction1(n1_prev);
    // std::cout<<"n0="<<n0<<", n1="<<n1<<", m0="<<m0<<", m1="<<m1<<std::endl;
    int flat_ind_m0m1=double_ind_to_flat_ind(m0,m1);

    double px_m0m1=Px_arma_vec[flat_ind_m0m1];
    double py_m0m1=Py_arma_vec[flat_ind_m0m1];
    arma::dvec p_tmp={px_m0m1,py_m0m1};

   return U4*p_tmp;

}



//local update of Px
void mc_computation::H_update_Px(const int &flattened_ind,const arma::dvec &Px_arma_vec_curr,
    const arma::dvec& Py_arma_vec_curr,const arma::dvec& Px_arma_vec_next,
    double& UCurr, double& UNext)
{
double H1_curr=H1(flattened_ind,Px_arma_vec_curr,Py_arma_vec_curr);

    double H1_next=H1(flattened_ind,Px_arma_vec_next,Py_arma_vec_curr);

    double part2_curr=0;
    double part2_next=0;
// std::cout<<"flattened_ind="<<flattened_ind<<std::endl;
    // std::cout<<"neigbors:"<<std::endl;//[-1,-1],[-1,0],...,[1,1]
    // for (const auto & vec: neigbors)
    // {
        // print_vector(vec);
    // }
    //part2 curr
    int neighbor_num=neigbors.size();
    // std::cout<<"neighbor_num="<<neighbor_num<<std::endl;
    for (int j=0;j<neighbor_num;j++)
    {
        part2_curr+=H2(flattened_ind,j,Px_arma_vec_curr,Py_arma_vec_curr);
    }//end j
    UCurr=H1_curr+0.5*part2_curr;
    // std::cout<<"UCurr="<<UCurr<<std::endl;

    //part2 next
    for (int j=0;j<neighbor_num;j++)
    {
        part2_next+=H2(flattened_ind,j,Px_arma_vec_next,Py_arma_vec_curr);
    }//end j
    UNext=H1_next+0.5*part2_next;
    // std::cout<<"UNext="<<UNext<<std::endl;
}

//local update of Py
void mc_computation::H_update_Py(const int &flattened_ind,const arma::dvec &Px_arma_vec_curr,
    const arma::dvec& Py_arma_vec_curr,
   const arma::dvec& Py_arma_vec_next, double& UCurr, double& UNext)
{
double H1_curr=H1(flattened_ind,Px_arma_vec_curr,Py_arma_vec_curr);
    double H1_next=H1(flattened_ind,Px_arma_vec_curr,Py_arma_vec_next);
    double part2_curr=0;
    double part2_next=0;
    //part2 curr
    int neighbor_num=neigbors.size();
    // std::cout<<"neighbor_num="<<neighbor_num<<std::endl;
    for (int j=0;j<neighbor_num;j++)
    {
        part2_curr+=H2(flattened_ind,j,Px_arma_vec_curr,Py_arma_vec_curr);
    }//end j
    UCurr=H1_curr+0.5*part2_curr;
    //part2 next
    for (int j=0;j<neighbor_num;j++)
    {
        part2_next+=H2(flattened_ind,j,Px_arma_vec_curr,Py_arma_vec_next);
    }//end j

    UNext=H1_next+0.5*part2_next;
    // std::cout<<"UCurr="<<UCurr<<std::endl;
    // std::cout<<"UNext="<<UNext<<std::endl;
}


void mc_computation::execute_mc_one_sweep(arma::dvec& Px_arma_vec_curr,
                              arma::dvec& Py_arma_vec_curr,
                              double& U_base_value,
                              arma::dvec& Px_arma_vec_next,
                              arma::dvec& Py_arma_vec_next)
{
    double UCurr=0;
    double UNext = 0;

    U_base_value=this->H_tot(Px_arma_vec_curr,Py_arma_vec_curr);
    //update Px
    for (int i = 0; i < N0 * N1; i++)
    {
        int flattened_ind = unif_in_0_N0N1(e2);
        this->proposal_uni(Px_arma_vec_curr, Px_arma_vec_next, flattened_ind);
        this->H_update_Px(flattened_ind,Px_arma_vec_curr,
            Py_arma_vec_curr,Px_arma_vec_next,UCurr,UNext);
        double r = this->acceptanceRatio_uni(Px_arma_vec_curr, Px_arma_vec_next,
                                            flattened_ind, UCurr, UNext);
        double u = distUnif01(e2);
        if (u <= r)
        {
            Px_arma_vec_curr = Px_arma_vec_next;
            // UCurr = UNext;
            U_base_value+=UNext-UCurr;
            // std::cout<<"U_base_value="<<U_base_value<<std::endl;
        } //end of accept-reject
    }//end updating Px

    //update Py
    for (int i = 0; i < N0 * N1; i++)
    {
        int flattened_ind = unif_in_0_N0N1(e2);
        this->proposal_uni(Py_arma_vec_curr, Py_arma_vec_next, flattened_ind);
        this->H_update_Py(flattened_ind,Px_arma_vec_curr,Py_arma_vec_curr,Py_arma_vec_next,UCurr,UNext);
        double r = this->acceptanceRatio_uni(Py_arma_vec_curr, Py_arma_vec_next, flattened_ind, UCurr, UNext);
        double u = distUnif01(e2);
        if (u <= r)
        {
            Py_arma_vec_curr = Py_arma_vec_next;
            // UCurr = UNext;
            U_base_value+=UNext-UCurr;
            // std::cout<<"U_base_value="<<U_base_value<<std::endl;
        } //end of accept-reject
    }//end updating Py
}

void mc_computation::execute_mc(const std::shared_ptr<double[]>& Px_vec,
                   const std::shared_ptr<double[]>& Py_vec,const int& flushNum)
{
    arma::dvec Px_arma_vec_curr(Px_vec.get(), N0 * N1);
    arma::dvec Px_arma_vec_next(N0 * N1, arma::fill::zeros);

    arma::dvec Py_arma_vec_curr(Py_vec.get(), N0 * N1);
    arma::dvec Py_arma_vec_next(N0 * N1, arma::fill::zeros);
    double U_base_value=-12345;

    int flushThisFileStart=this->flushLastFile+1;
    for (int fls = 0; fls < flushNum; fls++)
    {
        const auto tMCStart{std::chrono::steady_clock::now()};
        for (int swp = 0; swp < sweepToWrite*sweep_multiple; swp++)
        {
            this->execute_mc_one_sweep(Px_arma_vec_curr,Py_arma_vec_curr,
               U_base_value,Px_arma_vec_next, Py_arma_vec_next);
            if(swp%sweep_multiple==0)
            {
                int swp_out=swp/sweep_multiple;
                this->U_data_all_ptr[swp_out]=U_base_value;
                std::memcpy(Px_all_ptr.get()+swp_out*N0*N1,Px_arma_vec_curr.memptr(),N0*N1*sizeof(double));
                std::memcpy(Py_all_ptr.get()+swp_out*N0*N1,Py_arma_vec_curr.memptr(),N0*N1*sizeof(double));
            }//end save to array
        }//end sweep for
        int flushEnd=flushThisFileStart+fls;
        std::string fileNameMiddle =  "flushEnd" + std::to_string(flushEnd);

        std::string out_U_PickleFileName = out_U_path+"/" + fileNameMiddle + ".U.pkl";
        std::string out_Px_PickleFileName=out_Px_path+"/"+fileNameMiddle+".Px.pkl";

        std::string out_Py_PickleFileName=out_Py_path+"/"+fileNameMiddle+".Py.pkl";
        //save U
        this->save_array_to_pickle(U_data_all_ptr,sweepToWrite,out_U_PickleFileName);

        //save Px
        this->save_array_to_pickle(Px_all_ptr,sweepToWrite*N0*N1,out_Px_PickleFileName);
        //save Py
        this->save_array_to_pickle(Py_all_ptr,sweepToWrite*N0*N1,out_Py_PickleFileName);
        const auto tMCEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
        std::cout << "flush " + std::to_string(flushEnd)  + ": "
                  << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;

    }//end flush for loop
    std::cout << "mc executed for " << flushNum << " flushes." << std::endl;

}