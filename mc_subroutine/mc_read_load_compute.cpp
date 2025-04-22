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


double mc_computation::H2(const int&n0, const int& n1, const int& ind_neighbor, const std::vector<int>& vec_neighbor,
    const arma::dvec& Px_arma_vec_curr,
                const arma::dvec& Py_arma_vec_curr)
{


}
