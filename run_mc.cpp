#include "./mc_subroutine/mc_read_load_compute.hpp"

#include <chrono>
// const int N=10000000;
// double globalArray[N];
int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }

    auto mcObj=mc_computation(std::string(argv[1]));



    // mcObj.init_and_run();









//test row/col speed
    // arma::arma_rng::set_seed(2145);
    // int length=100;
    // arma::dvec v = arma::randu<arma::dvec>(length);
    //
    // arma::sp_dmat rand_mat=arma::sprandu< arma::sp_dmat>(length, length, 0.3);
    // // arma::dmat rand_mat=arma::randu<arma::dmat>(length, length);
    // int repeat_num=100000000;
    //
    // int  non_zero_num=static_cast<int>(std::sqrt(0.3*repeat_num));
    // std::cout<<" non_zero_num in a row="<<non_zero_num<<std::endl;
    // // int x=((-2 % length) + length) % length;
    // // std::cout<<"x="<<x<<std::endl;
    // //row
    // const auto t_row_Start{std::chrono::steady_clock::now()};
    // for (int i=0;i<repeat_num;i++)
    // {
    //     int i_mod=i%length;
    //
    //     // double tmp_loop=0;
    //     // for (int j=i_mod-non_zero_num;j<i_mod+non_zero_num+1;j++)
    //     // {
    //     //     int j_mod=((j % length) + length) % length;
    //     //     // std::cout<<"j_mod="<<j_mod<<std::endl;
    //     //     tmp_loop+=rand_mat(i_mod,j_mod)*v(j_mod);
    //     // }
    //     // double rst=arma::dot(rand_mat.row(i_mod),v);
    //     // double tmp=arma::dot(rand_mat.row(i_mod).t(),v);
    //     double tmp=arma::dot(rand_mat.col(i_mod),v);
    //     // for (int j=i-non_zero_num;j<i+non_zero_num+1;i++)
    //     // {
    //     //
    //     // }
    //     // if (i%50000==0)
    //     // {
    //     // std::cout<<"iteration "<<i<<" finished."<<std::endl;
    //     // }//end if
    // }
    //
    // const auto t_row_End{std::chrono::steady_clock::now()};
    // const std::chrono::duration<double> elapsed_secondsAll{t_row_End - t_row_Start};
    // std::cout<<"row inner product time: "<<elapsed_secondsAll.count()<<" s"<<std::endl;
    //
    //
    // //col





}
