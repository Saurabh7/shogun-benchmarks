#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/base/init.h>
#include <shogun/io/CSVFile.h>
#include <chrono>
#include <shogun/distance/EuclideanDistance.h>



using namespace shogun;
using namespace std;
using namespace std::chrono;  

int main(int argc, char **argv)
{
  init_shogun_with_defaults();
  int32_t num_clusters=100;


  CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> ();
  CCSVFile* test_file = new CCSVFile("student.csv");
  SGMatrix<float64_t> test_m;
  test_m.load(test_file);
  features->set_feature_matrix(test_m);
  SG_REF(features);
  CEuclideanDistance* distance=new CEuclideanDistance(features, features);
  
  high_resolution_clock::time_point t1 = high_resolution_clock::now();  
  
  CKMeans* clustering=new CKMeans(num_clusters, distance);
  clustering->train(features);
  clustering->apply();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  SG_SPRINT("%d", duration);
  
  SG_UNREF(clustering);
  SG_UNREF(features);

  exit_shogun();
  return 0;

}

  

