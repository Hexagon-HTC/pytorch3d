#include <torch/script.h>

#include <tuple>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

// Compute indices of K nearest neighbors in pointcloud p2 to points
// in pointcloud p1.
//
// Args:
//    p1: FloatTensor of shape (N, P1, D) giving a batch of pointclouds each
//        containing P1 points of dimension D.
//    p2: FloatTensor of shape (N, P2, D) giving a batch of pointclouds each
//        containing P2 points of dimension D.
//    lengths1: LongTensor, shape (N,), giving actual length of each P1 cloud.
//    lengths2: LongTensor, shape (N,), giving actual length of each P2 cloud.
//    K: int giving the number of nearest points to return.
//    version: Integer telling which implementation to use.
//
// Returns:
//    p1_neighbor_idx: LongTensor of shape (N, P1, K), where
//        p1_neighbor_idx[n, i, k] = j means that the kth nearest
//        neighbor to p1[n, i] in the cloud p2[n] is p2[n, j].
//        It is padded with zeros so that it can be used easily in a later
//        gather() operation.
//
//    p1_neighbor_dists: FloatTensor of shape (N, P1, K) containing the squared
//        distance from each point p1[n, p, :] to its K neighbors
//        p2[n, p1_neighbor_idx[n, p, k], :].
std::tuple<torch::Tensor, torch::Tensor> KNearestNeighborIdxCuda(const torch::Tensor &p1, const torch::Tensor &p2, const torch::Tensor &lengths1, const torch::Tensor &lengths2,
                                                                 const int norm, const int K, int version);

// Implementation which is exposed.
std::tuple<torch::Tensor, torch::Tensor> KNearestNeighborIdx(const torch::Tensor &p1, const torch::Tensor &p2, const torch::Tensor &lengths1, const torch::Tensor &lengths2,
                                                             int64_t K, int64_t version)
{
    CHECK_CUDA(p1);
    IS_CONTIGUOUS(p1);
    CHECK_CUDA(p2);
    IS_CONTIGUOUS(p2);

    CHECK_CUDA(lengths1);
    IS_CONTIGUOUS(lengths1);
    CHECK_CUDA(lengths2);
    IS_CONTIGUOUS(lengths2);

    return KNearestNeighborIdxCuda(p1, p2, lengths1, lengths2, 2, K, version);
}

// Utility to check whether a KNN version can be used.
//
// Args:
//    version: Integer in the range 0 <= version <= 3 indicating one of our
//        KNN implementations.
//    D: Number of dimensions for the input and query point clouds
//    K: Number of neighbors to be found
//
// Returns:
//    Whether the indicated KNN version can be used.
bool KnnCheckVersion(int version, const int64_t D, const int64_t K);

static auto registry = torch::RegisterOperators("pytorch3d_knn_ops::query", &KNearestNeighborIdx);
