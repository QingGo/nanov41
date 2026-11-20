// R85: cuBLASLt FP8 wgrad (dW[N,K] = X^T @ dY) — vendor transposed-A GEMM for sm_120 training.
//
// Layouts (no operand copies for variant 0):
//   x8  is [M,K] row-major fp8; logically A = X^T [K,M] stored column-major (lda = K).
//   dy8 is [M,N] row-major fp8; logically B = dY [M,N] stored "transposed" as [N,M] column-major
//       (ldb = N), so transb = T.
//   C = dW^T is [K,N] column-major with ldc = K, which is exactly dW [N,K] row-major.
// Scaling: per-tensor fp8 scales (device float*), as required by cuBLASLt FP8.
#include <torch/extension.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace {

cublasLtHandle_t get_handle() {
  static cublasLtHandle_t h = nullptr;
  if (h == nullptr) {
    TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
  }
  return h;
}

void check(cublasStatus_t st, const char* what) {
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cuBLASLt ", what, " failed: ", int(st));
}

}  // namespace

torch::Tensor wgrad_lt(torch::Tensor xA, torch::Tensor xB, torch::Tensor sa, torch::Tensor sb,
                       int64_t variant) {
  TORCH_CHECK(xA.is_cuda() && xA.dtype() == torch::kFloat8_e4m3fn, "operands must be cuda fp8");
  TORCH_CHECK(xB.is_cuda() && xB.dtype() == torch::kFloat8_e4m3fn, "operands must be cuda fp8");
  TORCH_CHECK(xA.is_contiguous() && xB.is_contiguous(), "operands must be contiguous");
  // logical shapes are inferred from the descriptor choice below; the token count M must be
  // identical in both operands' underlying layouts, so the caller passes xb (variant 0) or xT
  // (variant 1) with the same M.
  const int M = (variant == 0) ? int(xA.size(0)) : int(xA.size(1));
  const int K = (variant == 0) ? int(xA.size(1)) : int(xA.size(0));
  const int N = (variant == 0) ? int(xB.size(1)) : int(xB.size(0));
  auto dw = torch::empty({N, K}, xA.options().dtype(torch::kBFloat16));

  auto stream = at::cuda::getCurrentCUDAStream();
  cublasLtHandle_t handle = get_handle();

  cublasLtMatmulDesc_t desc = nullptr;
  check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
  cublasOperation_t op_n = CUBLAS_OP_N, op_t = CUBLAS_OP_T;
  if (variant == 0) {
    // A = X^T [K,M] stored column-major (this is x8 [M,K] row-major, no copy);
    // B = dY [M,N] stored as [N,M] column-major (this is dy8 [M,N] row-major, no copy), transb=T.
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)), "set transa");
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)), "set transb");
  } else {
    // A = X^T [K,M] row-major == A^T [M,K] column-major, so transa=T with lda=M.
    // B = dY [M,N] column-major (caller passes dyt = dy8.t().contiguous()), transb=N.
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)), "set transa");
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)), "set transb");
  }
  TORCH_CHECK(sa.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat,
              "scales must be float32 (kept alive by the caller; the GEMM is async)");
  TORCH_CHECK(sa.is_contiguous() && sb.is_contiguous(), "scales must be contiguous");
  float* sa_p = sa.data_ptr<float>();
  float* sb_p = sb.data_ptr<float>();
  check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa_p, sizeof(float*)), "set a scale");
  check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb_p, sizeof(float*)), "set b scale");

  cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
  if (variant == 0) {
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, K, M, K);  // A: K x M col-major, lda=K
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, N, M, N);  // B: N x M col-major (transb=T), ldb=N
  } else {
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, M, K, M);  // A^T: M x K col-major, lda=M (transa=T)
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, M, N, M);  // B: M x N col-major, ldb=M
  }
  cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, K, N, K);       // C: K x N col-major == dW[N,K] row-major

  // algorithm selection (cached per shape+variant) with a scratch workspace
  static std::mutex mu;
  static std::unordered_map<long long, std::unique_ptr<cublasLtMatmulAlgo_t>> algo_cache;
  static void* workspace = nullptr;
  static size_t workspace_size = 0;
  const long long key = ((long long)variant << 48) ^ ((long long)M << 24) ^ ((long long)K << 12) ^ N;
  cublasLtMatmulAlgo_t* algo = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = algo_cache.find(key);
    if (it == algo_cache.end()) {
      cublasLtMatmulPreference_t pref = nullptr;
      cublasLtMatmulPreferenceCreate(&pref);
      size_t ws = 32ull * 1024 * 1024;
      if (workspace == nullptr) {
        cudaMalloc(&workspace, ws);
        workspace_size = ws;
      }
      cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace_size, sizeof(workspace_size));
      cublasLtMatmulHeuristicResult_t heur{};
      int got = 0;
      cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(handle, desc, Ad, Bd, Cd, Cd, pref, 1,
                                                         &heur, &got);
      cublasLtMatmulPreferenceDestroy(pref);
      auto holder = std::make_unique<cublasLtMatmulAlgo_t>();
      if (hs == CUBLAS_STATUS_SUCCESS && got > 0) {
        *holder = heur.algo;
      } else {
        std::memset(holder.get(), 0, sizeof(cublasLtMatmulAlgo_t));
      }
      it = algo_cache.emplace(key, std::move(holder)).first;
    }
    algo = it->second.get();
  }
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasLtMatmul(handle, desc, &alpha, xA.data_ptr(), Ad, xB.data_ptr(), Bd,
                                     &beta, dw.data_ptr(), Cd, dw.data_ptr(), Cd,
                                     algo, workspace, workspace_size, stream.stream());
  check(st, "Matmul");
  cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatrixLayoutDestroy(Bd);
  cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulDescDestroy(desc);
  return dw;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wgrad_lt", &wgrad_lt, "FP8 wgrad via cuBLASLt (transposed-A, no operand copies)");
}
