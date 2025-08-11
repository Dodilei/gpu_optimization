#define PI 3.14159

__forceinline__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__forceinline__ __device__ float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__forceinline__ __device__ float3 operator*(const float& a, const float3& b)
{
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__forceinline__ __device__ float4 operator*(const float& a, const float4& b)
{
    return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

__forceinline__ __device__ float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

__forceinline__ __device__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__forceinline__ __device__ float3 operator/(const float& a, const float3& b)
{
    return make_float3(a/b.x, a/b.y, a/b.z);
}

__forceinline__ __device__ float3 operator/(const float3& a, const float& b)
{
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__forceinline__ __device__ float internal_sum(const float3& a)
{
    return a.x+a.y+a.z;
}

__forceinline__ __device__ float internal_sum(const float4& a)
{
    return a.x+a.y+a.z+a.w;
}

__forceinline__ __device__ float4 powers4(const float& a)
{
    return make_float4(1.0f, a, a*a, a*a*a);
}

__forceinline__ __device__ float3 powers3(const float& a)
{
    return make_float3(1.0f, a, a*a);
}

extern "C" {
__global__ void stage_aerodinamica(
  float* output,
  const int* num_vars,
  const float* scale_params,
  const float* const_params,
  const float4 wcl_coeffs_2d,
  const float4 wcd_coeffs_2d,
  const float alpha_var_2d,
  const float aL0,
  const float a1,
  const float a2,
  const int N
) {
    constexpr float phi = 1.1;
    constexpr float swet_ratio = 2.62f;
    constexpr float width_fuse_norm = 0.14/2.4;
    const float k_fuse = 1.0f - 2*pow(width_fuse_norm, 2);
    const float fuse_ratio = 1.0f - width_fuse_norm;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int remaining_idx = idx;

    float* params = new float[N];
    // Loop backwards from the last parameter (least significant) to the first.
    #pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        // Get the number of variations for the current dimension
        int current_num_vars = num_vars[i];

        // Calculate the individual index for this dimension using modulo
        int individual_index = remaining_idx % current_num_vars;

        // Calculate and store the final scaled value
        // We write all N parameters for a given idx contiguously
        params[i] = const_params[i] + individual_index * scale_params[i];

        // Update the remaining index for the next (more significant) dimension
        remaining_idx /= current_num_vars;
    }

    const float AR = params[0];
    const float afil = params[1];

    float4 CLW;
    float4 CDW;

    // TODO conferir incidência da asa
    {
      const float f_lambda = 0.0524f * pow(afil, 4) - 0.15f * pow(afil, 3) + 0.1659f * pow(afil, 2) - 0.0706f * afil + 0.0119f;
      const float theo_eff = 1.0f/(1.0f+f_lambda*AR);

      const float alpha_var_ratio = 1.0f / (1.0f + (alpha_var_2d / (PI * theo_eff * AR)));
      const float4 wcl_coeffs_3d = alpha_var_ratio * wcl_coeffs_2d;


      const float CD_0 = 0.0055f * swet_ratio;
      const float4 CDK = wcd_coeffs_2d + make_float4(CD_0,0.0f,0.0f,0.0f);

      const float Q = 1.0f / (theo_eff * k_fuse);
      const float4 KI = 0.38*CDK + make_float4(Q,0.0f,0.0f,0.0f);

      CLW.x = internal_sum(wcl_coeffs_3d*powers4(aL0));
      CDW.x = internal_sum((CDK + phi*CLW.x*CLW.x*KI)*powers4(aL0));

      CLW.y = internal_sum(wcl_coeffs_3d*powers4(0));
      CDW.y = internal_sum((CDK + phi*CLW.x*CLW.x*KI)*powers4(0));

      CLW.z = internal_sum(wcl_coeffs_3d*powers4(a1));
      CDW.z = internal_sum((CDK + phi*CLW.x*CLW.x*KI)*powers4(a1));

      CLW.w = internal_sum(wcl_coeffs_3d*powers4(a2));
      CDW.w = internal_sum((CDK + phi*CLW.x*CLW.x*KI)*powers4(a2));
    }

    CLW = fuse_ratio*CLW;

    output[idx] = CDW.w;
}
}
