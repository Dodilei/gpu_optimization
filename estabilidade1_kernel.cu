__forceinline__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__forceinline__ __device__ float3 operator*(const float& a, const float3& b)
{
    return make_float3(a*b.x, a*b.y, a*b.z);
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
__global__ void stage_estabilidade2(
  float* output,
  const int* num_vars, // AR, afil, seh_ratio, XACH_norm
  const float* scale_params,
  const float* const_params,
  const float4 wcl_coeffs,
  const float3 wcm_coeffs,
  const float dCLWda,
  const float dCLHda_eh,
  const float dCMWda,
  const float XACW_norm,
  const int N
) {
    constexpr float ME = 0.08;
    constexpr float nh0 = 0.9;
    constexpr float YH_norm = 0.3;
    constexpr float prop_pressure_ratio = 1.5;
    constexpr float propwash_ratio = 0.08;
    constexpr float eh_affected_ratio = 0.5;

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
    const float seh_ratio = params[2];
    const float XACH_norm = params[3];

    const float KA = (1.0f/AR) - (1.0f/(1.0f+pow(AR,1.7f)));
    const float Kl = (10.0f - 3.0f*afil)/7.0f;                                 
    const float KH = (1.0f - (YH_norm/AR))*pow((2.0f*(XACH_norm-XACW_norm)/AR),(-1.0f/3.0f));  

    const float downwash_ratio = pow(4.44f*(KA*Kl*KH),1.19f);                          

    const float eh_aoa_rate_ratio = (1 - downwash_ratio - propwash_ratio);     

    const float eh_dpressure_ratio = nh0*(1 + eh_affected_ratio*prop_pressure_ratio);

    const float cl_eff_ratio = eh_dpressure_ratio * seh_ratio * eh_aoa_rate_ratio;     
    const float dCLHda = cl_eff_ratio * dCLHda_eh;                                    

    const float XNP_norm = (dCMWda + dCLWda*XACW_norm + dCLHda*XACH_norm)/(dCLWda + dCLHda);

    const float XCG_norm = XNP_norm - ME;

    output[idx] = XCG_norm;
}
}
