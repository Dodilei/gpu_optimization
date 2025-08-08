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
  const int* num_vars, // AR, b, seh_ratio, XACH_norm, ih
  const float* scale_params,
  const float* const_params,
  const float4 wcl_coeffs,
  const float3 wcm_coeffs,
  const float rho,
  const float T0,
  const float a,
  const float dCLHda_eh,
  const float XACW_norm,
  const float XCG_norm,
  const float downwash_ratio,
  const float amax,
  const float a0,
  const int N
) {
    constexpr float ef_solo_lift = 1.1;
    constexpr float ef_solo_epsilon = 1;
    constexpr float q_dec_est = 0.5*1.118*10*10;
    constexpr float nh0 = 0.9;
    constexpr float prop_pressure_ratio = 3;
    constexpr float propwash_ratio = 0.08;
    constexpr float eh_affected_ratio = 0.5;
    constexpr float ZT = 0.075;
    constexpr float ip = 2;
    constexpr float flap_dcl = ef_solo_lift*0.6*0.85*0.9*0.0785*0.9;

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
    const float b_wing = params[1];
    const float seh_ratio = params[2];
    const float XACH_norm = params[3];
    const float ih = params[4];

    const float CLW_amax = internal_sum(wcl_coeffs*powers4(amax));

    const float CMW0_amax = internal_sum(wcm_coeffs*powers3(amax));

    const float MAC = (b/AR);

    const float dCLHda_eh = ef_solo_lift*dCLHda_eh;

    const float downwash_ratio = ef_solo_epsilon*downwash_ratio;

    const float ah_amax = (amax + ih) - downwash_ratio*(amax - a0) - propwash_ratio*(amax + ip);

    const float eh_dpressure_ratio = nh0*(1.0f + eh_affected_ratio*prop_pressure_ratio);

    const float eh_eff_ratio = eh_dpressure_ratio * seh_ratio;

    const float CMLW_amax = (XCG_norm - XACW_norm)*(ef_solo_lift*CLW_amax);

    const float CMT = -(T0/q_dec_est - (2.0f*a)/rho)*(ZT/MAC);

    const float CM_amax = CMW0_amax + CMLW_amax + CMT;

    const float CMH_amax = eh_eff_ratio*(XCG_norm - XACH_norm)*dCLHda_eh*ah_amax;

    const float CMt_amax = CM_amax + CMH_amax;

    const float req_dclh = CMt_amax/(eh_eff_ratio*(XCG_norm - XACH_norm));

    const float deflection = flap_dcl*((float) (req_dclh <= 0.48f) * (1030.0f*req_dclh) + (float) (req_dclh > 0.48f) * (6250.0f*(req_dclh - 0.48f) + 494.4f));

    output[idx] = deflection;
}
}
