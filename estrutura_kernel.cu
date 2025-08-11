#define PI 3.14159f

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
__global__ void stage_estrutura(
  float* output,
  const int* num_vars, 
  const float* scale_params,
  const float* const_params,
  const float S_ev,
  const float XACW_norm,
  const float CL_amax,
  const float CL_amax_eh,
  const float XCG_norm,
  const int N
) {
    constexpr float long_perc = 0.8f;

    constexpr float phi_wing = 0.5f;
    constexpr float phi_eh = 0.5f;
    constexpr float phi_ev = 0.5f;

    constexpr float rho_tail = 1500.0f;
    constexpr float rho_wlong = 1500.0f;
    constexpr float rho_ehlong = 1500.0f;

    constexpr float K_sigma_tail = 0.0015;
    constexpr float K_sigma_wing = 0.0004f;
    constexpr float K_sigma_eh = 0.001066f;

    constexpr float K_A_tail = 2.0f*PI;
    constexpr float K_A_wlong = 2.0f*PI;
    constexpr float K_A_ehlong = 2.0f*PI;

    constexpr float t_tail = 0.0005f;
    constexpr float t_wlong = 0.0005f;
    constexpr float t_ehlong = 0.001f;

    constexpr float mass_wservo = 0.012f;
    constexpr float mass_ehservo = 0.020f;
    constexpr float mass_evservo = 0.009f;

    constexpr float mass_fuse = 0.800f;
    constexpr float xmin_fuse = 0.0f;
    constexpr float xmax_fuse = 0.4f;

    constexpr int n_comps = 5;
    //                               Bat. P, Bat. C, ESC,   Motor+H, Fuse
    constexpr float comp_mass[5] = { 0.467,  0.150,  0.050,  0.450,  mass_fuse};
    constexpr float comp_xmin[5] = {-0.200, -0.250, -0.300, -0.350,  xmin_fuse};
    constexpr float comp_xmax[5] = { 0.200,  0.600,  0.000,  0.100,  xmax_fuse};

    float comp_xmid[5];
    #pragma unroll
    for (int comp = 0; comp < n_comps; ++comp) {
      comp_xmid[comp] = 0.5f*(comp_xmin[comp] + comp_xmax[comp]);
    }

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
    const float AR_eh = params[1];
    const float b_wing = params[2];
    const float seh_ratio = params[3];
    const float XACH_norm = params[4];


    const float MAC = b_wing/AR;
    const float S_wing = MAC*b_wing;

    const float S_eh = S_wing*seh_ratio;
    const float b_eh = pow(S_eh*AR_eh, 0.5f);

    const float XCG_alvo = XCG_norm*MAC;


    const float mass_wlong = long_perc*b_wing*rho_wlong*K_A_wlong*pow(t_wlong, 0.5f)*K_sigma_wing*pow(b_wing*S_wing*CL_amax, 0.5f);
    const float mass_wing = phi_wing*S_wing + mass_wlong + 2*mass_wservo;
    const float x_wing = XACW_norm*MAC;

    const float mass_ehlong = b_eh*rho_ehlong*K_A_ehlong*pow(t_ehlong, 0.5f)*K_sigma_eh*pow(b_eh*S_eh*CL_amax_eh, 0.5f);
    const float mass_eh = phi_eh*S_eh + mass_ehlong + mass_ehservo;
    const float x_eh = XACH_norm*MAC;

    const float mass_ev = phi_ev*S_ev + 2*mass_evservo;
    const float x_ev = x_eh;

    const float mass_tail = (XACH_norm-0.8f)*MAC*rho_tail*K_A_tail*pow(t_tail, 0.5f)*K_sigma_tail*pow(XACH_norm*MAC*S_eh*CL_amax_eh, 0.5f);
    const float x_tail = 0.5f*(XACH_norm-0.8f)*MAC + 0.8f*MAC;


    // P. ref. -> asa
    // Comps fixos: asa, eh, ev, tail
    const float m0 = mass_wing + mass_eh + mass_ev + mass_tail;
    const float cg0 = (mass_wing*x_wing + mass_eh*x_eh + mass_ev*x_ev + mass_tail*x_tail)/m0;

    float m_total = m0;
    #pragma unroll 
    for (int comp = 0; comp < n_comps; ++comp) {
      m_total += comp_mass[comp];
    }

    float XCG_atual = cg0*(m0/m_total);
    #pragma unroll
    for (int comp = 0; comp < n_comps; ++comp) {
      XCG_atual += comp_xmid[comp]*(comp_mass[comp]/m_total);
    }

    #pragma unroll
    for (int comp = 0; comp < n_comps; ++comp) {
      float x_cmin = comp_xmin[comp];
      float x_cmax = comp_xmax[comp];
      float x_c0 = comp_xmid[comp];
      float m_c = comp_mass[comp];

      float x_c = x_c0 + (m_total / m_c) * (XCG_alvo - XCG_atual);

      bool ismin = x_c < x_cmin;
      bool ismax = x_c > x_cmax;
      x_c = (float) (ismin) * (x_cmin) + (float) (ismax) * (x_cmax) + (float) (! (ismin || ismax)) * x_c;

      XCG_atual += (x_c - x_c0) * (m_c / m_total);
    }

    XCG_atual = XCG_atual / ((float) (abs(XCG_atual - XCG_alvo) < 0.01f));

    output[idx] = XCG_atual;
}
}
