#define PI 3.14159f
#define DEG2RAD 0.017453f
#define IDEG2RAD 57.2957f

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

template <typename LambdaType>
__forceinline__ __device__ void RK4(LambdaType dfunc, float3& y0, const float& xf)
{
    float dx = xf/100.0f;
    // TODO maybe add pragma unroll here if 100 can be lowered
    for (int i = 0; i < 100; i++) {
        float3 y_prev = y0;
        float3 k1 = dfunc(y_prev               );
        float3 k2 = dfunc(y_prev + (dx/2.0f)*k1);
        float3 k3 = dfunc(y_prev + (dx/2.0f)*k2);
        float3 k4 = dfunc(y_prev +  dx      *k3);

        y0 = y_prev + (dx/6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);
    }
}



extern "C" {
__global__ void stage_single_kernel(
  float* output,
  const int* num_vars,
  const float* scale_params,
  const float* const_params,
  const float4 wcl_coeffs_2d,
  const float4 wcd_coeffs_2d,
  const float4 hcl_coeffs_2d,
  const float4 hcd_coeffs_2d,
  const float3 wcm_coeffs_2d,
  const float XACW_norm,
  const float dCLWda_2d,
  const float dCMWda_2d,
  const float dCLHdaH_2d,
  const float a0L,
  const float amax,
  const int N,
  bool* criteria_1,
  bool* criteria_2,
  bool* criteria_3
) {
    // Parâmetros de sistema
    constexpr float rho = 1.118f;
    constexpr float g_acc = 9.806f;
    constexpr float mu = 0.04f;

    // Parâmetros propulsivos
    constexpr float T0 = 40.0f;
    constexpr float a = 0.1f;
    constexpr float dW_max = 600.0f;

    // Parâmetros construtivos
    constexpr float swet_ratio = 2.62f;
    constexpr float width_fuse_norm = 0.14f/2.4f;
    constexpr float sev_ratio = 0.054f; //TODO
    constexpr float a1 = 11;
    constexpr float Y_a2_norm = 0.2;

    // Efeito solo
    constexpr float efsolo_lift = 1.1f;
    constexpr float efsolo_dw = 0.8f;
    constexpr float efsolo_dh = 0.98f;

    // Margem estática
    constexpr float ME = 0.08f;

    // Parâmetros de efeito aerodinâmico EH
    constexpr float nh0 = 0.9f;
    constexpr float YH_norm = 0.3f;

    // Parâmetros de fluxo de propulsão
    constexpr float q_dec_est = 0.5f*1.118f*10.0f*10.0f;
    constexpr float propwash_ratio = 0.08f;
    constexpr float prop_pressure_ratio_dec = 3.0f;
    constexpr float prop_pressure_ratio_cruz = 1.5f;
    constexpr float eh_affected_ratio = 0.5f;
    constexpr float ZT = 0.075f;
    constexpr float ip = 2.0f;

    // Fatores de flap simples (profundor)
    constexpr float flap_dcl = efsolo_lift*0.6f*0.85f*0.9f*0.0785f*0.9f;
    constexpr float flap_cd_factor = 0.25f*0.66f*0.3f; // Erik Olson, NASA (cf/c=0.3)

    // Comprimento percentual da longarina
    constexpr float long_perc = 0.8f;

    // Densidade de área
    constexpr float phi_wing = 0.5f;
    constexpr float phi_eh = 0.5f;
    constexpr float phi_ev = 0.5f;

    // Massa específica
    constexpr float rho_tail = 1500.0f;
    constexpr float rho_wlong = 1500.0f;
    constexpr float rho_ehlong = 1500.0f;

    // Constante de tensão
    constexpr float K_sigma_tail = 0.0015;
    constexpr float K_sigma_wing = 0.0004f;
    constexpr float K_sigma_eh = 0.001066f;

    // Constante de área
    constexpr float K_A_tail = 2.0f*PI;
    constexpr float K_A_wlong = 2.0f*PI;
    constexpr float K_A_ehlong = 2.0f*PI;

    // Espessura
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
    constexpr float comp_mass[5] = { 0.467f,  0.150f,  0.050f,  0.450f,  mass_fuse};
    constexpr float comp_xmin[5] = {-0.200f, -0.250f, -0.300f, -0.350f,  xmin_fuse};
    constexpr float comp_xmax[5] = { 0.200f,  0.600f,  0.000f,  0.100f,  xmax_fuse};


    // --- 3D Index Calculation ---
    // Step 1: Calculate the thread's global ID within each dimension (x, y, z).
    // This is the thread's position across the entire grid, not just its local block.
    const int globalThreadIdX = threadIdx.x + blockIdx.x * blockDim.x;
    const int globalThreadIdY = threadIdx.y + blockIdx.y * blockDim.y;
    const int globalThreadIdZ = threadIdx.z + blockIdx.z * blockDim.z;

    // Step 2: Calculate the total width and height of the grid in threads.
    // This is needed to flatten the 3D index into a 1D index.
    const int grid_width_in_threads = gridDim.x * blockDim.x;
    const int grid_height_in_threads = gridDim.y * blockDim.y;

    // Step 3: Flatten the 3D global ID into a single, unique 1D index.
    // This formula maps the (x, y, z) coordinate to a single integer.
    const int idx = globalThreadIdX +
              globalThreadIdY * grid_width_in_threads +
              globalThreadIdZ * grid_width_in_threads * grid_height_in_threads;

    // --- Parameter Generation Logic (from your original code) ---

    // NOTE: Changed `new float[N]` to a stack-allocated array.
    // `new` is not usable inside a __global__ kernel. This works if N is small.
    // For large N, consider using shared memory or global memory.
    float params[N_PARAMS];

    int remaining_idx = idx;

    // Loop backwards from the last parameter (least significant) to the first.
    #pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        // Get the number of variations for the current dimension
        int current_num_vars = num_vars[i];

        // Calculate the individual index for this dimension using modulo
        int individual_index = remaining_idx % current_num_vars;

        // Calculate and store the final scaled value
        params[i] = const_params[i] + individual_index * scale_params[i];

        // Update the remaining index for the next (more significant) dimension
        remaining_idx /= current_num_vars;
    }

    const float AR          = params[0]; 
    const float afil        = params[1];
    const float iw          = params[2];
    const float AR_eh       = params[3];
    const float ih          = params[4];
    const float seh_ratio   = params[5];
    const float XACH_norm   = params[6];
    const float b_wing      = params[7];
    const float k01         = params[8];

    const float a1_w = (float) ((a1+iw) < amax) * a1_w + (float) ((a1+iw) >= amax) * amax;
    const float4 alpha_wing = make_float4(0.0f+iw, a0L, a1_w, amax);

    float4 CLW, CDW, CMW, CLH, CDH, alpha_eh;
    float dCLWda_3d, dCLHda_3d, dCMWda_3d, eh_aoa_rate_ratio;
    { // SCOPE AERODINAMICA 1
      // TODO conferir incidência da asa -> criar alpha_wing com incidencia
      {// SCOPE WING
        const float k_fuse = 1.0f - 2.0f*pow(width_fuse_norm, 2);
        const float fuse_ratio = 1.0f - width_fuse_norm;

        const float f_lambda = 0.0524f * pow(afil, 4) - 0.15f * pow(afil, 3) + 0.1659f * pow(afil, 2) - 0.0706f * afil + 0.0119f;
        const float theo_eff = 1.0f/(1.0f+f_lambda*AR);

        const float alpha_var_ratio = 1.0f / (1.0f + (dCLWda_2d * IDEG2RAD / (PI * theo_eff * AR)));
        const float4 wcl_coeffs_3d = alpha_var_ratio * wcl_coeffs_2d;
        dCLWda_3d = alpha_var_ratio * dCLWda_2d;

        const float CD_0 = 0.0055f * swet_ratio;
        const float4 CDK = wcd_coeffs_2d + make_float4(CD_0,0.0f,0.0f,0.0f);

        const float Q = 1.0f / (theo_eff * k_fuse);
        const float4 KI = 0.38f*CDK + make_float4(Q/(PI*AR),0.0f,0.0f,0.0f);

        dCMWda_3d = dCMWda_2d;

        CLW.x = internal_sum(wcl_coeffs_3d*powers4(alpha_wing.x));
        CDW.x = internal_sum((CDK + efsolo_dw*CLW.x*CLW.x*KI)*powers4(alpha_wing.x));
        CMW.x = internal_sum(wcm_coeffs_2d*powers3(alpha_wing.x));

        CLW.y = internal_sum(wcl_coeffs_3d*powers4(alpha_wing.y));
        CDW.y = internal_sum((CDK + efsolo_dw*CLW.y*CLW.y*KI)*powers4(alpha_wing.y));
        CMW.y = internal_sum(wcm_coeffs_2d*powers3(alpha_wing.y));

        CLW.z = internal_sum(wcl_coeffs_3d*powers4(alpha_wing.z));
        CDW.z = internal_sum((CDK + efsolo_dw*CLW.z*CLW.z*KI)*powers4(alpha_wing.z));
        CMW.z = internal_sum(wcm_coeffs_2d*powers3(alpha_wing.z));

        CLW.w = internal_sum(wcl_coeffs_3d*powers4(alpha_wing.w));
        CDW.w = internal_sum((CDK + efsolo_dw*CLW.w*CLW.w*KI)*powers4(alpha_wing.w));
        CMW.w = internal_sum(wcm_coeffs_2d*powers3(alpha_wing.w));

        CLW = efsolo_lift*fuse_ratio*CLW;
      }

      {// SCOPE Downwash
        const float KA = (1.0f/AR) - (1.0f/(1.0f+pow(AR,1.7f)));
        const float Kl = (10.0f - 3.0f*afil)/7.0f;                                 
        const float KH = (1.0f - (YH_norm/AR))*pow((2.0f*(XACH_norm-XACW_norm)/AR),(-1.0f/3.0f));  

        const float downwash_ratio = pow(4.44f*(KA*Kl*KH),1.19f);
        eh_aoa_rate_ratio = (1.0f - downwash_ratio - propwash_ratio);     

        alpha_eh.x = (alpha_wing.x + ih) - downwash_ratio*(alpha_wing.x - alpha_wing.y) - propwash_ratio*(alpha_wing.x + ip);
        alpha_eh.y = (alpha_wing.y + ih) - downwash_ratio*(alpha_wing.y - alpha_wing.y) - propwash_ratio*(alpha_wing.y + ip);
        alpha_eh.z = (alpha_wing.z + ih) - downwash_ratio*(alpha_wing.z - alpha_wing.y) - propwash_ratio*(alpha_wing.z + ip);
        alpha_eh.w = (alpha_wing.w + ih) - downwash_ratio*(alpha_wing.w - alpha_wing.y) - propwash_ratio*(alpha_wing.w + ip);
      }

      {// SCOPE EH
        const float f_lambda = 0.01f;
        const float theo_eff = 1.0f/(1.0f+f_lambda*AR_eh);

        const float k_a0 = (dCLHdaH_2d * IDEG2RAD / (PI * theo_eff * AR_eh));

        const float alpha_var_eh_ratio = 1.0f / sqrtf(1.0f + k_a0 + k_a0*k_a0);
        const float4 hcl_coeffs_3d = alpha_var_eh_ratio * hcl_coeffs_2d;
        const float dCLHdaH_3d = alpha_var_eh_ratio * dCLHdaH_2d;
        dCLHda_3d = dCLHdaH_3d * eh_aoa_rate_ratio;

        const float4 CDK = hcd_coeffs_2d;

        const float4 KI = 0.38f*CDK + make_float4(1.0f/(theo_eff*PI*AR_eh),0.0f,0.0f,0.0f);

        CLH.x = internal_sum(hcl_coeffs_3d*powers4(alpha_eh.x));
        CDH.x = internal_sum((CDK + efsolo_dh*CLH.x*CLH.x*KI)*powers4(alpha_eh.x));

        CLH.y = internal_sum(hcl_coeffs_3d*powers4(alpha_eh.y));
        CDH.y = internal_sum((CDK + efsolo_dh*CLH.y*CLH.y*KI)*powers4(alpha_eh.y));

        CLH.z = internal_sum(hcl_coeffs_3d*powers4(alpha_eh.z));
        CDH.z = internal_sum((CDK + efsolo_dh*CLH.z*CLH.z*KI)*powers4(alpha_eh.z));

        CLH.w = internal_sum(hcl_coeffs_3d*powers4(alpha_eh.w));
        CDH.w = internal_sum((CDK + efsolo_dh*CLH.w*CLH.w*KI)*powers4(alpha_eh.w));

        CLH = efsolo_lift*CLH;
      }
    }

    float XCG_norm;
    { // SCOPE ESTABILIDADE 1 
    // TODO conferir, apenas aqui não tem efeito solo                     
      const float eh_dpressure_ratio = nh0*(1.0f + eh_affected_ratio*prop_pressure_ratio_cruz);

      const float dCLHda_eff = eh_dpressure_ratio * seh_ratio * dCLHda_3d;                                    

      const float XNP_norm = (dCMWda_3d + dCLWda_3d*XACW_norm + dCLHda_eff*XACH_norm)/(dCLWda_3d + dCLHda_eff);

      XCG_norm = XNP_norm - ME;
    }

    const float MAC = b_wing/AR;
    const float S_wing = MAC*b_wing;

    const float S_eh = S_wing*seh_ratio;
    const float b_eh = pow(S_eh*AR_eh, 0.5f);

    const float S_ev = S_wing*sev_ratio;

    bool prof_def_allowed;
    { // SCOPE ESTABILIDADE 2 

      const float eh_dpressure_ratio = nh0*(1.0f + eh_affected_ratio*prop_pressure_ratio_dec);
      const float eh_eff_arm = eh_dpressure_ratio * seh_ratio *(XCG_norm - XACH_norm);

      const float CMT = -(T0/q_dec_est - (2.0f*a)/rho)*(ZT/MAC);

      { // SCOPE a1
        const float CMLW_a1 = (XCG_norm - XACW_norm)*(CLW.z);
        const float CM_a1 = CMW.z + CMLW_a1 + CMT;
        const float CMH_a1 = eh_eff_arm*CLH.z;

        CLH.z = -CM_a1/eh_eff_arm;

        const float CMt_a1 = CM_a1 + CMH_a1;
        const float req_dclh_a1 = CMt_a1/(eh_eff_arm);

        const float prof_def_a1 = flap_dcl*((float) (req_dclh_a1 <= 0.48f) * (1030.0f*req_dclh_a1) + (float) (req_dclh_a1 > 0.48f) * (6250.0f*(req_dclh_a1 - 0.48f) + 494.4f));
        const float prof_drag_increase_a1 = flap_cd_factor*dCLHdaH_2d*prof_def_a1*prof_def_a1*DEG2RAD;

        CDH.z += prof_drag_increase_a1;
      }

      { // SCOPE a2
        const float CMLW_a2 = (XCG_norm - XACW_norm)*(CLW.w);
        const float CM_a2 = CMW.w + CMLW_a2 + CMT;
        const float CMH_a2 = eh_eff_arm*CLH.w;

        CLH.w = -CM_a2/eh_eff_arm;

        const float CMt_a2 = CM_a2 + CMH_a2;
        const float req_dclh_a2 = CMt_a2/(eh_eff_arm);

        const float prof_def_a2 = flap_dcl*((float) (req_dclh_a2 <= 0.48f) * (1030.0f*req_dclh_a2) + (float) (req_dclh_a2 > 0.48f) * (6250.0f*(req_dclh_a2 - 0.48f) + 494.4f));
        const float prof_drag_increase_a2 = flap_cd_factor*dCLHdaH_2d*prof_def_a2*prof_def_a2*DEG2RAD;

        CDH.w += prof_drag_increase_a2;

        prof_def_allowed = prof_def_a2 < 35.0f;
      }

    }


    bool XCG_attainable;
    float PV;
    { // SCOPE ESTRUTURAS 
      float comp_xmid[5];
      #pragma unroll
      for (int comp = 0; comp < n_comps; ++comp) {
        comp_xmid[comp] = 0.5f*(comp_xmin[comp] + comp_xmax[comp]);
      }

      const float XCG_alvo = XCG_norm*MAC;

      const float w_corr_factor = K_sigma_wing*pow(b_wing*S_wing*(0.1f+CLW.w), 0.5f);
      const float mass_wlong = long_perc*b_wing*rho_wlong*K_A_wlong*pow(t_wlong, 0.5f)*w_corr_factor;
      const float mass_wing = phi_wing*S_wing + mass_wlong + 2*mass_wservo;
      const float x_wing = XACW_norm*MAC;

      const float eh_corr_factor = K_sigma_eh*pow(b_eh*S_eh*(0.1f+abs(CLH.w)), 0.5f);
      const float mass_ehlong = b_eh*rho_ehlong*K_A_ehlong*pow(t_ehlong, 0.5f)*eh_corr_factor;
      const float mass_eh = phi_eh*S_eh + mass_ehlong + mass_ehservo;
      const float x_eh = XACH_norm*MAC;

      const float mass_ev = phi_ev*S_ev + 2.0f*mass_evservo;
      const float x_ev = x_eh;

      const float tail_corr_factor = K_sigma_tail*pow(XACH_norm*MAC*S_eh*(0.1f+abs(CLH.w)), 0.5f);
      const float mass_tail = (XACH_norm-0.8f)*MAC*rho_tail*K_A_tail*pow(t_tail, 0.5f)*tail_corr_factor;
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

      XCG_attainable = (abs(XCG_atual - XCG_alvo) < 0.01f);

      XCG_norm = XCG_atual/MAC;
      PV = m_total;
    }


    const float4 CLG = CLW + seh_ratio*CLH;
    const float4 CDG = CDW + seh_ratio*CDH;

    float result_PF;
    bool minimum_CP;
    { // SCOPE DESEMPENHO

      // Define function f
      auto f = [=] (float P) -> float {
          // Parâmetros de aceleração no estágio 1
          float A0 = (g_acc/P)*(T0 - P*mu);
          float B0 = (g_acc/P)*(0.5f*rho*S_wing*(CDG.x  - mu*CLG.x ) + a);

          float Vst = sqrtf((2.0f*P)/(rho*S_wing*CLG.z));         // Vel. de estol em alpha_1
          float V01 = Vst*k01;                                    // Vel. do estágio 1

          // Deslocamento no estágio 1
          float disp_0 = (0.5f/B0)*logf((A0)/(A0-B0*V01*V01));

          // Deslocamento restante para o estágio 2 (max. 55m)
          float disp_2 = fmaxf(55.0f - disp_0, 0.0f);

          // Função de entrada de RK4, cálculo das derivadas de movimento
          auto dfunc = [=] (float3 y) -> float3 {

            float CLG_dec = (float) ((y.z/MAC) <= Y_a2_norm) * CLG.z + (float) ((y.z/MAC) > Y_a2_norm) * CLG.w;
            float CDG_dec = (float) ((y.z/MAC) <= Y_a2_norm) * CDG.z + (float) ((y.z/MAC) > Y_a2_norm) * CDG.w;

            float accyp = (g_acc/P)*(0.5f*rho*S_wing*CLG_dec*y.x*y.x);           // Acc. em y (coord. avião)

            float acc_drag = -(g_acc/P)*(0.5f*rho*S_wing*CDG_dec*y.x*y.x);       // Acc. da sust. (coord. avião)
            float acc_prop = (g_acc/P)*(T0 - a*(y.x*y.x+y.y*y.y));        // Acc. da propulsão (coord. avião)
            float accxp = acc_drag + acc_prop;                        // Acc. em x (coord. avião)

            float theta = atanf(y.y/y.x);                             // Angulo de voo (trajetória)

            // Acc. corrigidas para sistema de coordenadas padrão
            float accx = accxp*cosf(theta) - accyp*sinf(theta);
            float accy = accxp*sinf(theta) + accyp*cosf(theta) - g_acc;

            // Derivadas de movimento (dvx/dx, dvy/dx, dy/dx)
            return (float)(y.x > 0.0f) * (make_float3(accx, accy, y.y) / y.x);

          };

          // Vetor inicial (vx0, vy0, y0)
          float3 y0 = make_float3(V01, 0.0f, 0.0f);

          // Aplicação de RK4
          RK4(dfunc, y0, disp_2);

          // Deslocamento em y final
          return fmaxf(0.0f, y0.z) - 0.9f;
      };

      float p1 = (5.0f+PV)*g_acc;
      const float p1_o = p1;
      float p2 = 25.0f*g_acc;
      
      float f_p1 = f(p1);

      #pragma unroll
      for (int i = 0; i < 30; ++i) {
          float p3 = p1 + (p2 - p1) * 0.5f; // More stable than (a+b)/2
          float f_p3 = f(p3);

          // --- Branchless Interval Update ---
          // Condition: Does the root lie in the left half [p1, p3]?
          // This is true if f(p1) and f(p3) have opposite signs.
          bool is_in_left_half = (f_p1 * f_p3 < 0.0f);

          // If 'is_in_left_half' is true (1), p2 becomes p3. p1 remains p1.
          // If 'is_in_left_half' is false (0), p1 becomes p3. p2 remains p2.
          p1 = (float)(!is_in_left_half) * p3 + (float)(is_in_left_half) * p1;
          p2 = (float)(is_in_left_half) * p3 + (float)(!is_in_left_half) * p2;
          
          // Update f_p1 only if 'p1' was updated.
          f_p1 = (float)(!is_in_left_half) * f_p3 + (float)(is_in_left_half) * f_p1;
      }

      // The root is the midpoint of the final, narrow interval.
      const float TOW = (float)(f_p1 > 0.0f) * (p1 + (p2 - p1) * 0.5f) + (float)(f_p1 == 0.0f) * (p1_o);
      minimum_CP = (f_p1 > 0.0f);
      
      const float TOM = TOW/g_acc;

      const float delta_b = b_wing - 4.0f;
      const float delta_W = dW_max - 600.0f;

      const float PEE = 25.0f*((TOM/PV) - 1.0f);

      // Example scalar function G(F, x, w)
      result_PF = max(0.0f, PEE - max(0.0, 100.0f*delta_b + ((float) (delta_b > 0.05f) * 20.0f)) - max(0.0f, 0.5f*delta_W));
    } // END SCOPE DESEMPENHO

    criteria_1[idx] = XCG_attainable;
    criteria_2[idx] = prof_def_allowed;
    criteria_3[idx] = minimum_CP;
    output[idx] = (float) XCG_attainable * (float) prof_def_allowed * result_PF;
}
}
