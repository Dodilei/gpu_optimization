// OBS coeffs should be in INCREASING power order
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

template <typename LambdaType>
__forceinline__ __device__ void RK4(LambdaType dfunc, float3& y0, const float& xf)
{
    float dx = xf/100.0f;

    for (int i = 0; i < 100; i++) {
        float3 y_prev = y0;
        float3 k1 = dfunc(y_prev               );
        float3 k2 = dfunc(y_prev + (dx/2.0f)*k1);
        float3 k3 = dfunc(y_prev + (dx/2.0f)*k2);
        float3 k4 = dfunc(y_prev +  dx      *k3);

        y0 = y_prev + (dx/6.0f) * (k1 + 2*k2 + 2*k3 + k4);
    }
}

extern "C" {
__global__ void stage_desempenho(
    float* results,          
    const float g_acc,
    const float rho,
    const float T0,
    const float a,
    const float mu,
    const float b_wing,
    const float dW_max,
    const float PV,
    const float4 wcl_coeffs,
    const float4 wcd_coeffs,
    const float* scale_params,
    const float* const_params,
    const int* num_vars,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int remaining_idx = idx;

    float* params = new float[N];
    // Loop backwards from the last parameter (least significant) to the first.
    for (int i = N - 1; i >= 0; --i) {
        // Get the number of variations for the current dimension
        int current_num_vars = num_vars[i];

        // Calculate the individual index for this dimension using modulo
        int individual_index = remaining_idx % current_num_vars;

        // Calculate and store the final scaled value
        // We write all N parameters for a given idx contiguously
        params[i] = const_params[i] + scale_params[i] * individual_index;

        // Update the remaining index for the next (more significant) dimension
        remaining_idx /= current_num_vars;
    }

    float a1 = params[0];
    float k01 = params[1];
    float a2 = params[2];
    float S = params[3];

    float CD_0 = wcd_coeffs.x;
    float CD_a1 = internal_sum(wcd_coeffs*powers4(a1));
    float CD_a2 = internal_sum(wcd_coeffs*powers4(a2));

    float CL_0 = wcl_coeffs.x;
    float CL_a1 = internal_sum(wcl_coeffs*powers4(a1));
    float CL_a2 = internal_sum(wcl_coeffs*powers4(a2));

    // Define function f
    auto f = [=] (float P) -> float {

        float V12 = sqrtf((2.0f*P)/(rho*S*CL_a1));         // Vel. de estol em alpha_1
        float V01 = V12*k01;                               // Vel. do estágio 1

        // Parâmetros de aceleração nos estágios 0 e 1
        float A0 = (g_acc/P)*(T0 - P*mu);
        float A1 = A0;
        float B0 = (g_acc/P)*(0.5f*rho*S*(CD_0  - mu*CL_0 ) + a);
        float B1 = (g_acc/P)*(0.5f*rho*S*(CD_a1 - mu*CL_a1) + a);

        // Deslocamento nos estágios 0 e 1
        float disp_0 = (0.5f/B0)*logf((A0)           /(A0-B0*V01*V01));
        float disp_1 = (0.5f/B1)*logf((A1-B1*V01*V01)/(A1-B1*V12*V12));

        // Deslocamento restante para o estágio 2 (max. 55m)
        float disp_2 = fmaxf(55.0f - disp_0 - disp_1, 0.0f);

        // Função de entrada de RK4, cálculo das derivadas de movimento
        auto dfunc = [=] (float3 y) -> float3 {

          float accyp = (g_acc/P)*(0.5f*rho*S*CL_a2*y.x*y.x);           // Acc. em y (coord. avião)

          float acc_lift = -(g_acc/P)*(0.5f*rho*S*CD_a2*y.x*y.x);       // Acc. da sust. (coord. avião)
          float acc_prop = (g_acc/P)*(T0 - a*(y.x*y.x+y.y*y.y));        // Acc. da propulsão (coord. avião)
          float accxp = acc_lift + acc_prop;                        // Acc. em x (coord. avião)

          float theta = atanf(y.y/y.x);                             // Angulo de voo (trajetória)

          // Acc. corrigidas para sistema de coordenadas padrão
          float accx = accxp*cosf(theta) - accyp*sinf(theta);
          float accy = accxp*sinf(theta) + accyp*cosf(theta) - g_acc;

          // Derivadas de movimento (dvx/dx, dvy/dx, dy/dx)
          return (float)(y.x > 0.0f) * (make_float3(accx, accy, y.y) / y.x);

        };

        // Vetor inicial (vx0, vy0, y0)
        float3 y0 = make_float3(V12, 0.0f, 0.0f);

        // Aplicação de RK4
        RK4(dfunc, y0, disp_2);

        // Deslocamento em y final
        return fmaxf(0.0f, y0.z) - 0.9f;
    };

    float p1 = 1*g_acc;
    float p1_o = p1;
    float p2 = 20*g_acc;
    
    float f_p1 = f(p1);

    for (int i = 0; i < 8; ++i) {
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
    float TOW = (float)(f_p1 > 0.0f) * (p1 + (p2 - p1) * 0.5f) + (float)(f_p1 <= 0.0f) * (p1_o);
    
    float TOM = TOW/g_acc;

    float delta_b = b_wing - 4;
    float delta_W = dW_max - 600;

    float PEE = 25*((TOM/PV) - 1);

    // Example scalar function G(F, x, w)
    float result_PF = max(0.0f, PEE - max(0.0, 100.0f*delta_b + ((float) (delta_b > 0.05f) * 20.0f)) - max(0.0f, 0.5f*delta_W));

    results[idx] = result_PF;
};
}
