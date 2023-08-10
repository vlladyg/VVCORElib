#include <math.h>

#ifndef RHOPREC
#warning "Defaulting to double precision"
#define RHOPREC double
#endif

void rho_k(const RHOPREC x_vec[][3], int N_x,
           const RHOPREC k_vec[][3], int N_k,
           RHOPREC (* restrict rho_k)[2]){

  int x_i, k_i;
  RHOPREC rho_ki_0, rho_ki_1;
  register RHOPREC alpha;

  {
  for(k_i=0; k_i<N_k; k_i++){

    rho_ki_0 = 0.0;
    rho_ki_1 = 0.0;

    for(x_i=0; x_i<N_x; x_i++){
      alpha = \
        x_vec[x_i][0] * k_vec[k_i][0] +
        x_vec[x_i][1] * k_vec[k_i][1] +
        x_vec[x_i][2] * k_vec[k_i][2];
      rho_ki_0 += cos(alpha);
      rho_ki_1 += sin(alpha);
    }
    rho_k[k_i][0] = rho_ki_0;
    rho_k[k_i][1] = rho_ki_1;
  }

  }
}


void rho_j_k(const RHOPREC x_vec[][3], const RHOPREC v_vec[][3], int N_x,
             const RHOPREC k_vec[][3], int N_k,
             RHOPREC (* restrict rho_k)[2],
             RHOPREC (* restrict j_k)[6]){

  int x_i, k_i;
  RHOPREC rho_ki_0, rho_ki_1;
  RHOPREC j_ki_0, j_ki_1, j_ki_2, j_ki_3, j_ki_4, j_ki_5;
  register RHOPREC alpha, ca, sa;

  {

    /* Both rho_k and j_k */
  for(k_i=0; k_i<N_k; k_i++){

    rho_ki_0 = 0.0;
    rho_ki_1 = 0.0;
    j_ki_0 = 0.0;
    j_ki_1 = 0.0;
    j_ki_2 = 0.0;
    j_ki_3 = 0.0;
    j_ki_4 = 0.0;
    j_ki_5 = 0.0;

    for(x_i=0; x_i<N_x; x_i++){
      alpha = \
        x_vec[x_i][0] * k_vec[k_i][0] +
        x_vec[x_i][1] * k_vec[k_i][1] +
        x_vec[x_i][2] * k_vec[k_i][2];
      ca = cos(alpha);
      sa = sin(alpha);
      rho_ki_0 += ca;
      rho_ki_1 += sa;
      j_ki_0 += ca * v_vec[x_i][0];
      j_ki_1 += sa * v_vec[x_i][0];
      j_ki_2 += ca * v_vec[x_i][1];
      j_ki_3 += sa * v_vec[x_i][1];
      j_ki_4 += ca * v_vec[x_i][2];
      j_ki_5 += sa * v_vec[x_i][2];
    }
    rho_k[k_i][0] = rho_ki_0;
    rho_k[k_i][1] = rho_ki_1;
    j_k[k_i][0] = j_ki_0;
    j_k[k_i][1] = j_ki_1;
    j_k[k_i][2] = j_ki_2;
    j_k[k_i][3] = j_ki_3;
    j_k[k_i][4] = j_ki_4;
    j_k[k_i][5] = j_ki_5;
  }

  }
}

void v_k(const RHOPREC v_vec[][3], int N_x,
             const RHOPREC (* restrict egv)[6], int N_k,
             RHOPREC (* restrict v_k)[6]){

  int x_i, k_i;
  RHOPREC v_ki_0, v_ki_1, v_ki_2, v_ki_3, v_ki_4, v_ki_5;

  {

    /* Iter k vector x m modes */
  for(k_i=0; k_i<N_k; k_i++){
    v_ki_0 = 0.0;
    v_ki_1 = 0.0;
    v_ki_2 = 0.0;
    v_ki_3 = 0.0;
    v_ki_4 = 0.0;
    v_ki_5 = 0.0;
    for(x_i=0; x_i<N_x; x_i++){
      v_ki_0 += egv[k_i][0] * v_vec[x_i][0];
      v_ki_1 += egv[k_i][1] * v_vec[x_i][0];
      v_ki_2 += egv[k_i][2] * v_vec[x_i][1];
      v_ki_3 += egv[k_i][3] * v_vec[x_i][1];
      v_ki_4 += egv[k_i][4] * v_vec[x_i][2];
      v_ki_5 += egv[k_i][5] * v_vec[x_i][2];
    }
    v_k[k_i][0] = v_ki_0;
    v_k[k_i][1] = v_ki_1;
    v_k[k_i][2] = v_ki_2;
    v_k[k_i][3] = v_ki_3;
    v_k[k_i][4] = v_ki_4;
    v_k[k_i][5] = v_ki_5;
  }

  }
}
