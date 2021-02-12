functions {
  vector strat_seir(real t, vector V, real[] p, int n_groups, matrix trans_matrix){

    vector[size(V)] dV;
    real N = sum(V);
    int s_end;
    int e_end;
    int i_end;
    int r_end;

    s_end = n_groups;
    e_end = 2 * s_end;
    i_end = s_end+e_end; // 3*n_groups
    r_end = 4 * n_groups;

    vector[n_groups] exposed;
    vector[n_groups] infectious;
    vector[n_groups] recovered;

    exposed = p[1] * (V[1:s_end] .* (trans_matrix * (p[1] * V[s_end:e_end] + V[e_end:i_end]))/N);
    infectious = p[3] * V[s_end:e_end];
    recovered = p[2] * V[e_end:i_end];

    dV[0:s_end] = -1 * exposed;
    dV[s_end:e_end] = exposed - infectious;
    dV[e_end:i_end] = infectious - recovered;
    dV[i_end:r_end] = recovered;

    return dV;
  }
}

data {
  real<lower=0> init_compartments[4]; 
  int<lower=1> n_groups; 
  real<lower=0> group_ratios[n_groups]; // percentage of population for each group
  int<lower=1> number_time_points;
  int<lower=1> number_parameters;
  int<lower=1> number_variables;
  real<lower=0> t_start;
  real<lower=0> time_series[number_time_points];
  int<lower=0> incidence_data[number_time_points-1];

  matrix[n_groups,n_groups] trans_matrix;
}

transformed data{
  vector[4*n_groups] init;
  for (i in 1:4){
    for (j in 1:n_groups){
      init[(n_groups*(i-1))+j] = init_compartments[i]*group_ratios[j];
    }
  }
}

parameters{
  //real<lower=0.00> params[number_parameters];
  /*
   * 1. alpha
   * 2. beta
   * 3. gamma
   * 4. eta
  */
  real<lower=0.0> a; // alpha
  real<lower=0.0> b; // beta
}

transformed parameters{
  vector[number_variables] x_hat[number_time_points]; // output from the ODE solver
  real<lower=0> model_incidence[number_time_points-1]; // diff of output from ODE solver to calculate incidence
  real params[number_parameters];
  params[1] = a;
  params[2] = b;
  params[3] = 1/5;
  params[4] = 1/3;


  x_hat = ode_rk45(strat_seir, init, t_start, time_series, params, n_groups, trans_matrix);

  for (i in 1:number_time_points-1){
    if(x_hat[i+1, 4] - x_hat[i, 4]<1e-5){
      model_incidence[i] = 0.0;
    }
    else{
      model_incidence[i] = x_hat[i+1, 4] - x_hat[i, 4];
    }
  }
}

model {
  /*
  params[1] ~ uniform(0,1); // alpha
  params[2] ~ gamma(1.1,2.5); // beta
  params[3] = 1/5; // gamma
  params[4] = 1/3; // eta
  */
  a ~ uniform(0,1); // alpha
  b ~ gamma(1.1,2.5); // beta
  

  for (t in 1:number_time_points-1) {
    incidence_data[t] ~ poisson(model_incidence[t]);
  }
}

generated quantities{
  vector[number_variables] sim_x_hat[number_time_points];
  real sim_incidence[number_time_points-1];

  sim_x_hat = ode_rk45(strat_seir, init, t_start, time_series, params, n_groups, trans_matrix);

  for (t in 1:number_time_points-1){
    if(sim_x_hat[t+1, 4] - sim_x_hat[t, 4]<1e-5){
      sim_incidence[t] = 0.0;
    }
    else{
      sim_incidence[t] =  sim_x_hat[t+1, 4] - sim_x_hat[t, 4];
    }
  }
}
