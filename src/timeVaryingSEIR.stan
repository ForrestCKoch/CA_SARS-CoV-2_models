functions {
  real get_infected(real t, real beta_start, real beta_end, real k,real m, vector V){
    real N = sum(V);
    real beta_eff;
    real infected;
    beta_eff = beta_end + (beta_start - beta_end)/(1+exp(-k*(m-t)));
    infected = beta_eff * V[1] * (0.2*sum(V[2:17])+1.2*sum(V[18:19])+sum(V[20:29])) / N;
    return infected;
  }

  vector strat_seir(real t, vector V, real beta_start, real beta_end, real m, real k){

    vector[num_elements(V)] dV;
    real N = sum(V);
    real beta_eff;
    real infected;

    infected = get_infected(t, beta_start, beta_end, k, m, V);

    dV[1] = -infected; // S
    dV[2] = infected - V[2]; // L1
    dV[3:5] = V[2:4] - V[3:5]; //L2-4
    dV[6] = 0.2 * V[5] - V[6]; // A1
    dV[7:17] = V[6:16] - V[7:17]; //A2-12
    dV[18] = 0.8 * V[5] - V[18]; //P1
    dV[19:29] = V[18:28] - V[19:29]; //P2,I1-10
    dV[30] = V[17] + V[29]; // R

    return dV;
  }
}

data {
  vector<lower=0>[4] init_props; 
  int<lower=1> number_time_points;
  int<lower=1> number_parameters;
  int<lower=1> number_variables;
  real<lower=0> t_start;
  real<lower=0> time_series[number_time_points];
  int<lower=0> incidence_data[number_time_points-1];
}

transformed data{
  vector<lower=0>[30] init;
  init[1] = init_props[1]; // S
  init[2:5] = to_vector(rep_array(init_props[2]/4,4)); // L1-4
  init[6:17] = to_vector(rep_array(0.2*init_props[3]/12,12)); //A1-12
  init[18:29] = to_vector(rep_array(0.8*init_props[3]/12,12)); //P1-I12
  init[30] = init_props[4];
}

parameters{
  real<lower=0> beta_start; // initial beta
  real<lower=0,upper=1> beta_fract; // proportion reduction in beta
  real<lower=0,upper=1> k;
}

transformed parameters{
}

model {
  vector[number_variables] x_hat[number_time_points]; // output from the ODE solver

  beta_start ~ cauchy(0,.25);
  beta_fract ~ beta(2,2);
  k ~ beta(2,8);

  x_hat = ode_bdf(strat_seir, init, t_start, time_series, beta_start, beta_start*beta_fract, number_time_points/2.0, k);

  for (t in 1:number_time_points-1) {
    incidence_data[t] ~ poisson(max([x_hat[t,19],0]));
  }
}

generated quantities{
  real beta_end = beta_start*beta_fract;
}
