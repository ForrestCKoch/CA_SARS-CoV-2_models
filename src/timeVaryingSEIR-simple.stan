functions {
  vector strat_seir(real t, vector V, real beta_start, real beta_end, real m, real k){

    //vector[size(V)] dV;
    vector[num_elements(V)] dV;
    real N = sum(V);
    real beta_eff;
    real infected;
    real recovered;

    beta_eff = beta_end + (beta_start - beta_end)/(1+exp(-k*(m-t)));

    infected = beta_eff * V[1] * V[2] / N;
    recovered = (1.0/7.0)*V[2];

    dV[1] = - infected;
    dV[2] = infected-recovered;
    dV[3] = recovered;

    /*
    int s_end;
    int e_end;
    int i_end;
    int r_end;
    vector[n_groups] exposed;
    vector[n_groups] infectious;
    vector[n_groups] recovered;
    
    s_end = n_groups;
    e_end = 2 * s_end;
    i_end = s_end+e_end; // 3*n_groups
    r_end = 4 * n_groups;

    //beta_eff = ;

    exposed = p[2] * (V[1:s_end] .* (trans_matrix * (p[1] * V[(s_end+1):e_end] + V[(e_end+1):i_end]))/N);
    infectious = p[3] * V[(s_end+1):e_end];
    recovered = p[4] * V[(e_end+1):i_end];

    dV[1:s_end] = -1 * exposed;
    dV[(s_end+1):e_end] = exposed - infectious;
    dV[(e_end+1):i_end] = infectious - recovered;
    dV[(i_end+1):r_end] = recovered;
    */

    return dV;
  }
}

data {
  vector<lower=0>[3] init; 
  int<lower=1> number_time_points;
  int<lower=1> number_parameters;
  int<lower=1> number_variables;
  real<lower=0> t_start;
  real<lower=0> time_series[number_time_points];
  int<lower=0> incidence_data[number_time_points-1];
}

transformed data{
}

parameters{
  //real<lower=0.00> params[number_parameters];
  /*
   * 1. alpha
   * 2. beta
   * 3. gamma
   * 4. eta
  */
  //real a; // alpha
  real<lower=0> beta_start; // initial beta
  real<lower=0> beta_end; // final beta
  real<lower=0,upper=1> m; // midpoint of the logistic function
  real<lower=0,upper=1> k;
  //real<lower=1> p1;
  //real<lower=1> p2;
}

transformed parameters{
  vector[number_variables] x_hat[number_time_points]; // output from the ODE solver
  //real<lower=0> model_incidence[number_time_points-1]; // diff of output from ODE solver to calculate incidence
  vector<lower=0>[number_time_points-1] model_incidence; // diff of output from ODE solver to calculate incidence

  x_hat = ode_bdf(strat_seir, init, t_start, time_series, beta_start, beta_end, m*number_time_points, k);
  //print(x_hat);

  for (i in 1:number_time_points-1){
    if(x_hat[i,1]-x_hat[i+1,1]<1e-5){
      model_incidence[i] = 0.0;
    }
    else{
      model_incidence[i] = x_hat[i,1]-x_hat[i+1,1];
    }
  }
}

model {
  beta_start ~ cauchy(0,.1);
  beta_end ~ cauchy(0,.1);
  m ~ beta(2,2);
  k ~ beta(2,2);
  
  for (t in 1:number_time_points-1) {
    incidence_data[t] ~ poisson(model_incidence[t]);
  }
  //print(incidence_data);
  //print(model_incidence);
}

generated quantities{
  /*
  vector[number_variables] sim_x_hat[number_time_points];
  real sim_incidence[number_time_points-1];

  sim_x_hat = ode_bdf(strat_seir, init, t_start, time_series, params, n_groups, trans_matrix);

  for (t in 1:number_time_points-1){
    if(sim_x_hat[t+1, 4] - sim_x_hat[t, 4]<1e-5){
      sim_incidence[t] = 0.0;
    }
    else{
      sim_incidence[t] =  sim_x_hat[t+1, 4] - sim_x_hat[t, 4];
    }
  }
  */
}
