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

    //vector[size(V)] dV;
    vector[num_elements(V)] dV;
    real N = sum(V);
    real beta_eff;
    real infected;

    infected = get_infected(t, beta_start, beta_end, k, m, V);

    dV[1] = -infected; // S
    dV[2] = infected - V[2]; // L1
    dV[3:5] = V[2:4] - V[3:5]; //L2-4
    dV[6] = 0.2 * V[5]; // A1
    dV[7:17] = V[6:16] - V[7:17]; //A2-12
    dV[18] = 0.8 * V[5]; //P1
    dV[19:29] = V[18:28] - V[19:29]; //P2,I1-10
    dV[30] = V[17] + V[29]; // R

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

  /*
  for (i in 1:number_time_points-1){
    if(x_hat[i,1]-x_hat[i+1,1]<1e-5){
      model_incidence[i] = 0.0;
    }
    else{
      model_incidence[i] = x_hat[i,1]-x_hat[i+1,1];
    }
  }
  */
  for (i in 1:number_time_points-1){
    model_incidence[i] = get_infected(i, beta_start, beta_end, k, m*number_time_points, x_hat[(i+1),:]);
  }
}

model {
  beta_start ~ normal(.45,1);
  beta_end ~ normal(.11,1);
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
