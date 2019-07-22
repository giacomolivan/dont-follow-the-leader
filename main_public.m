%%% This code simulates the model introduced in the paper "Don't follow the
%%% leader: How ranking performance reduces meritocracy" by Giacomo Livan
%%% (2019)

clear all
close all

N = 200; % Number of agents
M = 1000; % Number of available actions
T = 500; % Time steps
q_vec = linspace(0,1,10); % Values of the parameter q of the model

Niter = 100; % Number of independent simulations to be run for each value of q

%%% Vectors to store the values of the quantities dicussed in the paper at
%%% the end of simulations
total_wealth = zeros(Niter,length(q_vec));
gini = zeros(Niter,length(q_vec));
corr_maxfit_wealth = zeros(Niter,length(q_vec));
corr_avgfit_wealth = zeros(Niter,length(q_vec));
diversity = zeros(Niter,length(q_vec));
min_wealth = zeros(Niter,length(q_vec));
max_wealth = zeros(Niter,length(q_vec));

for qq = 1:length(q_vec)
    
    q = q_vec(qq);
    
    for ni = 1:Niter

        p = rand(M,1); % Societal payoffs of actions
        alpha = rand(N,M); % Relative individual payoffs of actions

        actions = randi(M,N,1); % Vector of strategies played at time zero

        wealth = []; % Initial wealth

        for i = 1:N
            wealth = [wealth; alpha(i,actions(i,end))*p(actions(i,end))];
        end

        for t = 1:T-1

            actions_new = zeros(N,1);
            wealth_new = wealth(:,end);
            tmp = rand(N,1);

            for i = 1:N

                % If random number is lower than individual payoff keep same action
                if tmp(i) < alpha(i,actions(i,end))*p(actions(i,end))

                    actions_new(i) = actions(i,end);
                    wealth_new(i) = wealth_new(i) + alpha(i,actions_new(i))*p(actions_new(i));

                else % If random number is higher than individual payoff change action

                    if rand < q % Find agents higher in ranking and copy their current strategy

                        f = find(wealth(:,end) > wealth(i,end)); 

                        if isempty(f) == 0
                           f = f(randperm(length(f)));
                           f = f(1);
                           actions_new(i) = actions(f,end);
                           wealth_new(i) = wealth_new(i) + alpha(i,actions_new(i))*p(actions_new(i));
                        else
                           actions_new(i) = actions(i,end);
                           wealth_new(i) = wealth_new(i) + alpha(i,actions_new(i))*p(actions_new(i));                    
                        end

                    else % Choose new random action

                        actions_new(i) = randi(M); 
                        wealth_new(i) = wealth_new(i) + alpha(i,actions_new(i))*p(actions_new(i));

                    end

                end

            end

            actions = [actions actions_new];
            wealth = [wealth wealth_new];

        end
                
        total_wealth(ni,qq) = sum(wealth(:,end));
        
        aux = sort(wealth(:,end));
       
        min_wealth(ni,qq) = sum(aux(1:20));
        max_wealth(ni,qq) = sum(aux(end-20:end));
        
        diversity(ni,qq) = length(unique(actions(:,end)))/M;

        %%% Gini coefficient

        G = 0;

        for i = 1:N
            for j = i+1:N

                G = G + abs(wealth(i,end) - wealth(j,end));

            end
        end

        gini(ni,qq) = G/(N*sum(wealth(:,end)));

        %%% Optimal actions

        opt_payoff = [];
        avg_fitness = [];
        max_fitness = [];

        for i = 1:N

            tmp = alpha(i,:)'.*p;
            avg_fitness = [avg_fitness; sum(tmp)/M];
            max_fitness = [max_fitness; max(tmp)];

            f = find(tmp == max(tmp));

            opt_payoff = [opt_payoff; alpha(i,f)*p(f)];

        end
                
        corr_maxfit_wealth(ni,qq) = corr(max_fitness,wealth(:,end),'type','Kendall');
        corr_avgfit_wealth(ni,qq) = corr(avg_fitness,wealth(:,end),'type','Kendall');
        
        fprintf('q = %3.2f, ni = %d\n',q,ni)
    
    end

end
