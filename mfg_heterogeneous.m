% =========================================================================
% MFG ETENDU - Modele heterogene Rdj = lambda_j * Rd
% D'apres Toumi, Malhame & Le Ny, Automatica 160 (2024) + extension
% Section 3.1 du rapport de stage 
%
% Modifications vs modele original :
%   - Rd remplace par Rdj = lambda_j * Rd, j = 1,...,D
%   - Riccati phi_j par cible (eq. 5j)
%   - Bassins quadratiques (Theorem 1*)
%   - phi_bar_het = blkdiag(phi_{j(1)}, ...) au lieu de Ih x phi
%   - Algo de point fixe : Anderson mixing (m=5) au lieu de Picard
%
% UTILISATION : modifier PARAMETRES puis Run.
% =========================================================================
clear; close all; clc;

% =========================================================================
%  PARAMETRES
% =========================================================================
A  = zeros(2);
B  = eye(2);

Rx = 1.0 * eye(2);
Rd = eye(2);
Ru = 5.0 * eye(2);
M  = 100.0 * eye(2);
lambda = struct('d1', eye(2), 'd2', diag([6 0.3]), ...
                'd3', eye(2), 'd4', diag([0.3 6]));

T        = 4.0;
N_agents = 1000;
N_MC     = 1500;
tol_P    = 0.02;
tol_X    = 0.5;
max_iter = 80;
N_time   = 200;

% Parametres Anderson mixing
m_anderson = 5;   % fenetre memoire

% =========================================================================
%  CONFIGURATION (identique a Toumi)
% =========================================================================
targets      = struct('d1',[-10;-10],'d2',[-10;10],'d3',[10;-10],'d4',[10;10]);
target_names = {'d1','d2','d3','d4'};
region_targets = {{'d1','d2'},{'d1','d3'},{'d2','d4'},{'d3','d4'}};
region_bounds  = {[-10 0 -10 10],[-10 10 -10 0],[-10 10 0 10],[0 10 -10 10]};
n_regions = 4;

S      = B / Ru * B';
t_grid = linspace(0, T, N_time)';
n      = 2;

region_face = {[0.93 0.95 1.0],[1.0 0.97 0.93],[0.93 1.0 0.95],[1.0 0.93 0.93]};

% =========================================================================
%  ETAPE 0 : T1_esc (minimum sur toutes les cibles)
% =========================================================================
fprintf('\n--- VERIFICATION T1_esc (Extended Modal) ---\n');
T1_esc = compute_T1_esc_het(Rx, Rd, lambda, target_names, S, M);
fprintf('  T=%.3f   T1_esc=%.3f\n', T, T1_esc);
if T >= T1_esc
    warning('T >= T1_esc ! Reduire T ou les lambda_j.');
else
    fprintf('  OK (marge=%.3f)\n', T1_esc - T);
end

% =========================================================================
%  ETAPE 1 : Riccati phi_j par cible (eq. 5j)
% =========================================================================
fprintf('\n--- Riccati par cible (Extended Modal)...\n');
tic;
phi_all = struct();
for k = 1:length(target_names)
    dname  = target_names{k};
    lam    = lambda.(dname);
    Rdj    = lam * Rd;
    Qj     = Rdj - Rx;
    phi_all.(dname) = solve_riccati(A, S, Qj, M, t_grid, N_time, n);
    fprintf('  phi_%s calculee (lambda=%.2f, Qj(1,1)=%.2f)\n', dname, lam, Qj(1,1));
end
fprintf('  Riccati terminees en %.2fs\n', toc);

% =========================================================================
%  ETAPE 2 : Agents Monte Carlo
% =========================================================================
rng(42);
samples_MC = sample_initial(N_MC);

% =========================================================================
%  ETAPE 3 : Anderson Mixing (point fixe)
% =========================================================================
fprintf('\n--- Anderson Mixing (m=%d, %d agents MC)...\n', m_anderson, N_MC);
fprintf('  Iter  | err_P   | err_X   | d1     d2     d3     d4\n');
fprintf('  ------+---------+---------+------------------------\n');
tic;
[P_eq, Xhat0_eq, Xr_trajs_eq, psi_all_eq, chi_all_eq, hist_err] = ...
    run_anderson(phi_all, samples_MC, targets, target_names, ...
                 region_targets, region_bounds, A, S, Rx, Rd, lambda, M, ...
                 t_grid, tol_P, tol_X, max_iter, m_anderson, n_regions, n, N_time);
fprintf('  Anderson termine en %.2fs\n', toc);

fprintf('\n--- Resultats equilibre :\n');
for k = 1:4
    p_tot = 0;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            if strcmp(region_targets{r}{j}, target_names{k})
                p_tot = p_tot + P_eq(r,j);
            end
        end
    end
    fprintf('  %s (lambda=%.2f) : %.1f%%\n', target_names{k}, lambda.(target_names{k}), p_tot*100);
end

% =========================================================================
%  ETAPE 4 : Simulation agents
% =========================================================================
fprintf('\n--- Simulation %d agents...\n', N_agents);
rng(0);
x0_sim = sample_initial(N_agents);

[assign_r, assign_d] = assign_targets_het(x0_sim, phi_all, psi_all_eq, chi_all_eq, ...
    region_targets, region_bounds, N_agents, n);

tic;
trajs = simulate_agents_het(x0_sim, assign_r, assign_d, phi_all, psi_all_eq, ...
    Xr_trajs_eq, A, S, t_grid, region_targets, n_regions, n, N_time);
fprintf('  Simulation terminee en %.2fs\n', toc);

% =========================================================================
%  FIGURES
% =========================================================================
color_map_r = [0.30 0.50 0.78; 0.35 0.70 0.40; 0.46 0.71 0.70; 0.95 0.56 0.17;
               0.88 0.34 0.35; 0.69 0.48 0.63; 1.00 0.61 0.65; 0.93 0.79 0.28];
color_keys  = {'1d1','1d2','2d1','2d3','3d2','3d4','4d3','4d4'};

% --- Figure 1 : Convergence Anderson ---
figure('Name','Convergence Anderson Mixing','NumberTitle','off','Position',[50 50 900 400]);
iters = (1:size(hist_err,1))';
ep    = max(hist_err(:,1), 1e-12);   % eviter log(0)
ex    = max(hist_err(:,2), 1e-12);
subplot(1,2,1);
semilogy(iters, ep, '-o', 'Color','b', 'LineWidth',1.5, 'MarkerSize',4); hold on;
semilogy([iters(1) iters(end)],[tol_P tol_P],'r--','LineWidth',1.2);
xlabel('Iteration'); ylabel('||P - P_{old}||');
title('Convergence P - Anderson'); grid on;
subplot(1,2,2);
semilogy(iters, ex, '-s', 'Color',[0.8 0.4 0], 'LineWidth',1.5, 'MarkerSize',4); hold on;
semilogy([iters(1) iters(end)],[tol_X tol_X],'r--','LineWidth',1.2);
xlabel('Iteration'); ylabel('||Xhat(0) - Xhat\_old(0)||');
title('Convergence Xhat(0) - Anderson'); grid on;
ha1 = axes('Position',[0 0 1 1],'Visible','off');
lam_str = sprintf('\\lambda=[%.1f,%.1f,%.1f,%.1f]', ...
    lambda.d1, lambda.d2, lambda.d3, lambda.d4);
text(0.5,0.98,sprintf('Anderson Mixing - Extended Modal (%s)', lam_str), ...
    'Parent',ha1,'HorizontalAlignment','center', ...
    'VerticalAlignment','top','FontSize',11,'FontWeight','bold');

% --- Figure 2 : Trajectoires ---
figure('Name','Agents Trajectories - Extended Modal','NumberTitle','off','Position',[100 100 700 700]);
hold on;
draw_regions(region_bounds, region_face, n_regions);
n_show = min(80, N_agents);
idx_show = randperm(N_agents, n_show);
for k = 1:n_show
    i = idx_show(k);
    col = get_color_rd(assign_r(i), assign_d{i}, color_keys, color_map_r);
    traj = squeeze(trajs(i,:,:));
    plot(traj(:,1), traj(:,2), 'Color',[col 0.35], 'LineWidth', 0.8);
    plot(traj(1,1), traj(1,2), 'k+', 'MarkerSize', 4);
end
draw_targets_lambda(targets, target_names, lambda);
rx_val = eig(Rx); rx_val = rx_val(1);
if rx_val > 0, cpt = 'Crowd avoidance'; else, cpt = 'Crowd following'; end
title(sprintf('%s - Rdj=lam_j*Rd, Rd=%.0f, T=%.0f', cpt, Rd(1,1), T), 'FontSize', 12);
xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;

% --- Figure 3 : Bassins d'attraction quadratiques ---
figure('Name','Bassins - Extended Modal (quadrics)','NumberTitle','off','Position',[200 100 700 700]);
hold on;
[xx, yy] = meshgrid(linspace(-9.9,9.9,80), linspace(-9.9,9.9,80));
gpts     = [xx(:) yy(:)]';
gr       = find_region_vec(gpts, region_bounds);
basin_map = zeros(size(gpts,2), 1);
for i = 1:size(gpts,2)
    r    = gr(i);
    bc   = inf; bi = 1;
    x0i  = gpts(:,i);
    for j = 1:length(region_targets{r})
        dname = region_targets{r}{j};
        phi0j = squeeze(phi_all.(dname)(1,:,:));
        psi0  = squeeze(psi_all_eq(r,j,1,:));
        chi0  = chi_all_eq(r,j,1);
        % Coût quadratique complet (bassins quadratiques, eq. 8 de MFG_etendus)
        c = 0.5*(x0i'*phi0j*x0i) + psi0'*x0i + chi0;
        if c < bc
            bc = c;
            bi = find(strcmp(target_names, dname));
        end
    end
    basin_map(i) = bi;
end
bg    = reshape(basin_map, size(xx));
cmap4 = [0.3 0.5 0.8; 0.2 0.7 0.4; 0.9 0.5 0.2; 0.7 0.3 0.7];
imagesc([-9.9 9.9], [-9.9 9.9], bg);
colormap(cmap4); clim([0.5 4.5]);
img_h = findobj(gca,'Type','image'); img_h.AlphaData = 0.4 * ones(size(bg));
contour(xx, yy, bg, [1.5 2.5 3.5], 'k--', 'LineWidth', 1.5);
for k = 1:N_agents
    di = find(strcmp(target_names, assign_d{k}));
    plot(x0_sim(1,k), x0_sim(2,k), '.', 'Color', cmap4(di,:), 'MarkerSize', 8);
end
draw_targets_lambda(targets, target_names, lambda);
xlabel('X (m)'); ylabel('Y (m)');
title('quadratics bassins D_{r,d_j}(X_r) - Extended Modal', 'FontSize', 12);
xlim([-11 11]); ylim([-11 11]); axis square; grid on;

% --- Figure 4 : Comparaison lambda uniformes vs heterogenes ---
fprintf('\n--- Comparison uniforme vs heterogeneous...\n');
lambda_uniform = struct('d1', 1.0, 'd2', 1.0, 'd3', 1.0, 'd4', 1.0);
lambda_het     = lambda;

sc_lambdas = {lambda_uniform, lambda_het};
sc_names   = {sprintf('Uniforme (lam=1)'), ...
              sprintf('Heterogene [%.1f,%.1f,%.1f,%.1f]', ...
              lambda.d1,lambda.d2,lambda.d3,lambda.d4)};

figure('Name','Uniforme vs Heterogeneous','NumberTitle','off','Position',[50 50 1200 600]);
for sc = 1:2
    lam_sc = sc_lambdas{sc};
    T1_sc  = compute_T1_esc_het(Rx, Rd, lam_sc, target_names, S, M);
    if T >= T1_sc
        fprintf('  Scenario %d : T >= T_esc (%.2f), ignore.\n', sc, T1_sc);
        continue;
    end
    phi_sc = struct();
    for k = 1:length(target_names)
        dn  = target_names{k};
        Qjk = lam_sc.(dn)*Rd - Rx;
        phi_sc.(dn) = solve_riccati(A, S, Qjk, M, t_grid, N_time, n);
    end
    [P_sc, ~, Xr_sc, psi_sc, chi_sc, ~] = run_anderson(phi_sc, samples_MC, ...
        targets, target_names, region_targets, region_bounds, A, S, Rx, Rd, lam_sc, M, ...
        t_grid, tol_P, tol_X, max_iter, m_anderson, n_regions, n, N_time);
    totals = zeros(1,4);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            di = find(strcmp(target_names, region_targets{r}{j}));
            totals(di) = totals(di) + P_sc(r,j);
        end
    end
    pct = sprintf('d1=%.0f%% d2=%.0f%% d3=%.0f%% d4=%.0f%%', ...
        totals(1)*100, totals(2)*100, totals(3)*100, totals(4)*100);
    fprintf('  %s : %s\n', sc_names{sc}, pct);
    rng(sc+10);
    x0_sc  = sample_initial(250);
    [ar_sc, ad_sc] = assign_targets_het(x0_sc, phi_sc, psi_sc, chi_sc, ...
        region_targets, region_bounds, 250, n);
    tr_sc = simulate_agents_het(x0_sc, ar_sc, ad_sc, phi_sc, psi_sc, ...
        Xr_sc, A, S, t_grid, region_targets, n_regions, n, N_time);
    subplot(1,2,sc); hold on;
    draw_regions(region_bounds, region_face, n_regions);
    for k = 1:min(70, 250)
        col = get_color_rd(ar_sc(k), ad_sc{k}, color_keys, color_map_r);
        tr  = squeeze(tr_sc(k,:,:));
        plot(tr(:,1), tr(:,2), 'Color',[col 0.4], 'LineWidth', 0.8);
        plot(tr(1,1), tr(1,2), 'k+', 'MarkerSize', 3);
    end
    draw_targets_lambda(targets, target_names, lam_sc);
    title({sc_names{sc}, pct}, 'FontSize', 9);
    xlabel('X (m)'); ylabel('Y (m)');
    xlim([-11 11]); ylim([-11 11]); axis square; grid on;
end
ha4 = axes('Position',[0 0 1 1],'Visible','off');
text(0.5,0.98,'Effet de l heterogeneite lambda_j sur les bassins', ...
    'Parent',ha4,'HorizontalAlignment','center', ...
    'VerticalAlignment','top','FontSize',12,'FontWeight','bold');

fprintf('\nToutes les figures generees.\n');

% =========================================================================
%                         FONCTIONS LOCALES
% =========================================================================

% -------------------------------------------------------------------------
function T1_esc = compute_T1_esc_het(Rx, Rd, lambda, target_names, S, M)
% T1_esc = min sur j de T1_esc^(j)   (Lemma 2*, MFG_etendus)
    t_escs = inf;
    for k = 1:length(target_names)
        lam = lambda.(target_names{k});
        Qj  = lam*Rd - Rx;
        nv  = size(Qj,1);
        for i = 1:nv
            Vii = Qj(i,i); Wii = S(i,i); Mii = M(i,i);
            if Vii < 0
                rt = sqrt(abs(Vii)*Wii);
                t_escs = min(t_escs, (1/rt)*(pi/2 + atan(Mii*sqrt(Wii/abs(Vii)))));
            end
        end
    end
    T1_esc = t_escs;
end

% -------------------------------------------------------------------------
function phi_all = solve_riccati(A, S, Q, M, t_grid, N_time, nv)
    Tv       = t_grid(end);
    tau_eval = Tv - t_grid(end:-1:1);
    rhs  = @(tau, phi_flat) riccati_rhs(tau, phi_flat, A, S, Q, nv);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',Tv/200);
    [~, sol] = ode45(rhs, tau_eval, M(:), opts);
    phi_all  = zeros(N_time, nv, nv);
    for k = 1:N_time
        phi_all(k,:,:) = reshape(sol(N_time+1-k,:), nv, nv);
    end
end

function dphi = riccati_rhs(~, phi_flat, A, S, Q, nv)
    phi  = reshape(phi_flat, nv, nv);
    dphi = -(phi*S*phi - phi*A - A'*phi - Q);
    dphi = dphi(:);
end

% -------------------------------------------------------------------------
function [psi_all, chi_all] = solve_psi_chi_j(phi_j_all, Xr_traj, d_vec, ...
                                   A, S, Rx, Rdj, M, t_grid, N_time, nv)
% Eq. (6j)-(7j) de MFG_etendus : Rdj = lambda_j * Rd
    Tv     = t_grid(end);
    psi_T  = -M * d_vec;
    chi_T  = 0.5 * d_vec' * M * d_vec;
    state0 = [psi_T; chi_T];
    phi_mat = reshape(phi_j_all, N_time, nv*nv);

    tau_eval = Tv - t_grid(end:-1:1);
    rhs  = @(tau, st) psi_chi_rhs_j(tau, st, Tv, phi_mat, Xr_traj, ...
                                     d_vec, A, S, Rx, Rdj, t_grid, nv);
    opts = odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',Tv/200);
    [~, sol] = ode45(rhs, tau_eval, state0, opts);

    psi_all = zeros(N_time, nv);
    chi_all = zeros(N_time, 1);
    for k = 1:N_time
        idx = N_time + 1 - k;
        psi_all(k,:) = sol(idx, 1:nv);
        chi_all(k)   = sol(idx, nv+1);
    end
end

function ds = psi_chi_rhs_j(tau, state, Tv, phi_mat, Xr_traj, d_vec, ...
                              A, S, Rx, Rdj, t_grid, nv)
    t    = max(t_grid(1), min(t_grid(end), Tv - tau));
    phi  = reshape(interp1(t_grid, phi_mat, t, 'linear'), nv, nv);
    Xr   = interp1(t_grid, Xr_traj, t, 'linear')';
    psi  = state(1:nv);
    % eq. (6j) : dpsi/dt = (phi_j*S - A')*psi + Rdj*dj - Rx*Xr
    dpsi_dt = (phi*S - A')*psi + Rdj*d_vec - Rx*Xr;
    % eq. (7j) : dchi/dt = 0.5*psi'*S*psi + 0.5*(Xr'*Rx*Xr - dj'*Rdj*dj)
    dchi_dt = 0.5*psi'*S*psi + 0.5*(Xr'*Rx*Xr - d_vec'*Rdj*d_vec);
    ds = [-dpsi_dt; -dchi_dt];
end

% -------------------------------------------------------------------------
function trajs = simulate_agents_het(x0_all, assign_r, assign_d, phi_all, ...
                     psi_all, Xr_trajs, A, S, t_grid, region_targets, ...
                     n_regions, nv, N_time)
% Simulation avec phi_j dependant de la cible (eq. 8j)
    N_ag  = size(x0_all, 2);
    Tv    = t_grid(end);
    trajs = zeros(N_ag, N_time, nv);

    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            d_name     = region_targets{r}{j};
            idx_agents = find(assign_r == r & strcmp(assign_d, d_name));
            if isempty(idx_agents)
                continue;
            end
            X0_coh   = x0_all(:, idx_agents);
            N_coh    = size(X0_coh, 2);
            psi_traj = squeeze(psi_all(r,j,:,:));   % N_time x nv
            phi_j    = phi_all.(d_name);             % N_time x nv x nv (specifique a dj)

            % Pre-calculer A - S*phi_j sur toute la grille
            ASp = zeros(N_time, nv, nv);
            for k = 1:N_time
                ASp(k,:,:) = A - S*squeeze(phi_j(k,:,:));
            end
            ASp_mat = reshape(ASp, N_time, nv*nv);

            rhs  = @(t, xvec) cohort_rhs(t, xvec, ASp_mat, psi_traj, ...
                                          S, t_grid, nv, N_coh);
            opts = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',Tv/100);
            [~, sol] = ode45(rhs, t_grid, X0_coh(:), opts);
            for ki = 1:N_coh
                cols = (ki-1)*nv + (1:nv);
                trajs(idx_agents(ki),:,:) = sol(:, cols);
            end
        end
    end
end

function dxvec = cohort_rhs(t, xvec, ASp_mat, psi_traj, S, t_grid, nv, N_coh)
    t_   = max(t_grid(1), min(t_grid(end), t));
    ASp  = reshape(interp1(t_grid, ASp_mat, t_, 'linear'), nv, nv);
    psi  = interp1(t_grid, psi_traj, t_, 'linear')';
    Sp   = -S * psi;
    X    = reshape(xvec, nv, N_coh);
    dX   = ASp * X + repmat(Sp, 1, N_coh);
    dxvec = dX(:);
end

% -------------------------------------------------------------------------
function r_all = find_region_vec(x0_all, region_bounds)
    N     = size(x0_all, 2);
    r_all = ones(N, 1);
    for k = 1:length(region_bounds)
        b    = region_bounds{k};
        mask = x0_all(1,:) >= b(1) & x0_all(1,:) < b(2) & ...
               x0_all(2,:) >= b(3) & x0_all(2,:) < b(4);
        r_all(mask) = k;
    end
end

% -------------------------------------------------------------------------
function [assign_r, assign_d] = assign_targets_het(x0_all, phi_all, psi_all, ...
                                    chi_all, region_targets, region_bounds, N, nv)
% Bassins quadratiques : J_{dj}(x) = 0.5*x'*phi_j(0)*x + psi_{r,dj}(0)'*x + chi_{r,dj}(0)
% (Theorem 1* de MFG_etendus - le terme quadratique n'est PLUS commun)
    assign_r = find_region_vec(x0_all, region_bounds);
    assign_d = cell(N, 1);

    for r = 1:max(assign_r)
        idx_r  = find(assign_r == r);
        if isempty(idx_r)
            continue;
        end
        x_r    = x0_all(:, idx_r);
        n_dest = length(region_targets{r});
        costs  = zeros(n_dest, length(idx_r));
        for j = 1:n_dest
            dname  = region_targets{r}{j};
            phi0j  = squeeze(phi_all.(dname)(1,:,:));
            psi0   = squeeze(psi_all(r,j,1,:));
            chi0   = chi_all(r,j,1);
            % Coût quadratique complet (phi_j specifique a dj)
            quad_j = 0.5 * sum(x_r .* (phi0j * x_r), 1);
            costs(j,:) = quad_j + psi0'*x_r + chi0;
        end
        [~, best_j] = min(costs, [], 1);
        for ki = 1:length(idx_r)
            assign_d{idx_r(ki)} = region_targets{r}{best_j(ki)};
        end
    end
end

% -------------------------------------------------------------------------
function [P_out, Xhat0_out, Xr_out, psi_out, chi_out, hist_err] = ...
    run_anderson(phi_all, samples_MC, targets, target_names, ...
                 region_targets, region_bounds, A, S, Rx, Rd, lambda, M, ...
                 t_grid, tol_P, tol_X, max_iter, m_and, n_regions, nv, N_time)
% Anderson Mixing pour le point fixe (P, Xhat0).
% Principe : G(x) = F(x) - x  (residu),  Anderson mixe les iterees.
% Reference : Walker & Ni (2011), SIAM J. Numer. Anal.

    N     = size(samples_MC, 2);
    max_d = max(cellfun(@length, region_targets));

    rng(42);
    P       = zeros(n_regions, max_d);
    Xhat0   = zeros(n_regions, max_d, nv);
    psi_out = zeros(n_regions, max_d, N_time, nv);
    chi_out = zeros(n_regions, max_d, N_time);

    % Initialisation aleatoire
    reg_mc  = find_region_vec(samples_MC, region_bounds);
    dest_mc = cell(N, 1);
    for i = 1:N
        r = reg_mc(i);
        dest_mc{i} = region_targets{r}{randi(length(region_targets{r}))};
    end
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            d_name = region_targets{r}{j};
            idx    = find(reg_mc == r & strcmp(dest_mc, d_name));
            P(r,j) = length(idx) / N;
            if ~isempty(idx)
                Xhat0(r,j,:) = mean(samples_MC(:,idx), 2)';
            end
        end
    end

    % Vecteur d'etat aplati x = [P(:); Xhat0(:)]
    x_cur = state2vec(P, Xhat0);
    dim   = length(x_cur);

    % Historique Anderson : colonnes = iterees passees
    X_hist = zeros(dim, 0);   % iterees x_{k-m..k}
    G_hist = zeros(dim, 0);   % residus G_{k-m..k}

    hist_err = zeros(max_iter, 2);
    n_iter   = 0;

    for it = 1:max_iter
        [P, Xhat0] = vec2state(x_cur, n_regions, max_d, nv);

        % --- Calculer Xr(t) pour chaque region ---
        Xr_cell = compute_Xr(P, Xhat0, region_targets, n_regions, nv, N_time);

        % --- psi et chi par cible (eq. 6j, 7j) ---
        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                dname = region_targets{r}{j};
                lam   = lambda.(dname);
                Rdj   = lam * Rd;
                [psi_rd, chi_rd] = solve_psi_chi_j(phi_all.(dname), Xr_cell{r}, ...
                    targets.(dname), A, S, Rx, Rdj, M, t_grid, N_time, nv);
                psi_out(r,j,:,:) = psi_rd;
                chi_out(r,j,:)   = chi_rd;
            end
        end

        % --- Assignation (bassins quadratiques) ---
        [new_ar, new_ad] = assign_targets_het(samples_MC, phi_all, psi_out, ...
            chi_out, region_targets, region_bounds, N, nv);

        % --- Nouvelle estimation (P_new, Xhat0_new) ---
        P_new     = zeros(n_regions, max_d);
        Xhat0_new = Xhat0;
        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                d_name = region_targets{r}{j};
                idx    = find(new_ar == r & strcmp(new_ad, d_name));
                P_new(r,j) = length(idx) / N;
                if ~isempty(idx)
                    Xhat0_new(r,j,:) = mean(samples_MC(:,idx), 2)';
                end
            end
        end

        x_new = state2vec(P_new, Xhat0_new);
        g_cur = x_new - x_cur;   % residu G(x_cur)

        err_P = max(max(abs(P_new - P)));
        err_X = max(max(max(abs(Xhat0_new - Xhat0))));

        n_iter         = it;
        hist_err(it,:) = [err_P, err_X];

        totals_d = zeros(1,4);
        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                di = find(strcmp(target_names, region_targets{r}{j}));
                totals_d(di) = totals_d(di) + P_new(r,j);
            end
        end
        fprintf('  %5d  | %.4f  | %.4f  | %.0f%%  %.0f%%  %.0f%%  %.0f%%\n', ...
            it, err_P, err_X, totals_d(1)*100, totals_d(2)*100, ...
            totals_d(3)*100, totals_d(4)*100);

        if err_P < tol_P && err_X < tol_X
            fprintf('  Converge en %d iterations (Anderson)\n', it);
            x_cur = x_new;
            break;
        end

        % --- Anderson update ---
        X_hist = [X_hist, x_cur];
        G_hist = [G_hist, g_cur];
        if size(X_hist, 2) > m_and
            X_hist = X_hist(:, end-m_and+1:end);
            G_hist = G_hist(:, end-m_and+1:end);
        end

        mk = size(G_hist, 2);
        if mk >= 2
            % Resoudre min ||sum(theta*G)||^2  s.c. sum(theta)=1
            % => least-squares sur DeltaG * theta = -g_cur  (formulation standard)
            DG = diff(G_hist, 1, 2);           % differences des residus
            Dx = diff(X_hist, 1, 2);
            % theta_bar : coefficients de (1-m) termes
            theta = (DG'*DG) \ (DG'*g_cur);
            % x_next = (1-sum(theta))*x_new + Dx*theta  (Walker & Ni eq. 2.4)
            x_cur = x_new - Dx*theta + (x_new - X_hist(:,end))*sum(theta);
            % Plus simplement (formulation equivalente plus stable) :
            x_cur = x_new - (Dx + DG)*theta;
            % Projeter P dans [0,1] apres Anderson
            x_cur = project_state(x_cur, n_regions, max_d, nv);
        else
            % Premiere iteration : simple pas de Picard avec amortissement
            x_cur = 0.5*x_new + 0.5*x_cur;
            x_cur = project_state(x_cur, n_regions, max_d, nv);
        end
    end
    if n_iter == max_iter
        fprintf('  Max iterations atteint\n');
        x_cur = x_new;
    end
    hist_err = hist_err(1:n_iter,:);

    [P_out, Xhat0_out] = vec2state(x_cur, n_regions, max_d, nv);

    % Recalculer Xr final
    Xr_final = compute_Xr(P_out, Xhat0_out, region_targets, n_regions, nv, N_time);
    Xr_out   = struct();
    for r = 1:n_regions
        Xr_out.(sprintf('r%d',r)) = Xr_final{r};
    end
end

% -------------------------------------------------------------------------
function Xr_cell = compute_Xr(P, Xhat0, region_targets, n_regions, nv, N_time)
    Xr_cell = cell(n_regions, 1);
    for r = 1:n_regions
        num = zeros(N_time, nv); denom = 0;
        for rr = 1:n_regions
            for j = 1:length(region_targets{rr})
                if any(strcmp(region_targets{r}, region_targets{rr}{j}))
                    p   = P(rr,j);
                    x0c = squeeze(Xhat0(rr,j,:))';
                    num   = num   + p * repmat(x0c, N_time, 1);
                    denom = denom + p;
                end
            end
        end
        Xr_cell{r} = num / max(denom, 1e-10);
    end
end

% -------------------------------------------------------------------------
% Helpers vectorisation etat
function v = state2vec(P, Xhat0)
    v = [P(:); Xhat0(:)];
end

function [P, Xhat0] = vec2state(v, n_regions, max_d, nv)
    np    = n_regions * max_d;
    P     = reshape(v(1:np), n_regions, max_d);
    Xhat0 = reshape(v(np+1:end), n_regions, max_d, nv);
end

function v = project_state(v, n_regions, max_d, nv)
    np   = n_regions * max_d;
    Pv   = v(1:np);
    Pv   = max(0, Pv);   % P >= 0
    v(1:np) = Pv;
end

% -------------------------------------------------------------------------
function samples = sample_initial(N)
    n1 = floor(N/3); n2 = N - n1;
    s1 = mvnrnd_basic([-8 -8], 40*eye(2), n1);
    s2 = mvnrnd_basic([-8  8], 10*eye(2), n2);
    samples = max(min([s1; s2]', 9.99), -9.99);
end

function samples = mvnrnd_basic(mu, Sigma, N)
    L = chol(Sigma, 'lower');
    samples = (L * randn(length(mu), N))' + repmat(mu, N, 1);
end

% -------------------------------------------------------------------------
function draw_regions(region_bounds, region_face, n_regions)
    for r = 1:n_regions
        b = region_bounds{r};
        fill([b(1) b(2) b(2) b(1)], [b(3) b(3) b(4) b(4)], ...
            region_face{r}, 'EdgeColor',[0.7 0.7 0.7], 'LineWidth',0.5, 'FaceAlpha',0.5);
        cx = (b(1)+b(2))/2; cy = (b(3)+b(4))/2;
        text(cx, cy, sprintf('L%d',r), 'HorizontalAlignment','center', ...
            'FontSize',13, 'Color',[0.6 0.6 0.6]);
    end
end

function draw_targets_lambda(targets, target_names, lambda)
    for k = 1:4
        d   = targets.(target_names{k});
        lam = lambda.(target_names{k});
        scatter(d(1), d(2), 250, 'p', 'MarkerFaceColor','yellow', ...
            'MarkerEdgeColor','black', 'LineWidth',1.5);
        text(d(1)+0.5, d(2)+0.5, sprintf('%s (\\lambda=%.1f)', target_names{k}, lam), ...
            'FontSize', 10, 'FontWeight','bold');
    end
end

function col = get_color_rd(r, d_name, color_keys, color_map_r)
    key = sprintf('%d%s', r, d_name);
    idx = find(strcmp(color_keys, key));
    if ~isempty(idx)
        col = color_map_r(idx,:);
    else
        col = [0.5 0.5 0.5];
    end
end
