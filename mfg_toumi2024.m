% =========================================================================
% MFG - Implementation
% D'apres Toumi, Malhame & Le Ny, Automatica 160 (2024)
%
% UTILISATION : modifier la section PARAMETRES puis faire Run.
% Compatible MATLAB R2026a. Aucune toolbox requise.
%
% =========================================================================
clear; close all; clc;

% =========================================================================
%  PARAMETRES - MODIFIER ICI
% =========================================================================
A  = eye(2);
B  = eye(2);

Rx = 1.0 * eye(2);    % >0 evitement  |  <0 suivi de foule
Rd = 1.0 * eye(2);    % stress vers la cible
Ru = 10.0 * eye(2);   % cout de controle
M  = 100.0 * eye(2);  % cout terminal

T        = 4.0;
N_agents = 400;
N_MC     = 1500;
alpha    = 0.5;
tol_P    = 0.02;
tol_X    = 0.5;
max_iter = 60;
N_time   = 200;

% =========================================================================
%  CONFIGURATION
% =========================================================================
targets      = struct('d1',[-10;-10],'d2',[-10;10],'d3',[10;-10],'d4',[10;10]);
target_names = {'d1','d2','d3','d4'};
region_targets = {{'d1','d2'},{'d1','d3'},{'d2','d4'},{'d3','d4'}};
region_bounds  = {[-10 0 -10 10],[-10 10 -10 0],[-10 10 0 10],[0 10 -10 10]};
n_regions = 4;

S      = B / Ru * B';
Q      = Rd - Rx;
t_grid = linspace(0, T, N_time)';
n      = 2;

region_face = {[0.93 0.95 1.0],[1.0 0.97 0.93],[0.93 1.0 0.95],[1.0 0.93 0.93]};

% =========================================================================
%  ETAPE 0 : T1_esc
% =========================================================================
fprintf('\n--- VERIFICATION T1_esc ---\n');
T1_esc = compute_T1_esc(Rx, Rd, S, M);
fprintf('  T=%.3f   T1_esc=%.3f\n', T, T1_esc);
if T >= T1_esc
    warning('T >= T1_esc ! Reduire T ou Rx.');
else
    fprintf('  OK (marge=%.3f)\n', T1_esc - T);
end

% =========================================================================
%  ETAPE 1 : Riccati
% =========================================================================
fprintf('\n--- Riccati...\n');
tic;
phi_all = solve_riccati(A, S, Q, M, t_grid, N_time, n);
fprintf('  phi calculee en %.2fs\n', toc);

% =========================================================================
%  ETAPE 2 : Agents Monte Carlo
% =========================================================================
rng(42);
samples_MC = sample_initial(N_MC);

% =========================================================================
%  ETAPE 3 : Picard
% =========================================================================
fprintf('\n--- Picard (%d agents MC)...\n', N_MC);
fprintf('  Iter  | err_P   | err_X   | d1     d2     d3     d4\n');
fprintf('  ------+---------+---------+------------------------\n');
tic;
[P_eq, Xhat0_eq, Xr_trajs_eq, psi_all_eq, chi_all_eq, hist_err] = ...
    run_picard(phi_all, samples_MC, targets, target_names, ...
               region_targets, region_bounds, A, S, Rx, Rd, M, ...
               t_grid, alpha, tol_P, tol_X, max_iter, n_regions, n, N_time);
fprintf('  Picard termine en %.2fs\n', toc);

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
    fprintf('  %s : %.1f%%\n', target_names{k}, p_tot*100);
end

% =========================================================================
%  ETAPE 4 : Simulation agents
% =========================================================================
fprintf('\n--- Simulation %d agents...\n', N_agents);
rng(0);
x0_sim = sample_initial(N_agents);
phi0   = squeeze(phi_all(1,:,:));

[assign_r, assign_d] = assign_targets_vec(x0_sim, phi0, ...
    psi_all_eq, chi_all_eq, region_targets, region_bounds, N_agents, n);

tic;
trajs = simulate_agents_fast(x0_sim, assign_r, assign_d, phi_all, ...
    psi_all_eq, Xr_trajs_eq, A, S, t_grid, region_targets, n_regions, n, N_time);
fprintf('  Simulation terminee en %.2fs\n', toc);

% =========================================================================
%  FIGURES
% =========================================================================
color_map_r = [0.30 0.50 0.78; 0.35 0.70 0.40; 0.46 0.71 0.70; 0.95 0.56 0.17;
               0.88 0.34 0.35; 0.69 0.48 0.63; 1.00 0.61 0.65; 0.93 0.79 0.28];
color_keys  = {'1d1','1d2','2d1','2d3','3d2','3d4','4d3','4d4'};

% --- Figure 1 : Convergence ---
figure('Name','Convergence Picard','NumberTitle','off','Position',[50 50 900 400]);
subplot(1,2,1);
semilogy(hist_err(:,1),'b-o','LineWidth',1.5,'MarkerSize',4); hold on;
plot([1 size(hist_err,1)],[tol_P tol_P],'r--','LineWidth',1.2);
xlabel('Iteration'); ylabel('||P - P_{old}||'); title('Convergence P'); grid on;
subplot(1,2,2);
semilogy(hist_err(:,2), '-s', 'Color',[0.8 0.4 0], 'LineWidth',1.5, 'MarkerSize',4); hold on;
plot([1 size(hist_err,1)],[tol_X tol_X],'r--','LineWidth',1.2);
xlabel('Iteration'); ylabel('||Xhat(0) - Xhat_{old}(0)||'); title('Convergence Xhat(0)'); grid on;
ha1 = axes('Position',[0 0 1 1],'Visible','off');
text(0.5,0.98,'Convergence Picard','Parent',ha1,'HorizontalAlignment','center',...
    'VerticalAlignment','top','FontSize',12,'FontWeight','bold');

% --- Figure 2 : Trajectoires ---
figure('Name','Trajectoires agents','NumberTitle','off','Position',[100 100 700 700]);
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
draw_targets(targets, target_names);
rx_val = eig(Rx); rx_val = rx_val(1);
rd_val = eig(Rd); rd_val = rd_val(1);
if rx_val > 0
    cpt = 'Evitement de congestion';
else
    cpt = 'Suivi de foule';
end
if rd_val > 0
    sstr = sprintf(', Rd=%.0f', rd_val);
else
    sstr = '';
end
title(sprintf('%s (Rx=%g%s, T=%.0f)', cpt, rx_val, sstr, T), 'FontSize', 13);
xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;

% --- Figure 3 : Bassins ---
figure('Name','Bassins d attraction','NumberTitle','off','Position',[200 100 700 700]);
hold on;
[xx, yy] = meshgrid(linspace(-9.9,9.9,80), linspace(-9.9,9.9,80));
gpts     = [xx(:) yy(:)]';
gr       = find_region_vec(gpts, region_bounds);
basin_map = zeros(size(gpts,2), 1);
for i = 1:size(gpts,2)
    r = gr(i);
    bc = inf; bi = 1;
    x0i = gpts(:,i);
    for j = 1:length(region_targets{r})
        psi0 = squeeze(psi_all_eq(r,j,1,:));
        chi0 = chi_all_eq(r,j,1);
        c = 0.5*(x0i'*phi0*x0i) + psi0'*x0i + chi0;
        if c < bc
            bc = c;
            bi = find(strcmp(target_names, region_targets{r}{j}));
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
draw_targets(targets, target_names);
xlabel('X (m)'); ylabel('Y (m)');
title('Bassins d attraction D_{r,d}(X_r)', 'FontSize', 13);
xlim([-11 11]); ylim([-11 11]); axis square; grid on;

% --- Figure 4 : 3 scenarios ---
fprintf('\n--- 3 scenarios...\n');
sc_Rx   = {-1.0*eye(2), 1.0*eye(2),  1.0*eye(2)};
sc_Rd   = { 0.0*eye(2), 0.0*eye(2), 20.0*eye(2)};
sc_name = {'Crowd Following (Rx<0, Rd=0)', ...
           'Crowd Avoidance (Rx>0, Rd=0)', ...
           'Avoidance+Stress (Rx>0, Rd=20)'};
figure('Name','3 scenarios','NumberTitle','off','Position',[50 50 1400 550]);
for sc = 1:3
    Rx_sc = sc_Rx{sc};
    Rd_sc = sc_Rd{sc};
    S_sc  = B / Ru * B';
    Q_sc  = Rd_sc - Rx_sc;
    if T >= compute_T1_esc(Rx_sc, Rd_sc, S_sc, M)
        fprintf('  Scenario %d : T >= T_esc, ignore.\n', sc);
        continue;
    end
    phi_sc = solve_riccati(A, S_sc, Q_sc, M, t_grid, N_time, n);
    [P_sc, dummy_Xh, Xr_sc, psi_sc, chi_sc, dummy_h] = run_picard( ...
        phi_sc, samples_MC, targets, target_names, ...
        region_targets, region_bounds, A, S_sc, Rx_sc, Rd_sc, M, ...
        t_grid, alpha, tol_P, tol_X, max_iter, n_regions, n, N_time);
    totals = zeros(1,4);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            di = find(strcmp(target_names, region_targets{r}{j}));
            totals(di) = totals(di) + P_sc(r,j);
        end
    end
    pct = sprintf('d1=%.0f%% d2=%.0f%% d3=%.0f%% d4=%.0f%%', ...
        totals(1)*100, totals(2)*100, totals(3)*100, totals(4)*100);
    fprintf('  Scenario %d : %s\n', sc, pct);
    rng(sc);
    x0_sc   = sample_initial(250);
    phi0_sc = squeeze(phi_sc(1,:,:));
    [ar_sc, ad_sc] = assign_targets_vec(x0_sc, phi0_sc, psi_sc, chi_sc, ...
        region_targets, region_bounds, 250, n);
    tr_sc = simulate_agents_fast(x0_sc, ar_sc, ad_sc, phi_sc, psi_sc, ...
        Xr_sc, A, S_sc, t_grid, region_targets, n_regions, n, N_time);
    subplot(1,3,sc); hold on;
    draw_regions(region_bounds, region_face, n_regions);
    for k = 1:min(60,250)
        col = get_color_rd(ar_sc(k), ad_sc{k}, color_keys, color_map_r);
        tr  = squeeze(tr_sc(k,:,:));
        plot(tr(:,1), tr(:,2), 'Color',[col 0.4], 'LineWidth', 0.8);
        plot(tr(1,1), tr(1,2), 'k+', 'MarkerSize', 3);
    end
    draw_targets(targets, target_names);
    title({sc_name{sc}, pct}, 'FontSize', 9);
    xlabel('X (m)'); ylabel('Y (m)');
    xlim([-11 11]); ylim([-11 11]); axis square; grid on;
end
ha4 = axes('Position',[0 0 1 1],'Visible','off');
text(0.5,0.98,'Comparaison 3 scenarios','Parent',ha4,'HorizontalAlignment','center',...
    'VerticalAlignment','top','FontSize',12,'FontWeight','bold');

% --- Figure 5 : Effet T ---
T_vals   = [4.0, 5.5];
Rx_f6    = eye(2);
Rd_f6    = zeros(2);
S_f6     = B / Ru * B';
T_esc_f6 = compute_T1_esc(Rx_f6, Rd_f6, S_f6, M);
figure('Name','Effet T','NumberTitle','off','Position',[50 200 1000 500]);
for kT = 1:2
    T_t    = T_vals(kT);
    t_t    = linspace(0, T_t, N_time)';
    subplot(1,2,kT); hold on;
    draw_regions(region_bounds, region_face, n_regions);
    ok = true;
    try
        phi_t = solve_riccati(A, S_f6, Rd_f6-Rx_f6, M, t_t, N_time, n);
    catch
        ok = false;
        title(sprintf('T=%.1f - Riccati instable', T_t));
    end
    if ok
        Xr_ap  = repmat([-8 8], N_time, 1);
        rng(kT*10);
        x0_f6  = sample_initial(150);
        phi0_t = squeeze(phi_t(1,:,:));
        pf6    = zeros(n_regions, 2, N_time, n);
        cf6    = zeros(n_regions, 2, N_time);
        Xr_st  = struct();
        for r = 1:n_regions
            Xr_st.(sprintf('r%d',r)) = Xr_ap;
            for j = 1:length(region_targets{r})
                [pr, cr] = solve_psi_chi(phi_t, Xr_ap, ...
                    targets.(region_targets{r}{j}), ...
                    A, S_f6, Rx_f6, Rd_f6, M, t_t, N_time, n);
                pf6(r,j,:,:) = pr;
                cf6(r,j,:)   = cr;
            end
        end
        [at, ad] = assign_targets_vec(x0_f6, phi0_t, pf6, cf6, ...
            region_targets, region_bounds, 150, n);
        tr_t = simulate_agents_fast(x0_f6, at, ad, phi_t, pf6, Xr_st, ...
            A, S_f6, t_t, region_targets, n_regions, n, N_time);
        if kT == 1
            col_t = [0 0 0];
        else
            col_t = [0.55 0.27 0.07];
        end
        for k = 1:min(60,150)
            tr = squeeze(tr_t(k,:,:));
            if any(abs(tr(:)) > 50)
                continue;
            end
            plot(tr(:,1), tr(:,2), 'Color',[col_t 0.4], 'LineWidth', 0.8);
            plot(tr(1,1), tr(1,2), 'k+', 'MarkerSize', 4);
            for frac = [0.25 0.5 0.75]
                ii = round(frac*(N_time-1)) + 1;
                plot(tr(ii,1), tr(ii,2), 'o', 'Color', col_t, ...
                    'MarkerFaceColor','white', 'MarkerSize', 4);
            end
        end
        draw_targets(targets, target_names);
        if T_t < T_esc_f6
            st = sprintf('T=%.1f < Tesc=%.2f  OK', T_t, T_esc_f6);
        else
            st = sprintf('T=%.1f > Tesc=%.2f  ATTENTION', T_t, T_esc_f6);
        end
        title({sprintf('T=%.1f', T_t), st}, 'FontSize', 10);
    end
    xlabel('X (m)'); ylabel('Y (m)');
    xlim([-11 11]); ylim([-11 11]); axis square; grid on;
end
ha5 = axes('Position',[0 0 1 1],'Visible','off');
text(0.5,0.98,'Effet T sur les trajectoires (Figure 6 du papier)',...
    'Parent',ha5,'HorizontalAlignment','center','VerticalAlignment','top',...
    'FontSize',11,'FontWeight','bold');

fprintf('\nToutes les figures generees.\n');

% =========================================================================
%                      FONCTIONS LOCALES
% =========================================================================

% -------------------------------------------------------------------------
function T1_esc = compute_T1_esc(Rx, Rd, S, M)
    V = Rd - Rx;
    nv = size(V,1);
    t_escs = inf(nv,1);
    for i = 1:nv
        Vii = V(i,i); Wii = S(i,i); Mii = M(i,i);
        if Vii < 0
            absV = abs(Vii);
            rt   = sqrt(absV * Wii);
            t_escs(i) = (1/rt) * (pi/2 + atan(Mii * sqrt(Wii/absV)));
        end
    end
    T1_esc = min(t_escs);
end

% -------------------------------------------------------------------------
function phi_all = solve_riccati(A, S, Q, M, t_grid, N_time, nv)
% UN SEUL ode45 : integre de tau=0 a tau=T (tau = T-t, backward).
% t_eval = T - t_grid inverse pour avoir tau croissant.
    Tv       = t_grid(end);
    tau_eval = Tv - t_grid(end:-1:1);   % croissant, de 0 a T
    rhs  = @(tau, phi_flat) riccati_rhs(tau, phi_flat, A, S, Q, nv);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',Tv/200);
    [dummy_t, sol] = ode45(rhs, tau_eval, M(:), opts);
    % sol(k,:) correspond a tau_eval(k) = T - t_grid(N_time+1-k)
    phi_all = zeros(N_time, nv, nv);
    for k = 1:N_time
        phi_all(k,:,:) = reshape(sol(N_time+1-k,:), nv, nv);
    end
end

function dphi = riccati_rhs(dummy_tau, phi_flat, A, S, Q, nv)
    phi  = reshape(phi_flat, nv, nv);
    dphi = -(phi*S*phi - phi*A - A'*phi - Q);
    dphi = dphi(:);
end

% -------------------------------------------------------------------------
function [psi_all, chi_all] = solve_psi_chi(phi_all, Xr_traj, d_vec, ...
                                 A, S, Rx, Rd, M, t_grid, N_time, nv)
% UN SEUL ode45 backward, avec tous les points en t_eval.
    Tv      = t_grid(end);
    psi_T   = -M * d_vec;
    chi_T   = 0.5 * d_vec' * M * d_vec;
    state0  = [psi_T; chi_T];
    phi_mat = reshape(phi_all, N_time, nv*nv);

    tau_eval = Tv - t_grid(end:-1:1);
    rhs  = @(tau, st) psi_chi_rhs(tau, st, Tv, phi_mat, Xr_traj, ...
                                   d_vec, A, S, Rx, Rd, t_grid, nv);
    opts = odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',Tv/200);
    [dummy_t, sol] = ode45(rhs, tau_eval, state0, opts);

    psi_all = zeros(N_time, nv);
    chi_all = zeros(N_time, 1);
    for k = 1:N_time
        idx = N_time + 1 - k;
        psi_all(k,:) = sol(idx, 1:nv);
        chi_all(k)   = sol(idx, nv+1);
    end
end

function ds = psi_chi_rhs(tau, state, Tv, phi_mat, Xr_traj, d_vec, ...
                           A, S, Rx, Rd, t_grid, nv)
    t   = max(t_grid(1), min(t_grid(end), Tv - tau));
    phi = reshape(interp1(t_grid, phi_mat, t, 'linear'), nv, nv);
    Xr  = interp1(t_grid, Xr_traj, t, 'linear')';
    psi = state(1:nv);
    dpsi_dt = (phi*S - A') * psi + Rd*d_vec - Rx*Xr;
    dchi_dt = 0.5*psi'*S*psi + 0.5*(Xr'*Rx*Xr - d_vec'*Rd*d_vec);
    % d/dtau = -d/dt  (changement de variable tau = T-t)
    ds = [-dpsi_dt; -dchi_dt];
end

% -------------------------------------------------------------------------
function trajs = simulate_agents_fast(x0_all, assign_r, assign_d, phi_all, ...
                     psi_all, Xr_trajs, A, S, t_grid, region_targets, ...
                     n_regions, nv, N_time)
% 1 ode45 par cohort (r,d) : integre tous ses agents simultanement.
% State = colonnes [x1 x2 ... xN_coh] empilees => nv*N_coh variables.
    N_ag  = size(x0_all, 2);
    Tv    = t_grid(end);
    trajs = zeros(N_ag, N_time, nv);

    % Pre-calculer A - S*phi sur toute la grille : N_time matrices nv x nv
    ASp = zeros(N_time, nv, nv);
    for k = 1:N_time
        ASp(k,:,:) = A - S * squeeze(phi_all(k,:,:));
    end
    ASp_mat = reshape(ASp, N_time, nv*nv);

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

            rhs  = @(t, xvec) cohort_rhs(t, xvec, ASp_mat, psi_traj, ...
                                          S, t_grid, nv, N_coh);
            opts = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',Tv/100);
            [dummy_t, sol] = ode45(rhs, t_grid, X0_coh(:), opts);
            % sol : N_time x (nv*N_coh)
            for ki = 1:N_coh
                cols = (ki-1)*nv + (1:nv);
                trajs(idx_agents(ki),:,:) = sol(:, cols);
            end
        end
    end
end

function dxvec = cohort_rhs(t, xvec, ASp_mat, psi_traj, S, t_grid, nv, N_coh)
    t_      = max(t_grid(1), min(t_grid(end), t));
    ASp     = reshape(interp1(t_grid, ASp_mat, t_, 'linear'), nv, nv);
    psi     = interp1(t_grid, psi_traj, t_, 'linear')';
    Sp      = -S * psi;
    X       = reshape(xvec, nv, N_coh);
    dX      = ASp * X + repmat(Sp, 1, N_coh);
    dxvec   = dX(:);
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

function r = find_region(x0, region_bounds)
    r = 1;
    for k = 1:length(region_bounds)
        b = region_bounds{k};
        if x0(1) >= b(1) && x0(1) < b(2) && x0(2) >= b(3) && x0(2) < b(4)
            r = k;
            return;
        end
    end
end

% -------------------------------------------------------------------------
function [assign_r, assign_d] = assign_targets_vec(x0_all, phi0, psi_all, ...
                                    chi_all, region_targets, region_bounds, N, nv)
% Vectorise : calcule J_d(x0) pour tous les agents en une seule operation.
% Le terme quadratique 0.5*x'*phi0*x est commun a toutes les destinations :
% il ne change pas le choix => seuls psi_d'*x + chi_d determinent d*.
    assign_r = find_region_vec(x0_all, region_bounds);
    assign_d = cell(N, 1);

    for r = 1:max(assign_r)
        idx_r  = find(assign_r == r);
        if isempty(idx_r)
            continue;
        end
        x_r    = x0_all(:, idx_r);
        n_dest = length(region_targets{r});
        quad   = 0.5 * sum(x_r .* (phi0 * x_r), 1);   % 1 x N_r (commun)
        costs  = zeros(n_dest, length(idx_r));
        for j = 1:n_dest
            psi0 = squeeze(psi_all(r,j,1,:));
            chi0 = chi_all(r,j,1);
            costs(j,:) = quad + psi0' * x_r + chi0;
        end
        [dummy_v, best_j] = min(costs, [], 1);
        for ki = 1:length(idx_r)
            assign_d{idx_r(ki)} = region_targets{r}{best_j(ki)};
        end
    end
end

% -------------------------------------------------------------------------
function [P_out, Xhat0_out, Xr_out, psi_out, chi_out, hist_err] = ...
    run_picard(phi_all, samples_MC, targets, target_names, ...
               region_targets, region_bounds, A, S, Rx, Rd, M, ...
               t_grid, alpha, tol_P, tol_X, max_iter, n_regions, nv, N_time)

    N     = size(samples_MC, 2);
    max_d = max(cellfun(@length, region_targets));

    rng(42);
    P       = zeros(n_regions, max_d);
    Xhat0   = zeros(n_regions, max_d, nv);
    psi_out = zeros(n_regions, max_d, N_time, nv);
    chi_out = zeros(n_regions, max_d, N_time);

    % Initialisation : assigner aleatoirement chaque agent
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
            if not(isempty(idx))
                Xhat0(r,j,:) = mean(samples_MC(:,idx), 2)';
            end
        end
    end

    hist_err = zeros(max_iter, 2);
    n_iter   = 0;

    for it = 1:max_iter
        P_old     = P;
        Xhat0_old = Xhat0;

        % Xr(t) = moyenne ponderee des Xhat0 (approximation constante)
        Xr_cell = cell(n_regions, 1);
        for r = 1:n_regions
            num = zeros(N_time, nv); denom = 0;
            for rr = 1:n_regions
                for j = 1:length(region_targets{rr})
                    if any(strcmp(region_targets{r}, region_targets{rr}{j}))
                        p    = P(rr,j);
                        x0c  = squeeze(Xhat0(rr,j,:))';
                        num   = num   + p * repmat(x0c, N_time, 1);
                        denom = denom + p;
                    end
                end
            end
            Xr_cell{r} = num / max(denom, 1e-10);
        end

        % psi et chi : 1 seul ode45 par paire (r,d)
        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                [psi_rd, chi_rd] = solve_psi_chi(phi_all, Xr_cell{r}, ...
                    targets.(region_targets{r}{j}), ...
                    A, S, Rx, Rd, M, t_grid, N_time, nv);
                psi_out(r,j,:,:) = psi_rd;
                chi_out(r,j,:)   = chi_rd;
            end
        end

        % Assignation vectorisee
        phi0 = squeeze(phi_all(1,:,:));
        [new_ar, new_ad] = assign_targets_vec(samples_MC, phi0, psi_out, ...
            chi_out, region_targets, region_bounds, N, nv);

        % Mise a jour P et Xhat0
        P_new     = zeros(n_regions, max_d);
        Xhat0_new = Xhat0;
        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                d_name = region_targets{r}{j};
                idx    = find(new_ar == r & strcmp(new_ad, d_name));
                P_new(r,j) = length(idx) / N;
                if not(isempty(idx))
                    Xhat0_new(r,j,:) = mean(samples_MC(:,idx), 2)';
                end
            end
        end

        err_P = max(max(abs(P_new - P_old)));
        err_X = max(max(max(abs(Xhat0_new - Xhat0_old))));
        P     = alpha * P_new     + (1-alpha) * P_old;
        Xhat0 = alpha * Xhat0_new + (1-alpha) * Xhat0_old;

        n_iter         = it;
        hist_err(it,:) = [err_P, err_X];

        totals_d = zeros(1,4);
        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                di = find(strcmp(target_names, region_targets{r}{j}));
                totals_d(di) = totals_d(di) + P(r,j);
            end
        end
        fprintf('  %5d  | %.4f  | %.4f  | %.0f%%  %.0f%%  %.0f%%  %.0f%%\n', ...
            it, err_P, err_X, totals_d(1)*100, totals_d(2)*100, ...
            totals_d(3)*100, totals_d(4)*100);

        if err_P < tol_P && err_X < tol_X
            fprintf('  Converge en %d iterations\n', it);
            break;
        end
    end
    if n_iter == max_iter
        fprintf('  Max iterations atteint\n');
    end
    hist_err = hist_err(1:n_iter,:);

    P_out     = P;
    Xhat0_out = Xhat0;

    Xr_out = struct();
    for r = 1:n_regions
        num = zeros(N_time, nv); denom = 0;
        for rr = 1:n_regions
            for j = 1:length(region_targets{rr})
                if any(strcmp(region_targets{r}, region_targets{rr}{j}))
                    p    = P(rr,j);
                    x0c  = squeeze(Xhat0(rr,j,:))';
                    num   = num   + p * repmat(x0c, N_time, 1);
                    denom = denom + p;
                end
            end
        end
        Xr_out.(sprintf('r%d',r)) = num / max(denom, 1e-10);
    end
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

function draw_targets(targets, target_names)
    for k = 1:4
        d = targets.(target_names{k});
        scatter(d(1), d(2), 250, 'p', 'MarkerFaceColor','yellow', ...
            'MarkerEdgeColor','black', 'LineWidth',1.5);
        text(d(1)+0.5, d(2)+0.5, sprintf('d%d',k), 'FontSize',12, 'FontWeight','bold');
    end
end

function col = get_color_rd(r, d_name, color_keys, color_map_r)
    key = sprintf('%d%s', r, d_name);
    idx = find(strcmp(color_keys, key));
    if not(isempty(idx))
        col = color_map_r(idx,:);
    else
        col = [0.5 0.5 0.5];
    end
end
