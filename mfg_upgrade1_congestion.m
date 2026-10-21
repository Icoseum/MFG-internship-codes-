% =========================================================================
% MFG UPGRADE I - Correction predictive de congestion 
% D'apres Toumi, Malhame & Le Ny (Automatica 160, 2024) + extension sec 3.2.4 
%
% Implemente precisement l'etape 2a de la Section 3.2.4 :
%   Step 0 (baseline)  : resoudre le modele heterogene (Section 3.1),
%                        congestion-blind (mu = 1 partout).
%   Step 1 (diagnostic): calculer le signal de congestion enrichi
%                        G_tilde_{r,dj}(t) (eq. 43) le long de la
%                        trajectoire d'equilibre courante, et identifier
%                        les paires (r,dj) congestionnees Z_cong (eq. 44).
%   Step 2a (correction): retuner lambda_j -> lambda_j * Lambda(sup G_tilde)
%                        via la carte de niveaux discretisee Lambda de la
%                        Section 3.2.5 (eq. 51), PUIS re-resoudre entierement
%                        le modele heterogene (Theoreme 5) avec ce nouveau
%                        lambda_eff(r,dj). On itere (eq. 45) jusqu'a
%                        stabilisation de l'assignation de niveaux (Z_cong)
%                        ou jusqu'a un budget fixe d'iterations exterieures.
%
% Difference structurelle cle vs le modele heterogene "pur" (Section 3.1) :
%   la ou lambda_j n'y depend que de la cible dj, la correction de
%   congestion depend de la PAIRE (region, cible) (Remarque 11 du rapport),
%   puisque deux cohortes visant la meme cible depuis des regions
%   differentes ne percoivent en general pas la meme congestion anticipee.
%   phi, psi, chi sont donc ici indexes par paire (r,j), pas seulement
%   par cible d.
%
% Une regle d'hysteresis (Remarque 6) evite le battement d'un niveau a
% l'autre d'une iteration exterieure a l'autre : on n'ajoute une paire au
% niveau superieur qu'au-dela d'un seuil haut Gtol_up, et on ne la
% redescend qu'en-deca d'un seuil bas Gtol_down < Gtol_up.
%
% UTILISATION : modifier la section PARAMETRES puis Run.
% Compatible MATLAB / GNU Octave. Aucune toolbox requise.
% =========================================================================
clear; close all; clc;

MAKE_PLOTS = true;   % mettre a false pour ne calculer que le point fixe
                      % (utile pour tester la logique sans generer de figures


% =========================================================================
%  PARAMETRES - MODELE DE BASE (Section 1 / 2.1)
% =========================================================================
A = zeros(2);

B = eye(2);

Rx = 1.0*eye(2); Rd = eye(2); Ru = 10.0*eye(2); M = 100*eye(2); T = 4.0;
lambda_target = struct('d1',1.0,'d2',2.0,'d3',20.0,'d4',1.3);   

N_agents = 1000;
N_MC     = 1200;
tol_P    = 0.02;
tol_X    = 0.5;
max_iter = 80;      % budget Anderson par resolution interne (par etape ext.)
N_time   = 200;
m_anderson = 5;

% =========================================================================
%  PARAMETRES - CONGESTION (Section 2.2.2, 2.2.5) ET BOUCLE EXTERIEURE (2.2.4)
% =========================================================================
eps_kernel = 1.2;     % bande passante du noyau mollifie Kε (eq. 33, Remarque 3)
kappa_D    = 1.0;     % poids de l'indice de desorganisation directionnelle (eq. 38)
neighbor_frac = 0.05; % seuil relatif definissant le voisinage N(r,dj) (eq. 37)

% Carte de niveaux discretisee Lambda (eq. 46), regime "panique" (croissant) :
% une paire (r,dj) plus congestionnee voit son cout de cible augmenter,
% ce qui (Proposition 3) retrecit son bassin et detourne la masse vers
% les cibles voisines moins chargees.
kappa_cong  = 8000;    % 4x plus agressif
p_cong      = 2.5;
alpha_relax = 0.25;    % plus de damping pour casser le cycle-limite
lambda_tol  = 0.02;
K_out_max   = 25;

% =========================================================================
%  CONFIGURATION GEOMETRIQUE (identique a Toumi / modele heterogene)
% =========================================================================
targets      = struct('d1',[-10;-10],'d2',[-10;10],'d3',[10;-10],'d4',[10;10]);
target_names = {'d1','d2','d3','d4'};
region_targets = {{'d1','d2'},{'d1','d3'},{'d2','d4'},{'d3','d4'}};
region_bounds  = {[-10 0 -10 10],[-10 10 -10 0],[-10 10 0 10],[0 10 -10 10]};
n_regions = 4;
max_d     = max(cellfun(@length, region_targets));

S      = B / Ru * B';
t_grid = linspace(0, T, N_time)';
n      = 2;

region_face = {[0.93 0.95 1.0],[1.0 0.97 0.93],[0.93 1.0 0.95],[1.0 0.93 0.93]};

fprintf('=== MFG Upgrade I : correction predictive de congestion ===\n');
fprintf('    (Section 2.2.4, etape 2a du rapport)\n\n');

% =========================================================================
%  ETAPE 0 : SOLVE BASELINE (congestion-blind, mode = 1 partout, mu = 1)
% =========================================================================
fprintf('--- STEP 0 : equilibre de reference (congestion-blind) ---\n');
 
lambda_eff = build_lambda_eff_cont(lambda_target, ...
    zeros(n_regions, max_d), region_targets, target_names, kappa_cong, p_cong);

check_horizon(lambda_eff, Rd, Rx, S, M, T, region_targets, n_regions);

phi_pair = solve_riccati_pairs(lambda_eff, Rd, Rx, A, S, M, t_grid, ...
                                N_time, n, region_targets, n_regions);

rng(42);
samples_MC = sample_initial(N_MC);

[P_cur, Xhat0_cur, psi_cur, chi_cur, Xr_cur, hist0] = run_anderson_pairs( ...
    phi_pair, samples_MC, targets, region_targets, region_bounds, ...
    A, S, Rx, Rd, lambda_eff, M, t_grid, tol_P, tol_X, max_iter, ...
    m_anderson, n_regions, n, N_time);

print_split(P_cur, region_targets, target_names, n_regions, 'Step 0 (baseline)');

% Trajectoires de cohorte (moyennes) associees a cet equilibre, necessaires
% pour evaluer le signal de congestion (eq. 33-38).
[Xhat_traj_cur, Xhat_dot_cur] = compute_cohort_traj(phi_pair, psi_cur, ...
    Xhat0_cur, A, S, t_grid, region_targets, n_regions, n, N_time);

Gamma_cur = compute_congestion(Xhat_traj_cur, Xhat_dot_cur, P_cur, ...
    region_targets, n_regions, eps_kernel, kappa_D, neighbor_frac, N_time);

fprintf('  Congestion max anticipee (baseline) :\n');
print_gamma(Gamma_cur, region_targets, n_regions);

% Historiques pour les figures de convergence de la boucle exterieure
lambda_hist = {lambda_eff};
Gamma_hist  = {Gamma_cur};
P_hist      = {P_cur};
label_hist  = {'Step 0'};

phi_pair_base   = phi_pair;   % conserve pour comparaison finale (Fig. 4)
Xhat_traj_base  = Xhat_traj_cur;
P_base          = P_cur;
psi_base        = psi_cur;
chi_base        = chi_cur;
Xr_base         = Xr_cur;

% =========================================================================
%  BOUCLE EXTERIEURE (eq. 40) : diagnostic -> correction 2a -> re-solve
% =========================================================================
fprintf('\n--- BOUCLE EXTERIEURE (Upgrade I, etape 2a) ---\n');

converged_outer = false;
for k = 1:K_out_max

        lambda_target_new = build_lambda_eff_cont(lambda_target, Gamma_cur, ...
        region_targets, target_names, kappa_cong, p_cong);
    lambda_eff_new = alpha_relax*lambda_target_new + (1-alpha_relax)*lambda_eff;

    rel_change = max(max(abs(lambda_eff_new - lambda_eff) ./ max(lambda_eff,1e-8)));
    fprintf('\n  [Outer %d] variation relative max de lambda_eff : %.4f\n', k, rel_change);

    if rel_change < lambda_tol
        fprintf('  lambda_eff stabilise (Lambda continue, eq. 41). Arret (eq. 40).\n');
        converged_outer = true;
        break;
    end

    % --- Step 2a : retuning lambda_j -> lambda_j * Lambda(sup G_tilde) ---
    lambda_eff = lambda_eff_new;
    check_horizon(lambda_eff, Rd, Rx, S, M, T, region_targets, n_regions);

    phi_pair = solve_riccati_pairs(lambda_eff, Rd, Rx, A, S, M, t_grid, ...
                                    N_time, n, region_targets, n_regions);

    % --- Re-solve complet du modele heterogene corrige (Theoreme 5) ---
    [P_cur, Xhat0_cur, psi_cur, chi_cur, Xr_cur, hist_k] = run_anderson_pairs( ...
        phi_pair, samples_MC, targets, region_targets, region_bounds, ...
        A, S, Rx, Rd, lambda_eff, M, t_grid, tol_P, tol_X, max_iter, ...
        m_anderson, n_regions, n, N_time);

    print_split(P_cur, region_targets, target_names, n_regions, ...
        sprintf('Step %d (corrige)', k));

    [Xhat_traj_cur, Xhat_dot_cur] = compute_cohort_traj(phi_pair, psi_cur, ...
        Xhat0_cur, A, S, t_grid, region_targets, n_regions, n, N_time);

    Gamma_cur = compute_congestion(Xhat_traj_cur, Xhat_dot_cur, P_cur, ...
        region_targets, n_regions, eps_kernel, kappa_D, neighbor_frac, N_time);

    fprintf('  Congestion max anticipee (apres correction) :\n');
    print_gamma(Gamma_cur, region_targets, n_regions);

    lambda_hist{end+1} = lambda_eff; %#ok<SAGROW>
    Gamma_hist{end+1}  = Gamma_cur;  %#ok<SAGROW>
    P_hist{end+1}      = P_cur;      %#ok<SAGROW>
    label_hist{end+1}  = sprintf('Step %d', k); %#ok<SAGROW>
end
if not(converged_outer)
    fprintf('\n  Budget K_out_max=%d atteint sans stabilisation complete (cf. Remarque 6).\n', K_out_max);
end

P_final         = P_cur;
Xhat0_final     = Xhat0_cur;
psi_final       = psi_cur;
chi_final       = chi_cur;
Xr_final        = Xr_cur;
phi_pair_final  = phi_pair;
Xhat_traj_final = Xhat_traj_cur;
Gamma_final     = Gamma_cur;

% =========================================================================
%  SIMULATION D'AGENTS : baseline (Step 0) vs corrige (Upgrade I)
% =========================================================================
fprintf('\n--- Simulation %d agents (baseline vs corrige)...\n', N_agents);
rng(0);
x0_sim = sample_initial(N_agents);

[ar_base, ad_base] = assign_targets_pairs(x0_sim, phi_pair_base, psi_base, ...
    chi_base, region_targets, region_bounds, N_agents, n);
trajs_base = simulate_agents_pairs(x0_sim, ar_base, ad_base, phi_pair_base, ...
    psi_base, A, S, t_grid, region_targets, n_regions, n, N_time);

[ar_fin, ad_fin] = assign_targets_pairs(x0_sim, phi_pair_final, psi_final, ...
    chi_final, region_targets, region_bounds, N_agents, n);
trajs_fin = simulate_agents_pairs(x0_sim, ar_fin, ad_fin, phi_pair_final, ...
    psi_final, A, S, t_grid, region_targets, n_regions, n, N_time);

fprintf('  Simulation terminee.\n');

% =========================================================================
%  FIGURES
% =========================================================================
if MAKE_PLOTS
    color_map_r = [0.30 0.50 0.78; 0.35 0.70 0.40; 0.46 0.71 0.70; 0.95 0.56 0.17;
                   0.88 0.34 0.35; 0.69 0.48 0.63; 1.00 0.61 0.65; 0.93 0.79 0.28];
    color_keys  = {'1d1','1d2','2d1','2d3','3d2','3d4','4d3','4d4'};

    % --- Figure 1 : convergence Anderson de la derniere resolution interne ---
    figure('Name','Convergence Anderson (derniere etape exterieure)', ...
        'NumberTitle','off','Position',[50 50 900 400]);
    subplot(1,2,1);
    semilogy(hist_safe(hist0(:,1)),'b-o','LineWidth',1.2,'MarkerSize',3); hold on;
    plot([1 max(size(hist0,1),1)],[tol_P tol_P],'r--','LineWidth',1.2);
    xlabel('Iteration interne'); ylabel('||P-P_{old}||');
    title('Step 0 : convergence P'); grid on;
    subplot(1,2,2);
    semilogy(hist_safe(hist0(:,2)), '-s',...
        'Color', [0.8 0.4 0], ...
        'LineWidth', 1.2, ...
        'MarkerSize', 3);
    hold on;
    plot([1 max(size(hist0,1),1)],[tol_X tol_X],'r--','LineWidth',1.2);
    xlabel('Iteration interne'); ylabel('||Xhat(0)-Xhat_{old}(0)||');
    title('Step 0 : convergence Xhat(0)'); grid on;

    % --- Figure 2 : boucle exterieure - evolution de Gamma_max et lambda_eff ---
    figure('Name','Boucle exterieure Upgrade I','NumberTitle','off', ...
        'Position',[80 80 1000 450]);
    K = length(Gamma_hist);
    subplot(1,2,1); hold on;
    pair_labels = {};
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            gline = zeros(K,1);
            for k = 1:K
                gline(k) = Gamma_hist{k}(r,j);
            end
            plot(0:K-1, gline, '-o', 'LineWidth', 1.3, 'MarkerSize', 4);
            pair_labels{end+1} = sprintf('L%d-%s', r, region_targets{r}{j}); %#ok<SAGROW>
        end
    end
    yl = ylim;
    legend(pair_labels, 'Location','eastoutside','FontSize',7);
    xlabel('Iteration exterieure k'); ylabel('\Gamma_{r,d_j} = sup_t G\_tilde');
    title('Congestion anticipee par paire (r,d_j)'); grid on;

    subplot(1,2,2); hold on;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            lline = zeros(K,1);
            for k = 1:K
                lline(k) = lambda_hist{k}(r,j);
            end
            plot(0:K-1, lline, '-s', 'LineWidth', 1.3, 'MarkerSize', 4);
        end
    end
    xlabel('Iteration exterieure k'); ylabel('$\lambda_{\mathrm{eff}}(r,d_j) = \lambda_j \cdot \mu_\ell$', ...
    'Interpreter', 'latex');
    title('Coefficient de stress corrige par paire'); grid on;
    ha2 = axes('Position',[0 0 1 1],'Visible','off');
    text(0.5,0.98,'Convergence de la boucle exterieure (eq. 40)','Parent',ha2, ...
        'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    % --- Figure 3 : trajectoires baseline vs corrige ---
    figure('Name','Trajectoires : baseline vs Upgrade I','NumberTitle','off', ...
        'Position',[100 100 1300 650]);
    n_show = min(80, N_agents);
    idx_show = randperm(N_agents, n_show);

    subplot(1,2,1); hold on;
    draw_regions(region_bounds, region_face, n_regions);
    for k = 1:n_show
        i = idx_show(k);
        col = get_color_rd(ar_base(i), ad_base{i}, color_keys, color_map_r);
        traj = squeeze(trajs_base(i,:,:));
        plot(traj(:,1), traj(:,2), 'Color', col, 'LineWidth', 0.8);
        plot(traj(1,1), traj(1,2), 'k+', 'MarkerSize', 4);
    end
    draw_targets(targets, target_names);
    title('Step 0 : congestion-blind', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;

    subplot(1,2,2); hold on;
    draw_regions(region_bounds, region_face, n_regions);
    for k = 1:n_show
        i = idx_show(k);
        col = get_color_rd(ar_fin(i), ad_fin{i}, color_keys, color_map_r);
        traj = squeeze(trajs_fin(i,:,:));
        plot(traj(:,1), traj(:,2), 'Color', col, 'LineWidth', 0.8);
        plot(traj(1,1), traj(1,2), 'k+', 'MarkerSize', 4);
    end
    draw_targets(targets, target_names);
    title('Upgrade I : corrige (etape 2a)', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;
    ha3 = axes('Position',[0 0 1 1],'Visible','off');
    text(0.5,0.98,'Effet de la correction predictive de congestion sur les trajectoires', ...
        'Parent',ha3,'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    % --- Figure 4 : bassins quadratiques baseline vs corrige ---
    figure('Name','Bassins : baseline vs Upgrade I','NumberTitle','off', ...
        'Position',[150 100 1300 650]);
    [xx, yy] = meshgrid(linspace(-9.9,9.9,70), linspace(-9.9,9.9,70));
    gpts     = [xx(:) yy(:)]';
    gr       = find_region_vec(gpts, region_bounds);
    cmap4    = [0.3 0.5 0.8; 0.2 0.7 0.4; 0.9 0.5 0.2; 0.7 0.3 0.7];

    subplot(1,2,1); hold on;
    bg_base = compute_basin_map(gpts, gr, phi_pair_base, psi_base, chi_base, ...
        region_targets, target_names, n_regions);
    imagesc([-9.9 9.9], [-9.9 9.9], reshape(bg_base, size(xx)));
    colormap(cmap4); clim([0.5 4.5]);
    h_img = findobj(gca, 'Type', 'image', '-depth', 1);
if ~isempty(h_img)
    h_img(1).AlphaData = 0.45 * ones(size(xx));
end
    contour(xx, yy, reshape(bg_base, size(xx)), [1.5 2.5 3.5], 'k--', 'LineWidth', 1.2);
    draw_targets(targets, target_names);
    title('Bassins - Step 0 (congestion-blind)', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;

    subplot(1,2,2); hold on;
    bg_fin = compute_basin_map(gpts, gr, phi_pair_final, psi_final, chi_final, ...
        region_targets, target_names, n_regions);
    imagesc([-9.9 9.9], [-9.9 9.9], reshape(bg_fin, size(xx)));
    colormap(cmap4); clim([0.5 4.5]);
    h_img = findobj(gca, 'Type', 'image', '-depth', 1);
if ~isempty(h_img)
    h_img(1).AlphaData = 0.45 * ones(size(xx));
end
    contour(xx, yy, reshape(bg_fin, size(xx)), [1.5 2.5 3.5], 'k--', 'LineWidth', 1.2);
    draw_targets(targets, target_names);
    title('Bassins - Upgrade I (corrige)', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;
    ha4 = axes('Position',[0 0 1 1],'Visible','off');
    text(0.5,0.98,'Retrecissement des bassins des cibles congestionnees (Proposition 3)', ...
        'Parent',ha4,'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    % --- Figure 5 : repartition des cibles, Step 0 -> Step final ---
    figure('Name','Repartition des cibles','NumberTitle','off','Position',[200 150 700 450]);
    K = length(P_hist);
    totals_mat = zeros(K, 4);
    for k = 1:K
        totals_mat(k,:) = target_totals(P_hist{k}, region_targets, target_names, n_regions);
    end
    bar(0:K-1, totals_mat*100, 'stacked');
    legend(target_names, 'Location','eastoutside');
    xlabel('Iteration exterieure k'); ylabel('Part de la population (%)');
    title('Redistribution de la masse entre cibles au fil de la correction', 'FontSize', 11);
    grid on;

    fprintf('\nToutes les figures generees.\n');
end

fprintf('\n=== Termine ===\n');

% =========================================================================
%                      FONCTIONS LOCALES
% =========================================================================

% -------------------------------------------------------------------------
function mode_idx = mask_unused(mode_idx, region_targets)
% Met NaN sur les emplacements (r,j) qui ne correspondent a aucune cible
% candidate de la region r (utile quand max_d > length(region_targets{r})).
    [n_regions, max_d] = size(mode_idx);
    for r = 1:n_regions
        for j = length(region_targets{r})+1 : max_d
            mode_idx(r,j) = NaN;
        end
    end
end

function lambda_eff = build_lambda_eff_cont(lambda_target, Gamma, ...
    region_targets, target_names, kappa_cong, p_cong)
% lambda_eff(r,j) = lambda_j * (1 + kappa_cong*Gamma(r,j)^p_cong)   (Lambda continue convexe, eq. 41)
[n_regions, max_d] = size(Gamma);
lambda_eff = nan(n_regions, max_d);
for r = 1:n_regions
    for j = 1:length(region_targets{r})
        dname = region_targets{r}{j};
        lam_j = lambda_target.(dname);
        lambda_eff(r,j) = lam_j * (1 + kappa_cong*Gamma(r,j)^p_cong);
    end
end
end

% -------------------------------------------------------------------------
function lambda_eff = build_lambda_eff(lambda_target, mode_idx, mu_levels, ...
                                        region_targets, target_names)
% lambda_eff(r,j) = lambda_j * mu_{mode_idx(r,j)}  (eq. 41, 46, 47)
    [n_regions, max_d] = size(mode_idx);
    lambda_eff = nan(n_regions, max_d);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            dname = region_targets{r}{j};
            lam_j = lambda_target.(dname);
            ell   = mode_idx(r,j);
            lambda_eff(r,j) = lam_j * mu_levels(ell);
        end
    end
end

% -------------------------------------------------------------------------
function check_horizon(lambda_eff, Rd, Rx, S, M, T, region_targets, n_regions)
% Verifie T < T1_esc pour chaque paire active (Assomption 3/6).
    Tesc = inf;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            Qj = lambda_eff(r,j)*Rd - Rx;
            nv = size(Qj,1);
            for i = 1:nv
                Vii = Qj(i,i); Wii = S(i,i); Mii = M(i,i);
                if Vii < 0
                    rt = sqrt(abs(Vii)*Wii);
                    Tesc = min(Tesc, (1/rt)*(pi/2 + atan(Mii*sqrt(Wii/abs(Vii)))));
                end
            end
        end
    end
    if T >= Tesc
        warning('T >= T1_esc (%.3f) pour au moins une paire (r,dj) ! Reduire T ou les niveaux mu.', Tesc);
    end
end

% -------------------------------------------------------------------------
function phi_pair = solve_riccati_pairs(lambda_eff, Rd, Rx, A, S, M, ...
                                         t_grid, N_time, nv, region_targets, n_regions)
% Riccati (eq. 11) resolue independamment pour chaque paire (r,dj), puisque
% lambda_eff depend maintenant de la paire et non plus seulement de dj.
    phi_pair = cell(n_regions, 1);
    for r = 1:n_regions
        phi_pair{r} = cell(length(region_targets{r}), 1);
        for j = 1:length(region_targets{r})
            Qj = lambda_eff(r,j)*Rd - Rx;
            phi_pair{r}{j} = solve_riccati(A, S, Qj, M, t_grid, N_time, nv);
        end
    end
end

function phi_all = solve_riccati(A, S, Q, M, t_grid, N_time, nv)
% UN SEUL ode45 (backward via tau = T - t), comme dans le code de base.
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
function [psi_all, chi_all] = solve_psi_chi_pair(phi_pair_rj, Xr_traj, d_vec, ...
                                   A, S, Rx, Rdj, M, t_grid, N_time, nv)
% Eq. (13)-(14) : identique a solve_psi_chi_j mais phi et Rdj sont ceux
% de la paire (r,dj) specifique.
    Tv     = t_grid(end);
    psi_T  = -M * d_vec;
    chi_T  = 0.5 * d_vec' * M * d_vec;
    state0 = [psi_T; chi_T];
    phi_mat = reshape(phi_pair_rj, N_time, nv*nv);

    tau_eval = Tv - t_grid(end:-1:1);
    rhs  = @(tau, st) psi_chi_rhs(tau, st, Tv, phi_mat, Xr_traj, ...
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

function ds = psi_chi_rhs(tau, state, Tv, phi_mat, Xr_traj, d_vec, ...
                            A, S, Rx, Rdj, t_grid, nv)
    t    = max(t_grid(1), min(t_grid(end), Tv - tau));
    phi  = reshape(interp1(t_grid, phi_mat, t, 'linear'), nv, nv);
    Xr   = interp1(t_grid, Xr_traj, t, 'linear')';
    psi  = state(1:nv);
    dpsi_dt = (phi*S - A')*psi + Rdj*d_vec - Rx*Xr;
    dchi_dt = 0.5*psi'*S*psi + 0.5*(Xr'*Rx*Xr - d_vec'*Rdj*d_vec);
    ds = [-dpsi_dt; -dchi_dt];
end

% -------------------------------------------------------------------------
function [P_out, Xhat0_out, psi_out, chi_out, Xr_out, hist_err] = ...
    run_anderson_pairs(phi_pair, samples_MC, targets, region_targets, ...
                        region_bounds, A, S, Rx, Rd, lambda_eff, M, ...
                        t_grid, tol_P, tol_X, max_iter, m_and, n_regions, nv, N_time)
% Point fixe (Anderson mixing, Walker & Ni 2011) generalise : phi et Rdj
% sont indexes par paire (r,j), et non plus seulement par cible.
    N     = size(samples_MC, 2);
    max_d = max(cellfun(@length, region_targets));

    rng(42);
    P       = zeros(n_regions, max_d);
    Xhat0   = zeros(n_regions, max_d, nv);
    psi_out = zeros(n_regions, max_d, N_time, nv);
    chi_out = zeros(n_regions, max_d, N_time);

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

    x_cur = state2vec(P, Xhat0);
    dim   = length(x_cur);
    X_hist = zeros(dim, 0);
    G_hist = zeros(dim, 0);

    hist_err = zeros(max_iter, 2);
    n_iter   = 0;
    x_new = x_cur;

    for it = 1:max_iter
        [P, Xhat0] = vec2state(x_cur, n_regions, max_d, nv);

        Xr_cell = compute_Xr(P, Xhat0, region_targets, n_regions, nv, N_time);

        for r = 1:n_regions
            for j = 1:length(region_targets{r})
                dname = region_targets{r}{j};
                Rdj   = lambda_eff(r,j) * Rd;
                [psi_rd, chi_rd] = solve_psi_chi_pair(phi_pair{r}{j}, ...
                    Xr_cell{r}, targets.(dname), A, S, Rx, Rdj, M, ...
                    t_grid, N_time, nv);
                psi_out(r,j,:,:) = psi_rd;
                chi_out(r,j,:)   = chi_rd;
            end
        end

        [new_ar, new_ad] = assign_targets_pairs(samples_MC, phi_pair, ...
            psi_out, chi_out, region_targets, region_bounds, N, nv);

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

        x_new = state2vec(P_new, Xhat0_new);
        g_cur = x_new - x_cur;

        err_P = max(max(abs(P_new - P)));
        err_X = max(max(max(abs(Xhat0_new - Xhat0))));
        n_iter = it;
        hist_err(it,:) = [err_P, err_X];

        if mod(it,10) == 0 || it == 1
            fprintf('    iter %3d | err_P=%.4f | err_X=%.4f\n', it, err_P, err_X);
        end

        if err_P < tol_P && err_X < tol_X
            x_cur = x_new;
            fprintf('    Converge en %d iterations.\n', it);
            break;
        end

        X_hist = [X_hist, x_cur];
        G_hist = [G_hist, g_cur];
        if size(X_hist, 2) > m_and
            X_hist = X_hist(:, end-m_and+1:end);
            G_hist = G_hist(:, end-m_and+1:end);
        end

        mk = size(G_hist, 2);
        x_prev = x_cur;
        did_anderson = false;
        if mk >= 2
            DG  = diff(G_hist, 1, 2);
            Dx  = diff(X_hist, 1, 2);
            reg = 1e-8 * max(1, norm(DG'*DG, 'fro'));   % regularisation de Tikhonov
            theta = (DG'*DG + reg*eye(mk-1)) \ (DG'*g_cur);
            x_try = x_new - (Dx + DG)*theta;
            if all(isfinite(x_try))
                x_cur = project_state(x_try, n_regions, max_d, nv);
                did_anderson = true;
            end
        end
        if not(did_anderson)
            % Repli : pas de Picard amorti si Anderson est mal conditionne
            % ou produit une iteree non finie (rcond quasi-nul, cf. avertissement).
            x_cur = 0.5*x_new + 0.5*x_prev;
            x_cur = project_state(x_cur, n_regions, max_d, nv);
        end
    end
    if n_iter == max_iter
        fprintf('    Max iterations interne atteint.\n');
        x_cur = x_new;
    end
    hist_err = hist_err(1:n_iter,:);

    [P_out, Xhat0_out] = vec2state(x_cur, n_regions, max_d, nv);
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
function v = state2vec(P, Xhat0)
    v = [P(:); Xhat0(:)];
end

function [P, Xhat0] = vec2state(v, n_regions, max_d, nv)
    np    = n_regions * max_d;
    P     = reshape(v(1:np), n_regions, max_d);
    Xhat0 = reshape(v(np+1:end), n_regions, max_d, nv);
end

function v = project_state(v, n_regions, max_d, nv)
    np = n_regions * max_d;
    Pv = v(1:np);
    Pv = max(0, Pv);
    v(1:np) = Pv;
end

% -------------------------------------------------------------------------
function [assign_r, assign_d] = assign_targets_pairs(x0_all, phi_pair, ...
                                    psi_all, chi_all, region_targets, ...
                                    region_bounds, N, nv)
% Bassins quadratiques (Theoreme 4), avec phi indexe par paire (r,j).
    assign_r = find_region_vec(x0_all, region_bounds);
    assign_d = cell(N, 1);

    for r = 1:max(assign_r)
        idx_r = find(assign_r == r);
        if isempty(idx_r)
            continue;
        end
        x_r    = x0_all(:, idx_r);
        n_dest = length(region_targets{r});
        costs  = zeros(n_dest, length(idx_r));
        for j = 1:n_dest
            phi0j = squeeze(phi_pair{r}{j}(1,:,:));
            psi0  = squeeze(psi_all(r,j,1,:));
            chi0  = chi_all(r,j,1);
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
function trajs = simulate_agents_pairs(x0_all, assign_r, assign_d, phi_pair, ...
                     psi_all, A, S, t_grid, region_targets, n_regions, nv, N_time)
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
            psi_traj = squeeze(psi_all(r,j,:,:));
            phi_j    = phi_pair{r}{j};

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
function [Xhat_traj, Xhat_dot] = compute_cohort_traj(phi_pair, psi_all, ...
                     Xhat0, A, S, t_grid, region_targets, n_regions, nv, N_time)
% Integre la dynamique en boucle fermee Xhat_dot = (A-S*phi)*Xhat - S*psi
% (une seule trajectoire deterministe par paire, partant de Xhat0(r,j,:)).
% C'est exactement la trajectoire moyenne de cohorte Xhat_{r,dj}(t) utilisee
% dans le signal de congestion (eq. 33, 37).
    Tv = t_grid(end);
    Xhat_traj = cell(n_regions, 1);
    Xhat_dot  = cell(n_regions, 1);
    for r = 1:n_regions
        Xhat_traj{r} = cell(length(region_targets{r}), 1);
        Xhat_dot{r}  = cell(length(region_targets{r}), 1);
        for j = 1:length(region_targets{r})
            phi_j    = phi_pair{r}{j};
            psi_traj = squeeze(psi_all(r,j,:,:));
            ASp = zeros(N_time, nv, nv);
            for k = 1:N_time
                ASp(k,:,:) = A - S*squeeze(phi_j(k,:,:));
            end
            ASp_mat = reshape(ASp, N_time, nv*nv);
            x0j = squeeze(Xhat0(r,j,:));
            rhs = @(t, x) cohort_rhs(t, x, ASp_mat, psi_traj, S, t_grid, nv, 1);
            opts = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',Tv/100);
            [~, sol] = ode45(rhs, t_grid, x0j, opts);
            Xhat_traj{r}{j} = sol;                       % N_time x nv
            dsol = zeros(N_time, nv);
            for k = 1:N_time
                dsol(k,:) = (cohort_rhs(t_grid(k), sol(k,:)', ASp_mat, ...
                                         psi_traj, S, t_grid, nv, 1))';
            end
            Xhat_dot{r}{j} = dsol;                       % N_time x nv
        end
    end
end

% -------------------------------------------------------------------------
function Gamma = compute_congestion(Xhat_traj, Xhat_dot, P, region_targets, ...
                     n_regions, eps_kernel, kappa_D, neighbor_frac, N_time)
% Signal de congestion enrichi G_tilde_{r,dj}(t) (eq. 33-38), puis
% Gamma_{r,dj} = sup_t G_tilde_{r,dj}(t) (eq. 39).
%
% Kernel mollifie gaussien (Remarque 3) : Kε(v) = 1/(2*pi*eps^2)*exp(-|v|^2/(2*eps^2)).
    max_d  = max(cellfun(@length, region_targets));
    Gamma  = nan(n_regions, max_d);
    Kmax   = 1/(2*pi*eps_kernel^2);

    % Aplatir la liste des paires pour un acces facile
    pair_r = []; pair_j = [];
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            pair_r(end+1) = r; %#ok<AGROW>
            pair_j(end+1) = j; %#ok<AGROW>
        end
    end
    h = length(pair_r);

    Gtilde = zeros(n_regions, max_d, N_time);
    Dstore = zeros(n_regions, max_d, N_time);

    for k = 1:N_time
        % positions et vitesses de toutes les paires a l'instant k
        pos = zeros(h,2); vel = zeros(h,2); mass = zeros(h,1);
        for p = 1:h
            pos(p,:)  = Xhat_traj{pair_r(p)}{pair_j(p)}(k,:);
            vel(p,:)  = Xhat_dot{pair_r(p)}{pair_j(p)}(k,:);
            mass(p)   = P(pair_r(p), pair_j(p));
        end
        for p = 1:h
            diffs = pos - repmat(pos(p,:), h, 1);
            d2    = sum(diffs.^2, 2);
            Kvals = (1/(2*pi*eps_kernel^2)) * exp(-d2/(2*eps_kernel^2));

            Ghat = sum(mass .* Kvals);

            nb_mask = Kvals > neighbor_frac*Kmax;   % voisinage N(r,dj), eq. (37)
            num_vec = sum(repmat(mass(nb_mask),1,2) .* vel(nb_mask,:), 1);
            denom   = sum(mass(nb_mask) .* sqrt(sum(vel(nb_mask,:).^2,2)));
            Dval    = 1 - norm(num_vec) / max(denom, 1e-8);
            Dstore(pair_r(p), pair_j(p), k) = Dval;
            Gtilde(pair_r(p), pair_j(p), k) = Ghat * (1 + kappa_D*Dval);
        end
    end

    for r = 1:n_regions
        for j = 1:length(region_targets{r})
           k_end = round(0.95*N_time);   % exclut les 5% derniers % de l'horizon (artefact vitesse -> 0)
            [Gamma(r,j), kstar] = max(squeeze(Gtilde(r,j,1:k_end)));
            Dpeak = Dstore(r,j,kstar);
            frac_horizon = kstar / N_time;
                        fprintf('  [diag] region %d -> cible %d : Gamma=%.4f atteint a t=%.0f%% de l''horizon, D=%.3f\n', ...
                r, j, Gamma(r,j), 100*frac_horizon, Dpeak);
        end
    end
end

% -------------------------------------------------------------------------
function mode_idx_new = update_modes_hysteresis(Gamma, mode_idx, up_thresh, ...
                                  down_thresh, L_modes, region_targets, n_regions)
% Regle d'hysteresis (Remarque 6) : on ne monte d'un niveau que si Gamma
% depasse le seuil haut correspondant, on ne redescend que si Gamma est
% repasse sous le seuil bas correspondant. Un seul changement de niveau
% par iteration exterieure (montee ou descente), pour un comportement
% conservateur et non oscillant.
    mode_idx_new = mode_idx;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            m = mode_idx(r,j);
            g = Gamma(r,j);
            if m < L_modes && g > up_thresh(m)
                mode_idx_new(r,j) = m + 1;
            elseif m > 1 && g < down_thresh(m-1)
                mode_idx_new(r,j) = m - 1;
            end
        end
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
function r_all = find_region_vec(x0_all, region_bounds)
% Partition en 4 triangles (diamant) se rejoignant a l'origine, coherente
% avec region_targets = {d1,d2},{d1,d3},{d2,d4},{d3,d4} : chaque triangle
% touche exactement les deux cibles-coins de son cote.
    x = x0_all(1,:); y = x0_all(2,:);
    N = size(x0_all, 2);
    r_all = ones(N, 1);
    r_all(x' <= -abs(y')) = 1;   % Gauche  -> {d1,d2}
    r_all(y' <= -abs(x')) = 2;   % Bas     -> {d1,d3}
    r_all(y' >=  abs(x')) = 3;   % Haut    -> {d2,d4}
    r_all(x' >=  abs(y')) = 4;   % Droite  -> {d3,d4}
end

% -------------------------------------------------------------------------
function bmap = compute_basin_map(gpts, gr, phi_pair, psi_all, chi_all, ...
                                   region_targets, target_names, n_regions)
    bmap = zeros(size(gpts,2), 1);
    for i = 1:size(gpts,2)
        r = gr(i);
        bc = inf; bi = 1;
        x0i = gpts(:,i);
        for j = 1:length(region_targets{r})
            phi0 = squeeze(phi_pair{r}{j}(1,:,:));
            psi0 = squeeze(psi_all(r,j,1,:));
            chi0 = chi_all(r,j,1);
            c = 0.5*(x0i'*phi0*x0i) + psi0'*x0i + chi0;
            if c < bc
                bc = c;
                bi = find(strcmp(target_names, region_targets{r}{j}));
            end
        end
        bmap(i) = bi;
    end
end

% -------------------------------------------------------------------------
function totals = target_totals(P, region_targets, target_names, n_regions)
    totals = zeros(1,4);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            di = find(strcmp(target_names, region_targets{r}{j}));
            totals(di) = totals(di) + P(r,j);
        end
    end
end

% -------------------------------------------------------------------------
function print_split(P, region_targets, target_names, n_regions, label)
    totals = target_totals(P, region_targets, target_names, n_regions);
    fprintf('  [%s] repartition : d1=%.1f%%  d2=%.1f%%  d3=%.1f%%  d4=%.1f%%\n', ...
        label, totals(1)*100, totals(2)*100, totals(3)*100, totals(4)*100);
end

function print_gamma(Gamma, region_targets, n_regions)
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            fprintf('    L%d -> %s : Gamma=%.4f\n', r, region_targets{r}{j}, Gamma(r,j));
        end
    end
end

function print_modes(mode_idx, region_targets, n_regions, mu_levels)
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            m = mode_idx(r,j);
            fprintf('    L%d -> %s : niveau %d (mu=%.2f)\n', r, region_targets{r}{j}, m, mu_levels(m));
        end
    end
end

function h = hist_safe(v)
    v = v(:);

    if isempty(v)
        h = 1;
        return;
    end

    positive_values = v(v > 0);

    if isempty(positive_values)
        v(:) = eps;
    else
        v(v <= 0) = 0.5 * min(positive_values);
    end

    h = v;
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
