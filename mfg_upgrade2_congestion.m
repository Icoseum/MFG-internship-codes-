% =========================================================================
% MFG UPGRADE II - Congestion reactive en ligne (Section 3.2.5)
% D'apres Toumi, Malhame & Le Ny (Automatica 160, 2024) + extensions du
% rapport
%
% Implemente precisement la Section 3.2.5 :
%   - Modes de trafic discretises (eq. 51-52) : un nombre fini de niveaux
%     de congestion ell = 1..L, chacun associe a un coefficient constant
%     mu_ell, ce qui ramene, pour un niveau fige, exactement au modele
%     heterogene de la Section 2.1 (Proposition 2).
%   - Bibliotheque hors-ligne B (eq. 53) : les h x L equations de Riccati
%     a coefficients constants phi^(ell)_{r,dj} sont TOUTES precalculees
%     une seule fois, avant le debut de l'iteration de point fixe.
%   - Boucle INTERIEURE reactive (eq. 54-55) : a chaque iteration k du
%     point fixe (et non plus a chaque etape EXTERIEURE comme l'Upgrade I),
%       Xhat^(k) -> G_tilde^(k) -> sigma^(k) -> phi^(k)(.) -> Xhat^(k+1)
%     C'est la difference structurelle cle avec l'Upgrade I : ici on ne
%     re-resout JAMAIS le modele heterogene jusqu'a convergence complete
%     a niveau fige ; le niveau sigma(r,dj) est re-choisi, dans la
%     bibliotheque B, a CHAQUE iteration du point fixe lui-meme.
%
% Consequence theorique majeure (Lemme 14, Remarque 9, Proposition 5) :
%   la carte de selection de niveau sigma(.) est DISCONTINUE aux points de
%   franchissement de seuil, donc l'argument de Brouwer de la Section 3.1
%   ne s'applique pas directement a la carte composite k -> k+1. Un point
%   fixe conjoint (P*, Xhat*, sigma*) n'est donc PAS garanti exister ; s'il
%   est atteint numeriquement (sigma stabilise ET P, Xhat stabilises),
%   c'est alors, et alors seulement, un veritable equilibre de champ moyen
%   sous sigma* (Proposition 5), avec la garantie epsilon_N-Nash usuelle
%   (Theoreme 6) sur les coefficients corriges lambda_j*mu_{sigma*(r,dj)}.
%   Ce script rapporte explicitement si cette convergence conjointe a ete
%   atteinte ou non (cf. le journal de mise au point : un cycle-limite de
%   periode 2 a deja ete observe empiriquement sur l'Upgrade I et reste un
%   risque, a fortiori ici, pour l'Upgrade II - Remarque 9).
%
% Une regle d'hysteresis (Remarque 8, dans l'esprit de la Remarque 6 de
% l'Upgrade I) attenue le battement d'un niveau a l'autre : la montee suit
% directement l'eq. (49) (peut sauter plusieurs niveaux d'un coup si G_tilde
% est tres grand), la descente est amortie a un niveau a la fois, et
% n'intervient qu'en-dessous d'un seuil bas distinct du seuil de montee.
%
% Difference structurelle vs le modele heterogene "pur" (Section 3.1) et
% vs l'Upgrade I : phi, psi, chi restent indexes par la PAIRE (region,
% cible) et non par la seule cible (Remarque 11), puisque deux cohortes
% visant la meme cible depuis des regions differentes peuvent se voir
% attribuer des niveaux de trafic differents.
%
% UTILISATION : modifier la section PARAMETRES puis Run.
% Compatible MATLAB / GNU Octave. Aucune toolbox requise.
% =========================================================================
clear; close all; clc;

MAKE_PLOTS = true;   % mettre a false pour ne calculer que le point fixe
                      % (utile pour tester la logique sans generer de figures)

% =========================================================================
%  PARAMETRES - MODELE DE BASE (Section 1 / 2.1)
% =========================================================================

A = zeros(2); B = eye(2);
Rx = 1.0*eye(2); Rd = eye(2); Ru = 10.0*eye(2); M = 100*eye(2); T = 4.0;
lambda_target = struct('d1',1.0,'d2',2.0,'d3',20.0,'d4',1.3);   % identique dans I et II

N_agents = 1000;
N_MC     = 1200;
tol_P    = 0.02;   % tolerance sur la masse P (convergence du point fixe)
tol_X    = 0.5;    % tolerance sur les moyennes de cohorte Xhat0
N_time   = 200;

% =========================================================================
%  PARAMETRES - CONGESTION (Section 2.2.2) ET MODES DE TRAFIC (Section 2.2.5)
% =========================================================================
eps_kernel    = 1.2;   % bande passante du noyau mollifie K_eps (eq. 33, Remarque 3)
kappa_D       = 1.0;   % poids de l'indice de desorganisation directionnelle (eq. 38)
neighbor_frac = 0.05;  % seuil relatif definissant le voisinage N(r,dj) (eq. 37)

% Modes de trafic (eq. 46), regime "panique" (croissant, Section 2.2.5) :
% mu_1 = 1 -> niveau "libre" (aucune correction, coincide avec le modele
% heterogene congestion-blind de la Section 2.1). Les niveaux superieurs
% penalisent de plus en plus fortement la cible pour une paire congestionnee.
mu_levels = [1.0, 2.0, 4.0, 8.0, 14.0, 24.0];
L_modes   = length(mu_levels);

% Seuils de bin (eq. 49) : sigma(r,dj) = max{ell : Gamma(r,dj) >= g_up(ell-1)}.
% g_up(1..L-1) delimitent les L niveaux ; g_down < g_up ajoute l'hysteresis
% de la Remarque 8 pour la DESCENTE uniquement (la montee suit l'eq. 49 telle
% quelle). Ordres de grandeur calibres empiriquement (cf. le journal de mise
% au point de l'Upgrade I, Section 3.1) ; a recalibrer via la Remarque 5 du
% rapport (ancrage physique de Feliciani & Nishinari, Table 3) si besoin.
g_up   = [0.015, 0.030, 0.050, 0.070, 0.090];
g_down = 0.80 * g_up;

alpha_relax = 0.5;   % relaxation numerique sur (P,Xhat0) - PAS partie de
                      % l'eq. (49)-(50) elle-meme (voir en-tete + Remarque 9) ;
                      % pur dispositif de stabilisation du point fixe reactif.
K_max = 60;           % budget d'iterations du point fixe reactif

% =========================================================================
%  CONFIGURATION GEOMETRIQUE (identique au modele heterogene / Upgrade I)
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

fprintf('=== MFG Upgrade II : congestion reactive en ligne ===\n');
fprintf('    (Section 2.2.5 du rapport)\n\n');

% =========================================================================
%  ASSOMPTION 6 : horizon uniforme sur TOUS les niveaux de trafic
% =========================================================================
check_horizon_levels(lambda_target, mu_levels, Rd, Rx, S, M, T, ...
    region_targets, n_regions);

% =========================================================================
%  BIBLIOTHEQUE HORS-LIGNE B (eq. 48) : h x L equations de Riccati
%  a coefficients CONSTANTS, resolues UNE SEULE FOIS avant l'iteration.
% =========================================================================
fprintf('\n--- Construction de la bibliotheque hors-ligne B (eq. 48) ---\n');
phi_lib = build_phi_library(lambda_target, mu_levels, Rd, Rx, A, S, M, ...
    t_grid, N_time, n, region_targets, n_regions);

% =========================================================================
%  ETAPE 0 (reference) : equilibre congestion-blind (niveau 1 partout),
%  RESOLU JUSQU'A CONVERGENCE COMPLETE (Theoreme 5). Sert uniquement de
%  point de comparaison "sans congestion" pour les figures finales ; ce
%  n'est PAS une etape du schema reactif lui-meme.
% =========================================================================
fprintf('\n--- Reference : equilibre congestion-blind (niveau 1, Theoreme 5) ---\n');
lambda_eff_free = build_lambda_eff(lambda_target, ones(n_regions,max_d), ...
    mu_levels, region_targets);
phi_pair_free = phi_for_mode(ones(n_regions,max_d), phi_lib, region_targets, n_regions);

rng(42);
samples_MC = sample_initial(N_MC);

[P_base, Xhat0_base, psi_base, chi_base, ~, hist0] = run_fixed_point_full( ...
    phi_pair_free, lambda_eff_free, samples_MC, targets, region_targets, ...
    region_bounds, A, S, Rx, Rd, M, t_grid, tol_P, tol_X, 80, n_regions, n, N_time);

print_split(P_base, region_targets, target_names, n_regions, 'Reference (congestion-blind)');

% =========================================================================
%  INITIALISATION DU SCHEMA REACTIF (eq. 49-50)
%  Niveau initial : mode 1 (libre) partout ; etat initial (P,Xhat0) tire
%  par Monte-Carlo, comme pour le modele de base.
% =========================================================================
fprintf('\n--- Initialisation du point fixe reactif (Upgrade II) ---\n');
mode_idx = ones(n_regions, max_d);
mode_idx = mask_unused(mode_idx, region_targets);

rng(42);
[P_cur, Xhat0_cur] = init_state_from_samples(samples_MC, region_targets, ...
    region_bounds, n_regions, max_d, n);

[psi_cur, chi_cur, phi_pair_cur, Xhat_traj_cur, Xhat_dot_cur] = ...
    solve_value_and_traj(mode_idx, phi_lib, mu_levels, lambda_target, Rd, ...
    Rx, A, S, M, targets, region_targets, n_regions, n, t_grid, N_time, ...
    P_cur, Xhat0_cur);

Gamma_cur = compute_congestion(Xhat_traj_cur, Xhat_dot_cur, P_cur, ...
    region_targets, n_regions, eps_kernel, kappa_D, neighbor_frac, N_time);

fprintf('  Congestion anticipee initiale (niveau 1 partout) :\n');
print_gamma(Gamma_cur, region_targets, n_regions);

% Historiques pour les figures
mode_hist  = {mode_idx};
Gamma_hist = {Gamma_cur};
P_hist     = {P_cur};
errP_hist  = [];
errX_hist  = [];

% =========================================================================
%  BOUCLE DE POINT FIXE REACTIVE (eq. 49-50) :
%     Xhat^(k) -> G_tilde^(k) -> sigma^(k) -> phi^(k)(.) -> Xhat^(k+1)
%  puis nouvelles masses/moyennes via les bassins (Theoreme 4) sous sigma^(k).
% =========================================================================
fprintf('\n--- Point fixe reactif (Upgrade II, Section 2.2.5) ---\n');

converged_joint = false;
for k = 1:K_max

    % --- sigma^(k) : selection de niveau (eq. 49) avec hysteresis (Remarque 8) ---
    mode_new = select_modes_reactive(Gamma_cur, mode_idx, g_up, g_down, ...
        L_modes, region_targets, n_regions);

    % --- phi^(k)(.) : simple lecture dans la bibliotheque B (aucun ODE ici) ---
    [psi_new, chi_new, phi_pair_new, Xhat_traj_new, Xhat_dot_new] = ...
        solve_value_and_traj(mode_new, phi_lib, mu_levels, lambda_target, ...
        Rd, Rx, A, S, M, targets, region_targets, n_regions, n, t_grid, ...
        N_time, P_cur, Xhat0_cur);   % Y1 sous sigma^(k) -> Xhat^(k+1)

    % --- nouvelles masses/moyennes via les bassins quadratiques (Theoreme 4) ---
    [ar_new, ad_new] = assign_targets_pairs(samples_MC, phi_pair_new, ...
        psi_new, chi_new, region_targets, region_bounds, N_MC, n);   % Y2 sous sigma^(k)
    [P_new, Xhat0_new] = state_from_assignment(samples_MC, ar_new, ad_new, ...
        region_targets, n_regions, max_d, n);

    err_P    = max(max(abs(P_new - P_cur)));
    err_X    = max(max(max(abs(Xhat0_new - Xhat0_cur))));
    mode_ok  = isequal(mode_new, mode_idx);
    errP_hist(end+1) = err_P; %#ok<SAGROW>
    errX_hist(end+1) = err_X; %#ok<SAGROW>

    fprintf('\n  [k=%2d] err_P=%.4f | err_X=%.4f | sigma stable : %d\n', ...
        k, err_P, err_X, mode_ok);
    print_modes(mode_new, region_targets, n_regions, mu_levels);
    print_gamma(Gamma_cur, region_targets, n_regions);

    mode_hist{end+1}  = mode_new;  %#ok<SAGROW>
    Gamma_hist{end+1} = Gamma_cur; %#ok<SAGROW>
    P_hist{end+1}     = P_new;     %#ok<SAGROW>

    if err_P < tol_P && err_X < tol_X && mode_ok
        fprintf(['\n  Convergence CONJOINTE atteinte : sigma auto-coherent, ', ...
            'P et Xhat stabilises (Proposition 5).\n']);
        P_cur = P_new; Xhat0_cur = Xhat0_new; mode_idx = mode_new;
        psi_cur = psi_new; chi_cur = chi_new; phi_pair_cur = phi_pair_new;
        Xhat_traj_cur = Xhat_traj_new; %#ok<NASGU>
        converged_joint = true;
        break;
    end

    % --- Relaxation numerique sur (P,Xhat0) : PAS partie de l'eq. (49)-(50)
    % elle-meme (alpha_relax=1 redonne l'iteration brute), simple dispositif
    % de stabilisation (voir en-tete). sigma, lui, n'est jamais amorti : c'est
    % un choix discret, "geler" un niveau intermediaire n'aurait pas de sens
    % dans la bibliotheque B.
    %
    % Le signal de congestion Gamma^(k+1) utilise pour choisir sigma^(k+1) est
    % evalue sur la trajectoire Xhat_traj_new/Xhat_dot_new deja calculee
    % ci-dessus sous sigma^(k) (sans re-resoudre les ODE psi/chi une seconde
    % fois) : cela evite de doubler le cout de chaque iteration (8 ODE psi/chi
    % + 8 integrations de cohorte, deja payees une fois par iteration, comme
    % pour l'Upgrade I) au prix d'un leger decalage, lorsque alpha_relax<1,
    % entre l'etat reellement propage (amorti) et celui ayant produit le
    % signal de diagnostic. Ce decalage disparait des que l'algorithme
    % approche la convergence (P_new ~ P_cur, Xhat0_new ~ Xhat0_cur).
    Gamma_cur = compute_congestion(Xhat_traj_new, Xhat_dot_new, P_new, ...
        region_targets, n_regions, eps_kernel, kappa_D, neighbor_frac, N_time);

    P_cur        = alpha_relax*P_new + (1-alpha_relax)*P_cur;
    Xhat0_cur    = alpha_relax*Xhat0_new + (1-alpha_relax)*Xhat0_cur;
    mode_idx     = mode_new;
    psi_cur      = psi_new; chi_cur = chi_new; phi_pair_cur = phi_pair_new; %#ok<NASGU>
end

if not(converged_joint)
    fprintf(['\n  Budget K_max=%d atteint SANS convergence conjointe confirmee.\n', ...
        '  Ceci est un risque theorique explicite de l''Upgrade II (Remarque 9) :\n', ...
        '  rien ne garantit que la carte composite sigma -> equilibre -> sigma''\n', ...
        '  possede un point fixe (Lemme 14 : sigma est discontinue aux seuils).\n', ...
        '  Inspecter mode_hist / Gamma_hist ci-dessous pour un eventuel cycle-limite\n', ...
        '  (deja observe empiriquement sur le schema analogue de l''Upgrade I).\n'], K_max);
end

print_split(P_cur, region_targets, target_names, n_regions, 'Upgrade II (final)');
fprintf('  Niveaux de trafic finaux (sigma) :\n');
print_modes(mode_idx, region_targets, n_regions, mu_levels);

P_final        = P_cur;
Xhat0_final    = Xhat0_cur;
psi_final      = psi_cur;
chi_final      = chi_cur;
phi_pair_final = phi_pair_cur;
mode_final     = mode_idx;

% =========================================================================
%  SIMULATION D'AGENTS : reference (congestion-blind) vs Upgrade II
% =========================================================================
fprintf('\n--- Simulation %d agents (reference vs Upgrade II)...\n', N_agents);
rng(0);
x0_sim = sample_initial(N_agents);

[ar_base, ad_base] = assign_targets_pairs(x0_sim, phi_pair_free, psi_base, ...
    chi_base, region_targets, region_bounds, N_agents, n);
trajs_base = simulate_agents_pairs(x0_sim, ar_base, ad_base, phi_pair_free, ...
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

    % --- Figure 1 : convergence du point fixe reactif ---
    figure('Name','Convergence du point fixe reactif','NumberTitle','off', ...
        'Position',[50 50 900 400]);
    subplot(1,2,1);
    semilogy(hist_safe(errP_hist),'b-o','LineWidth',1.2,'MarkerSize',3); hold on;
    plot([1 max(length(errP_hist),1)],[tol_P tol_P],'r--','LineWidth',1.2);
    xlabel('Iteration k du point fixe'); ylabel('||P^{(k+1)}-P^{(k)}||');
    title('Convergence de P (eq. 49-50)'); grid on;
    subplot(1,2,2);
    semilogy(hist_safe(errX_hist),'-s','Color',[0.8 0.4 0], ...
        'LineWidth',1.2,'MarkerSize',3); hold on;
    plot([1 max(length(errX_hist),1)],[tol_X tol_X],'r--','LineWidth',1.2);
    xlabel('Iteration k du point fixe'); ylabel('||Xhat0^{(k+1)}-Xhat0^{(k)}||');
    title('Convergence de Xhat(0) (eq. 49-50)'); grid on;

    % --- Figure 2 : evolution de Gamma et du niveau de trafic sigma par paire ---
    figure('Name','Congestion reactive et niveaux de trafic','NumberTitle','off','Position',[80 80 1050 520]);   % un peu plus haut pour la légende en bas
    K = length(Gamma_hist);
    pair_labels = {};
    subplot(1,2,1); hold on;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            gline = zeros(K,1);
            for k = 1:K, gline(k) = Gamma_hist{k}(r,j); end
            plot(0:K-1, gline, '-o', 'LineWidth', 1.3, 'MarkerSize', 4);
            pair_labels{end+1} = sprintf('L%d-%s', r, region_targets{r}{j}); %#ok<SAGROW>
        end
    end
    for ell = 1:L_modes-1
        yline_val = g_up(ell);
        plot([0 K-1], [yline_val yline_val], 'k:', 'LineWidth', 0.8);
    end
    lg = legend(pair_labels, 'Location','southoutside', 'NumColumns', 4,'FontSize',8);
    xlabel('Iteration k'); ylabel('\Gamma_{r,d_j} = sup_t G\_tilde (eq. 39)');
    title('Congestion anticipee, seuils g\_up en pointilles'); grid on;

    subplot(1,2,2); hold on;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            mline = zeros(K,1);
            for k = 1:K, mline(k) = mode_hist{k}(r,j); end
            stairs(0:K-1, mline, '-', 'LineWidth', 1.5);
        end
    end
    ylim([0.5, L_modes+0.5]); yticks(1:L_modes);
    xlabel('Iteration k'); ylabel('Niveau de trafic \sigma(r,d_j) (eq. 49)');
    title('Assignation de niveau par paire (r,d_j)'); grid on;
    ha2 = axes('Position',[0 0 1 1],'Visible','off');
    text(0.5,0.98,'Boucle reactive interieure de l''Upgrade II (eq. 49-50)','Parent',ha2, ...
        'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    % --- Figure 3 : trajectoires reference vs Upgrade II ---
    figure('Name','Trajectoires : reference vs Upgrade II','NumberTitle','off', ...
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
    title('Reference : congestion-blind (niveau 1)', 'FontSize', 12);
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
    title('Upgrade II : reactif (Section 2.2.5)', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;
    ha3 = axes('Position',[0 0 1 1],'Visible','off');
    text(0.5,0.98,'Effet de la correction reactive de congestion sur les trajectoires', ...
        'Parent',ha3,'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    % --- Figure 4 : bassins quadratiques reference vs Upgrade II ---
    figure('Name','Bassins : reference vs Upgrade II','NumberTitle','off', ...
        'Position',[150 100 1300 650]);
    [xx, yy] = meshgrid(linspace(-9.9,9.9,70), linspace(-9.9,9.9,70));
    gpts     = [xx(:) yy(:)]';
    gr       = find_region_vec(gpts, region_bounds);
    cmap4    = [0.3 0.5 0.8; 0.2 0.7 0.4; 0.9 0.5 0.2; 0.7 0.3 0.7];

    subplot(1,2,1); hold on;
    bg_base = compute_basin_map(gpts, gr, phi_pair_free, psi_base, chi_base, ...
        region_targets, target_names, n_regions);
    imagesc([-9.9 9.9], [-9.9 9.9], reshape(bg_base, size(xx)));
    colormap(cmap4); clim([0.5 4.5]);
    h_img = findobj(gca, 'Type', 'image', '-depth', 1);
    if ~isempty(h_img), h_img(1).AlphaData = 0.45 * ones(size(xx)); end
    contour(xx, yy, reshape(bg_base, size(xx)), [1.5 2.5 3.5], 'k--', 'LineWidth', 1.2);
    draw_targets(targets, target_names);
    title('Bassins - reference (congestion-blind)', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;

    subplot(1,2,2); hold on;
    bg_fin = compute_basin_map(gpts, gr, phi_pair_final, psi_final, chi_final, ...
        region_targets, target_names, n_regions);
    imagesc([-9.9 9.9], [-9.9 9.9], reshape(bg_fin, size(xx)));
    colormap(cmap4); clim([0.5 4.5]);
    h_img = findobj(gca, 'Type', 'image', '-depth', 1);
    if ~isempty(h_img), h_img(1).AlphaData = 0.45 * ones(size(xx)); end
    contour(xx, yy, reshape(bg_fin, size(xx)), [1.5 2.5 3.5], 'k--', 'LineWidth', 1.2);
    draw_targets(targets, target_names);
    title('Bassins - Upgrade II (reactif)', 'FontSize', 12);
    xlabel('X (m)'); ylabel('Y (m)'); xlim([-11 11]); ylim([-11 11]); axis square; grid on;
    ha4 = axes('Position',[0 0 1 1],'Visible','off');
    text(0.5,0.98,'Bassins sous le dernier niveau sigma attribue a chaque paire', ...
        'Parent',ha4,'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontSize',12,'FontWeight','bold');

    % --- Figure 5 : repartition des cibles au fil des iterations reactives ---
    figure('Name','Repartition des cibles','NumberTitle','off','Position',[200 150 700 450]);
    K = length(P_hist);
    totals_mat = zeros(K, 4);
    for k = 1:K
        totals_mat(k,:) = target_totals(P_hist{k}, region_targets, target_names, n_regions);
    end
    bar(0:K-1, totals_mat*100, 'stacked');
    legend(target_names, 'Location','eastoutside');
    xlabel('Iteration k du point fixe reactif'); ylabel('Part de la population (%)');
    title('Redistribution de la masse au fil du schema reactif (Upgrade II)', 'FontSize', 11);
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

% -------------------------------------------------------------------------
function check_horizon_levels(lambda_target, mu_levels, Rd, Rx, S, M, T, ...
                               region_targets, n_regions)
% Assomption 6 du rapport : T < Tesc(modes) = min sur TOUTES les paires
% (r,dj) ET TOUS les niveaux ell=1..L des temps d'echappement T1_esc^(j,ell).
% Contrairement a l'Upgrade I (qui ne verifie que le lambda_eff courant a
% chaque etape exterieure), l'Upgrade II doit garantir la bonne definition
% de la bibliotheque ENTIERE des le depart, puisque n'importe quelle paire
% peut se voir attribuer n'importe quel niveau a n'importe quelle iteration.
    Tesc = inf;
    L = length(mu_levels);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            dname = region_targets{r}{j};
            lam_j = lambda_target.(dname);
            for ell = 1:L
                Qj = lam_j*mu_levels(ell)*Rd - Rx;
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
    end
    if T >= Tesc
        warning(['T >= T_esc(modes) = %.3f (Assomption 6) : au moins une paire ', ...
            '(r,dj) viole l''horizon a un des niveaux mu. Reduire T ou mu_levels.'], Tesc);
    else
        fprintf('  Assomption 6 verifiee : T=%.2f < T_esc(modes)=%.3f (sur %d niveaux).\n', ...
            T, Tesc, L);
    end
end

% -------------------------------------------------------------------------
function phi_lib = build_phi_library(lambda_target, mu_levels, Rd, Rx, A, ...
                                      S, M, t_grid, N_time, nv, region_targets, n_regions)
% Bibliotheque hors-ligne B (eq. 48) : phi_lib{r}{j}{ell} = solution de
% l'equation de Riccati (eq. 11) a coefficients constants pour la paire
% (r,dj) au niveau de trafic ell, Q = lambda_j*mu_ell*Rd - Rx.
% Cout total : h x L resolutions, fait UNE SEULE FOIS avant l'iteration.
    L = length(mu_levels);
    phi_lib = cell(n_regions,1);
    h = 0;
    for r = 1:n_regions
        phi_lib{r} = cell(length(region_targets{r}),1);
        for j = 1:length(region_targets{r})
            dname = region_targets{r}{j};
            lam_j = lambda_target.(dname);
            phi_lib{r}{j} = cell(L,1);
            for ell = 1:L
                Qj = lam_j*mu_levels(ell)*Rd - Rx;
                phi_lib{r}{j}{ell} = solve_riccati(A, S, Qj, M, t_grid, N_time, nv);
            end
            h = h + 1;
        end
    end
    fprintf('  Bibliotheque B construite : %d paires (region,cible) x %d niveaux = %d Riccati.\n', ...
        h, L, h*L);
end

function phi_all = solve_riccati(A, S, Q, M, t_grid, N_time, nv)
% UN SEUL ode45 (backward via tau = T - t).
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
function phi_pair = phi_for_mode(mode_idx, phi_lib, region_targets, n_regions)
% Extrait, pour l'assignation de niveau mode_idx courante, un phi_pair du
% meme format que dans le modele heterogene / l'Upgrade I (indexe par
% paire (r,j)), en piochant simplement dans la bibliotheque B. Aucun ODE
% n'est resolu ici : c'est exactement le point de l'eq. (48).
    phi_pair = cell(n_regions,1);
    for r = 1:n_regions
        phi_pair{r} = cell(length(region_targets{r}),1);
        for j = 1:length(region_targets{r})
            ell = mode_idx(r,j);
            phi_pair{r}{j} = phi_lib{r}{j}{ell};
        end
    end
end

% -------------------------------------------------------------------------
function lambda_eff = build_lambda_eff(lambda_target, mode_idx, mu_levels, region_targets)
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
function mode_idx_new = select_modes_reactive(Gamma, mode_idx, g_up, g_down, ...
                                  L_modes, region_targets, n_regions)
% Selection de niveau reactive (eq. 49), avec hysteresis a la descente
% (Remarque 8) pour eviter le battement d'un niveau a l'autre :
%   - MONTEE : suit l'eq. (49) telle quelle, sigma_up = max{ell : Gamma >= g_up(ell-1)} ;
%     peut sauter directement plusieurs niveaux si Gamma est tres grand.
%   - DESCENTE : amortie a un niveau a la fois, et seulement si Gamma est
%     repasse sous le seuil bas g_down(niveau_courant - 1) < g_up correspondant.
    mode_idx_new = mode_idx;
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            g = Gamma(r,j);
            m = mode_idx(r,j);

            ell_up = 1;
            for ell = 1:L_modes-1
                if g >= g_up(ell)
                    ell_up = ell + 1;
                end
            end

            if ell_up > m
                mode_idx_new(r,j) = ell_up;         % montee (eq. 49), directe
            elseif m > 1 && g < g_down(m-1)
                mode_idx_new(r,j) = m - 1;           % descente amortie
            end
        end
    end
end

% -------------------------------------------------------------------------
function [psi_all, chi_all, phi_pair, Xhat_traj, Xhat_dot] = solve_value_and_traj( ...
    mode_idx, phi_lib, mu_levels, lambda_target, Rd, Rx, A, S, M, targets, ...
    region_targets, n_regions, nv, t_grid, N_time, P, Xhat0)
% Etant donnee une assignation de niveau sigma = mode_idx et un etat
% courant (P, Xhat0) :
%   1) recupere phi^(sigma)_{r,dj} dans la bibliotheque B (eq. 48) ;
%   2) resout psi, chi (eq. 44-45, Lemme 13) pour ce sigma fige ;
%   3) integre la dynamique en boucle fermee pour produire la trajectoire
%      de cohorte Xhat(t) (le "Y1 sous sigma" de l'eq. 50, cf. Lemme 9).
    max_d = size(mode_idx,2);
    phi_pair    = phi_for_mode(mode_idx, phi_lib, region_targets, n_regions);
    lambda_eff  = build_lambda_eff(lambda_target, mode_idx, mu_levels, region_targets);

    Xr_cell = compute_Xr(P, Xhat0, region_targets, n_regions, nv, N_time);

    psi_all = zeros(n_regions, max_d, N_time, nv);
    chi_all = zeros(n_regions, max_d, N_time);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            dname = region_targets{r}{j};
            Rdj   = lambda_eff(r,j) * Rd;
            [psi_rd, chi_rd] = solve_psi_chi_pair(phi_pair{r}{j}, Xr_cell{r}, ...
                targets.(dname), A, S, Rx, Rdj, M, t_grid, N_time, nv);
            psi_all(r,j,:,:) = psi_rd;
            chi_all(r,j,:)   = chi_rd;
        end
    end

    [Xhat_traj, Xhat_dot] = compute_cohort_traj(phi_pair, psi_all, Xhat0, ...
        A, S, t_grid, region_targets, n_regions, nv, N_time);
end

% -------------------------------------------------------------------------
function [psi_all, chi_all] = solve_psi_chi_pair(phi_pair_rj, Xr_traj, d_vec, ...
                                   A, S, Rx, Rdj, M, t_grid, N_time, nv)
% Eq. (44)-(45) (Lemme 13) : phi et Rdj sont ceux de la paire (r,dj) au
% niveau de trafic courant, ici geles sur un pas du point fixe reactif.
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
function [Xhat_traj, Xhat_dot] = compute_cohort_traj(phi_pair, psi_all, ...
                     Xhat0, A, S, t_grid, region_targets, n_regions, nv, N_time)
% Integre la dynamique en boucle fermee Xhat_dot = (A-S*phi)*Xhat - S*psi
% (Lemme 9) : trajectoire moyenne de cohorte Xhat_{r,dj}(t) utilisee dans
% le signal de congestion (eq. 33, 37) a l'iteration SUIVANTE.
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
function Gamma = compute_congestion(Xhat_traj, Xhat_dot, P, region_targets, ...
                     n_regions, eps_kernel, kappa_D, neighbor_frac, N_time)
% Signal de congestion enrichi G_tilde_{r,dj}(t) (eq. 33-38), puis
% Gamma_{r,dj} = sup_t G_tilde_{r,dj}(t) (eq. 39), en excluant les 5%
% derniers de l'horizon (artefact de decouplage vitesse -> 0, cf. le
% journal de mise au point de l'Upgrade I, Section 2.2).
%
% Noyau mollifie gaussien (Remarque 3) : Kε(v) = 1/(2*pi*eps^2)*exp(-|v|^2/(2*eps^2)).
    max_d  = max(cellfun(@length, region_targets));
    Gamma  = nan(n_regions, max_d);
    Kmax   = 1/(2*pi*eps_kernel^2);

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
            k_end = round(0.95*N_time);   % exclut les 5% derniers % de l'horizon
            [Gamma(r,j), ~] = max(squeeze(Gtilde(r,j,1:k_end)));
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
function [P, Xhat0] = init_state_from_samples(samples_MC, region_targets, ...
                                    region_bounds, n_regions, max_d, nv)
% Tirage initial (P^(0), Xhat0^(0)) : cible initiale uniforme parmi les
% candidates de la region, exactement comme pour le modele de base.
    N = size(samples_MC,2);
    P = zeros(n_regions,max_d);
    Xhat0 = zeros(n_regions,max_d,nv);
    reg_mc = find_region_vec(samples_MC, region_bounds);
    dest_mc = cell(N,1);
    for i = 1:N
        r = reg_mc(i);
        dest_mc{i} = region_targets{r}{randi(length(region_targets{r}))};
    end
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            d_name = region_targets{r}{j};
            idx = find(reg_mc==r & strcmp(dest_mc,d_name));
            P(r,j) = length(idx)/N;
            if not(isempty(idx))
                Xhat0(r,j,:) = mean(samples_MC(:,idx),2)';
            end
        end
    end
end

% -------------------------------------------------------------------------
function [P, Xhat0] = state_from_assignment(samples_MC, ar, ad, ...
                                    region_targets, n_regions, max_d, nv)
% Recalcule (P, Xhat0) a partir d'une assignation region/cible par agent
% (issue des bassins de Theoreme 4 sous un sigma donne).
    N = size(samples_MC,2);
    P = zeros(n_regions,max_d);
    Xhat0 = zeros(n_regions,max_d,nv);
    for r = 1:n_regions
        for j = 1:length(region_targets{r})
            d_name = region_targets{r}{j};
            idx = find(ar==r & strcmp(ad,d_name));
            P(r,j) = length(idx)/N;
            if not(isempty(idx))
                Xhat0(r,j,:) = mean(samples_MC(:,idx),2)';
            end
        end
    end
end

% -------------------------------------------------------------------------
function r_all = find_region_vec(x0_all, region_bounds) %#ok<INUSD>
% Partition en 4 triangles (diamant) se rejoignant a l'origine, coherente
% avec region_targets = {d1,d2},{d1,d3},{d2,d4},{d3,d4} : chaque triangle
% touche exactement les deux cibles-coins de son cote (meme correction
% geometrique que dans le script de l'Upgrade I).
    x = x0_all(1,:); y = x0_all(2,:);
    N = size(x0_all, 2);
    r_all = ones(N, 1);
    r_all(x' <= -abs(y')) = 1;   % Gauche  -> {d1,d2}
    r_all(y' <= -abs(x')) = 2;   % Bas     -> {d1,d3}
    r_all(y' >=  abs(x')) = 3;   % Haut    -> {d2,d4}
    r_all(x' >=  abs(y')) = 4;   % Droite  -> {d3,d4}
end

% -------------------------------------------------------------------------
function [assign_r, assign_d] = assign_targets_pairs(x0_all, phi_pair, ...
                                    psi_all, chi_all, region_targets, ...
                                    region_bounds, N, nv) %#ok<INUSD>
% Bassins quadratiques (Theoreme 4), phi/psi/chi indexes par paire (r,j)
% et geles au niveau sigma courant pour cet appel.
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

% -------------------------------------------------------------------------
function [P_out, Xhat0_out, psi_out, chi_out, Xr_out, hist_err] = ...
    run_fixed_point_full(phi_pair, lambda_eff, samples_MC, targets, ...
        region_targets, region_bounds, A, S, Rx, Rd, M, t_grid, tol_P, ...
        tol_X, max_iter, n_regions, nv, N_time)
% Point fixe (Picard amorti) du modele heterogene a NIVEAU FIGE (Theoreme 5),
% utilise UNIQUEMENT pour calculer la reference congestion-blind (Etape 0)
% servant de comparaison aux figures ; ce n'est pas une brique du schema
% reactif de l'Upgrade II lui-meme (qui, lui, ne resout jamais ce point
% fixe interieur jusqu'a convergence a niveau fige - c'est precisement la
% difference avec l'Upgrade I, cf. en-tete du script).
    N     = size(samples_MC, 2);
    max_d = max(cellfun(@length, region_targets));

    rng(42);
    [P, Xhat0] = init_state_from_samples(samples_MC, region_targets, ...
        region_bounds, n_regions, max_d, nv);

    psi_out = zeros(n_regions, max_d, N_time, nv);
    chi_out = zeros(n_regions, max_d, N_time);
    hist_err = zeros(max_iter, 2);
    n_iter = 0;

    for it = 1:max_iter
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

        [ar_new, ad_new] = assign_targets_pairs(samples_MC, phi_pair, ...
            psi_out, chi_out, region_targets, region_bounds, N, nv);
        [P_new, Xhat0_new] = state_from_assignment(samples_MC, ar_new, ...
            ad_new, region_targets, n_regions, max_d, nv);

        err_P = max(max(abs(P_new - P)));
        err_X = max(max(max(abs(Xhat0_new - Xhat0))));
        n_iter = it;
        hist_err(it,:) = [err_P, err_X];

        if mod(it,10) == 0 || it == 1
            fprintf('    iter %3d | err_P=%.4f | err_X=%.4f\n', it, err_P, err_X);
        end

        if err_P < tol_P && err_X < tol_X
            P = P_new; Xhat0 = Xhat0_new;
            fprintf('    Converge en %d iterations.\n', it);
            break;
        end

        P     = 0.5*P_new + 0.5*P;
        Xhat0 = 0.5*Xhat0_new + 0.5*Xhat0;
    end
    if n_iter == max_iter
        fprintf('    Max iterations atteint.\n');
    end
    hist_err = hist_err(1:n_iter,:);

    P_out = P; Xhat0_out = Xhat0;
    Xr_final = compute_Xr(P_out, Xhat0_out, region_targets, n_regions, nv, N_time);
    Xr_out = struct();
    for r = 1:n_regions
        Xr_out.(sprintf('r%d',r)) = Xr_final{r};
    end
end

% -------------------------------------------------------------------------
function bmap = compute_basin_map(gpts, gr, phi_pair, psi_all, chi_all, ...
                                   region_targets, target_names, n_regions) %#ok<INUSD>
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
            ell = mode_idx(r,j);
            fprintf('    L%d -> %s : niveau %d (mu=%.2f)\n', r, region_targets{r}{j}, ...
                ell, mu_levels(ell));
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
