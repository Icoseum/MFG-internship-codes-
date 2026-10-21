% =========================================================================
% MFG - Section 3.3 du rapport : alternative densitaire variationnelle
% de Benamou, Carlier & Santambrogio
%
% Contexte (rapport, Section 3.3) : contrairement aux Upgrades I et II
% (Sections 3.2.4-3.2.5, deja implementes), qui preservent la structure
% LQ/Riccati en discretisant la congestion en modes de trafic constants
% par morceaux (Proposition 2), ce script ABANDONNE cette structure et
% resout directement le systeme couple
%
%   -d_t phi + |grad phi|^2/2 = g(x, rho)                      (HJ, eq.59)
%    d_t rho - div(rho grad phi) = 0,  rho(0,.) = rho0          (CE, eq.60)
%
% via la reduction variationnelle convexe (eq. 61-62) :
%
%   min_{rho,w}  Ibar(rho,w) = int_0^T int_Omega [ |w|^2/(2 rho) + G(x,rho) ]
%                              + int_Omega Psi rho_T
%   s.c. d_t rho + div(w) = 0,  rho(0,.)=rho0                   (eq. 61/3)
%
% resolu numeriquement par l'algorithme de Lagrangien augmente ALG2
% (Fortin-Glowinski), Section 3.3.7 du rapport / Section 6.1 de BCS.
%
% CHOIX DE MODELISATION (Section 3.3.2 du rapport, route (b)) :
%   On abandonne le choix discret de cible {d_j} region par region : la
%   preference de destination est encodee UNE FOIS pour toutes dans un
%   cout terminal continu Psi : Omega -> R (ici : distance quadratique a
%   la cible la plus proche parmi les 4 cibles du modele de base, pour
%   rester dans la continuite geometrique des scripts Upgrade I/II).
%   Cf. Remarque 13 du rapport : aucun bassin d'attraction (Theoreme 1/4)
%   n'existe plus dans ce cadre.
%
% DEUX REGIMES DE CONGESTION (Section 3.3.5 du rapport) :
%   - "soft"  : G(rho) = rho^m/m  (m>1)      -> g(rho)=rho^(m-1)
%   - "hard"  : G(rho) = 0 sur [0,1], +inf sinon (contrainte de capacite)
%   Les deux sont resolues et comparees (cf. Fig. 2-4 de BCS).
%
% CE QUI EST PERDU (Remarques 13-15 du rapport, rappel) :
%   - pas de choix discret de cible (bassins quadriques/hyperplans perdus)
%   - pas de loi de feedback fermee (pas de phi_r,dj(t) matriciel type
%     Riccati) : tout est resolu numeriquement (ALG2)
%   - pas de garantie epsilon_N-Nash en population finie (Theoreme 6/3
%     de [1] n'a pas d'analogue etabli ici, cf. BCS Section 1)
%
% GARDE-FOU NUMERIQUE : les operateurs discrets temps/espace utilises
% dans l'etape 1 d'ALG2 (resolution "elliptique" en phi) sont verifies
% par un test d'adjonction explicite (produit scalaire) au demarrage du
% script. Si ce test echoue, le script s'arrete avec une erreur claire.
%
% UTILISATION : modifier la section PARAMETRES puis Run.
% Compatible MATLAB (teste pour R2026a). Aucune toolbox requise (pcg est
% une fonction MATLAB de base).
% =========================================================================
clear; close all; clc;

MAKE_PLOTS = true;

fprintf('=== MFG Section 2.3 : cadre variationnel de Benamou-Carlier-Santambrogio ===\n');
fprintf('    (alternative densitaire a la structure LQ/Riccati des Sections 1-2.2)\n\n');

% =========================================================================
%  AUTO-TEST : les operateurs discrets "cdiff" / "cdiff_adjoint" utilises
%  pour l'etape 1 (solve elliptique en phi) doivent former une paire
%  adjointe exacte au sens du produit scalaire discret. On le verifie
%  numeriquement AVANT de faire confiance au reste du script.
% =========================================================================
rng(0);
u_test = randn(6,5,4);
p_test = randn(6,5,4);
for dim_test = 1:3
    lhs = sum(sum(sum( cdiff(u_test,dim_test,1.0) .* p_test )));
    rhs = sum(sum(sum( u_test .* cdiff_adjoint(p_test,dim_test,1.0) )));
    err = abs(lhs-rhs)/max(1,abs(lhs));
    assert(err < 1e-10, sprintf('ERREUR : cdiff/cdiff_adjoint non adjoints (dim=%d, err=%.3e)', dim_test, err));
end
fprintf('[auto-test] operateurs adjoints (temps + 2 directions spatiales) : OK\n\n');

% =========================================================================
%  PARAMETRES - DOMAINE ET GRILLE (identique en echelle aux scripts
%  Upgrade I/II : carre [-10,10]^2, cibles aux 4 coins)
% =========================================================================
Lx = 20; Ly = 20;
nx = 22; ny = 22;                 % resolution spatiale (augmenter si besoin)
h  = Lx/nx;                       % pas d'espace (dx=dy)
xs = linspace(-10+h/2, 10-h/2, nx);
ys = linspace(-10+h/2, 10-h/2, ny);
[XX, YY] = meshgrid(xs, ys);      % taille ny x nx

T   = 4.0;
nt  = 26;                         % nb d'intervalles de temps (augmenter si besoin)
dt  = T/nt;
ntp1 = nt+1;

targets      = struct('d1',[-10;-10],'d2',[-10;10],'d3',[10;-10],'d4',[10;10]);
target_names = {'d1','d2','d3','d4'};

% =========================================================================
%  PARAMETRES - ALG2 (Section 2.3.7)
% =========================================================================
r_alg2      = 1.0;      % parametre d'augmentation (comme BCS Section 6.1)
n_alg2_iter = 250;       % budget d'iterations ALG2 (BCS: ~2000 pour convergence
                         % fine ; qualitativement visible bien plus tot)
pcg_tol     = 1e-6;
pcg_maxit   = 300;
newton_iter = 25;        % Newton (congestion "soft") par point de grille

MODES_TO_RUN = {'soft','hard'};
m_soft = 6;              % exposant de G(rho)=rho^m/m (identique a BCS Fig.3, m=6)

% =========================================================================
%  DONNEES : densite initiale rho0 et cout terminal Psi (route (b),
%  Section 2.3.2 : Psi encode la preference de destination, remplace le
%  choix discret {d_j})
% =========================================================================
rho0 = 0.9*exp(-((XX+8).^2+(YY+8).^2)/(2*2.2^2)) + ...
       0.5*exp(-((XX+8).^2+(YY-8).^2)/(2*1.6^2));
rho0 = rho0 / (sum(rho0(:))*h^2);     % masse totale normalisee a 1

dist2 = inf(size(XX));
for k = 1:4
    d = targets.(target_names{k});
    dist2 = min(dist2, (XX-d(1)).^2 + (YY-d(2)).^2);
end
Psi = 0.03 * dist2;                   % cout terminal (echelle raisonnable vs r=1)

fprintf('Masse initiale (verification) : %.6f\n\n', sum(rho0(:))*h^2);

% =========================================================================
%  BOUCLE SUR LES DEUX REGIMES DE CONGESTION (Section 2.3.5)
% =========================================================================
results = struct();
for im = 1:length(MODES_TO_RUN)
    mode = MODES_TO_RUN{im};
    fprintf('--- ALG2 : congestion "%s" ---\n', mode);
    res = run_alg2(mode, m_soft, rho0, Psi, r_alg2, dt, h, ntp1, ...
                    n_alg2_iter, pcg_tol, pcg_maxit, newton_iter, nx, ny);
    results.(mode) = res;
    fprintf('  masse finale (couche T) : %.6f   |  max(rho) final : %.4f\n\n', ...
        sum(squeeze(res.rho_hist(end,:,:)),'all')*h^2, max(res.rho_hist(:)));
end

% =========================================================================
%  DIAGNOSTIC / VALIDATION (fait echo a la Figure 2 de BCS : decroissance
%  du residu L2 de l'equation HJ, ici mesure comme ||Dt(phi)+a||+||Dx(phi)+b||)
% =========================================================================
if MAKE_PLOTS
    figure('Name','Convergence ALG2 (cf. Fig. 2 de BCS)','Color','w');
    hold on;
    for im = 1:length(MODES_TO_RUN)
        mode = MODES_TO_RUN{im};
        semilogy(1:n_alg2_iter, results.(mode).residual_hist, 'LineWidth',1.3, ...
                 'DisplayName', mode);
    end
    set(gca,'YScale','log');
    xlabel('iterations'); ylabel('residu HJ (log)');
    title('Convergence de l''algorithme ALG2 (Section 2.3.7)');
    legend('show'); grid on; box on;

    % --- Figure 1 (style BCS) : potentiel terminal Psi et densite initiale
    figure('Name','Domaine : Psi et rho0 (cf. Fig. 1 de BCS)','Color','w');
    subplot(1,2,1);
    imagesc(xs, ys, Psi); axis xy equal tight; colorbar;
    title('Cout terminal \Psi(x) (route (b), Remarque 13)');
    xlabel('x_1'); ylabel('x_2');
    subplot(1,2,2);
    imagesc(xs, ys, rho0); axis xy equal tight; colorbar;
    title('Densite initiale \rho_0');
    xlabel('x_1'); ylabel('x_2');

    % --- Snapshots temporels de la densite (cf. Fig. 3/4 de BCS)
    snap_idx = round(linspace(1, ntp1, 6));
    for im = 1:length(MODES_TO_RUN)
        mode = MODES_TO_RUN{im};
        figure('Name', sprintf('Evolution de la densite - congestion "%s"', mode), 'Color','w');
        for s = 1:length(snap_idx)
            subplot(2,3,s);
            k = snap_idx(s);
            imagesc(xs, ys, squeeze(results.(mode).rho_hist(k,:,:)));
            axis xy equal tight;
            caxis([0, max(1.05, max(results.(mode).rho_hist(:)))]);
            colorbar;
            title(sprintf('t = %.2f', (k-1)*dt));
        end
        sgtitle(sprintf('Congestion "%s" (m=%d si "soft") - Section 2.3', mode, m_soft));
    end
end

% =========================================================================
%  RAPPEL DES LIMITES DU MODELE (Section 2.3.6 du rapport)
% =========================================================================
fprintf('--- Rappel (Section 2.3.6 du rapport) ---\n');
fprintf('  * Pas de choix discret de cible : Psi encode la preference une\n');
fprintf('    fois pour toutes (Remarque 13) - aucun bassin d''attraction ici.\n');
fprintf('  * Pas de loi de feedback fermee : phi est resolu numeriquement\n');
fprintf('    (pas de matrice de Riccati) - Remarque 14.\n');
fprintf('  * Pas de garantie epsilon_N-Nash en population finie etablie pour\n');
fprintf('    ce cadre (Remarque 15, cf. aussi BCS Section 1, "delicate issue,\n');
fprintf('    beyond the scope of the present paper").\n');


% =========================================================================
%                          FONCTIONS LOCALES
% =========================================================================

function res = run_alg2(mode, m_soft, rho0, Psi, r, dt, h, ntp1, ...
                         n_iter, pcg_tol, pcg_maxit, newton_iter, nx, ny)
% Boucle ALG2 complete (Section 2.3.7 du rapport / Section 6.1 de BCS)
% pour un regime de congestion donne ('soft' ou 'hard').

    sz = [ntp1, ny, nx];
    N  = prod(sz);

    terminal_mask = zeros(sz); terminal_mask(end,:,:) = 1;
    rho0_pad = zeros(sz); rho0_pad(1,:,:) = reshape(rho0, [1,ny,nx]);

    % --- initialisation des variables primales/duales ---
    phi = zeros(sz);
    a   = zeros(sz);
    b   = zeros([sz,2]);
    rho = repmat(reshape(rho0,[1,ny,nx]), [ntp1,1,1]);   % chaud demarrage
    w   = zeros([sz,2]);
    c   = Psi;
    mu  = zeros(ny,nx);

    residual_hist = zeros(n_iter,1);
    rho_hist_stride = max(1, floor(n_iter/1));  % on ne garde que l'etat final courant a chaque iter (leger)

    Mfun = @(x) Mop_vec(x, sz, r, dt, h);

    for it = 1:n_iter
        % ---------------- STEP 1 : mise a jour de phi (elliptique) ------
        DtT_rho = cdiff_adjoint(rho,1,dt);
        DxT_w   = cdiff_adjoint(w(:,:,:,1),2,h) + cdiff_adjoint(w(:,:,:,2),3,h);
        DtT_a   = cdiff_adjoint(a,1,dt);
        DxT_b   = cdiff_adjoint(b(:,:,:,1),2,h) + cdiff_adjoint(b(:,:,:,2),3,h);
        mu_pad = zeros(sz); mu_pad(end,:,:) = mu;
        c_pad  = zeros(sz); c_pad(end,:,:)  = c;

        RHS = rho0_pad + DtT_rho + DxT_w - mu_pad - r*DtT_a - r*DxT_b + r*c_pad;

        [phi_vec, flag] = pcg(Mfun, RHS(:), pcg_tol, pcg_maxit, [], [], phi(:)); %#ok<ASGLU>
        phi = reshape(phi_vec, sz);

        % ---------------- STEP 2 : mise a jour de (a,b) et de c ----------
        A0 = cdiff(phi,1,dt);
        B0y = cdiff(phi,2,h);
        B0x = cdiff(phi,3,h);

        if strcmp(mode,'hard')
            [a,b] = prox_hard(A0,B0y,B0x,rho,w,r);
        else
            [a,b] = prox_soft(A0,B0y,B0x,rho,w,r,m_soft,newton_iter);
        end

        phiT = squeeze(phi(end,:,:));
        c = min(phiT + mu/r, Psi);

        % ---------------- STEP 3 : mise a jour des variables duales ------
        rho = rho - r*(A0 + a);
        w(:,:,:,1) = w(:,:,:,1) - r*(B0y + b(:,:,:,1));
        w(:,:,:,2) = w(:,:,:,2) - r*(B0x + b(:,:,:,2));
        mu = mu + r*(phiT - c);

        % ---------------- diagnostic de convergence -----------------------
        res_t = A0 + a;
        res_x = B0y + b(:,:,:,1);
        res_y = B0x + b(:,:,:,2);
        residual_hist(it) = log10(sqrt(mean(res_t(:).^2)+mean(res_x(:).^2)+mean(res_y(:).^2)) + 1e-16);

        if mod(it,50)==0 || it==1
            fprintf('  [%s] iter %4d/%4d : log10(residu HJ) = %.3f, masse(T) = %.4f\n', ...
                mode, it, n_iter, residual_hist(it), sum(squeeze(rho(end,:,:)),'all')*h^2);
        end
    end

    res.phi = phi;
    res.rho_hist = rho;               % (ntp1,ny,nx) : "trajectoire" temporelle
    res.residual_hist = residual_hist;
end

% -------------------------------------------------------------------------
function y = Mop_vec(x, sz, r, dt, h)
% Operateur elliptique de l'etape 1 d'ALG2 (forme matrice-vecteur, utilise
% par pcg) :  M(phi) = r*(Dt^T Dt + Dx^T Dx)(phi) + r*(masque terminal).*phi
    phi = reshape(x, sz);
    term = r*( cdiff_adjoint(cdiff(phi,1,dt),1,dt) ...
             + cdiff_adjoint(cdiff(phi,2,h ),2,h ) ...
             + cdiff_adjoint(cdiff(phi,3,h ),3,h ) );
    term(end,:,:) = term(end,:,:) + r*phi(end,:,:);
    y = term(:);
end

% -------------------------------------------------------------------------
function d = cdiff(u, dim, h)
% Difference avant "clamped" (Neumann) : d_i=(u_{i+1}-u_i)/h pour i<n,
% d_n=0. Appliquee le long de la dimension DIM (1=temps,2=y,3=x).
    d = zeros(size(u));
    n = size(u,dim);
    switch dim
        case 1
            d(1:n-1,:,:) = (u(2:n,:,:) - u(1:n-1,:,:))/h;
        case 2
            d(:,1:n-1,:) = (u(:,2:n,:) - u(:,1:n-1,:))/h;
        case 3
            d(:,:,1:n-1) = (u(:,:,2:n) - u(:,:,1:n-1))/h;
    end
end

% -------------------------------------------------------------------------
function d = cdiff_adjoint(p, dim, h)
% Adjoint exact de cdiff (verifie par le test numerique en tete de script):
%   (D^T p)_1 = -p_1/h ,  (D^T p)_i = (p_{i-1}-p_i)/h pour 2<=i<=n-1 ,
%   (D^T p)_n =  p_{n-1}/h.
    d = zeros(size(p));
    n = size(p,dim);
    switch dim
        case 1
            d(1,:,:) = -p(1,:,:)/h;
            if n>=3, d(2:n-1,:,:) = (p(1:n-2,:,:)-p(2:n-1,:,:))/h; end
            d(n,:,:) = p(n-1,:,:)/h;
        case 2
            d(:,1,:) = -p(:,1,:)/h;
            if n>=3, d(:,2:n-1,:) = (p(:,1:n-2,:)-p(:,2:n-1,:))/h; end
            d(:,n,:) = p(:,n-1,:)/h;
        case 3
            d(:,:,1) = -p(:,:,1)/h;
            if n>=3, d(:,:,2:n-1) = (p(:,:,1:n-2)-p(:,:,2:n-1))/h; end
            d(:,:,n) = p(:,:,n-1)/h;
    end
end

% -------------------------------------------------------------------------
function [a,b] = prox_hard(A0,B0y,B0x,rho,w,r)
% Etape proximale ponctuelle (a,b), regime de congestion "dur" (Section
% 2.3.5) : G(rho)=0 sur [0,1], +inf sinon => G*(p)=p^+ (partie positive),
% q(s) := (G*)'(s) = 1{s>0} (fonction en escalier => solution EXPLICITE
% par selection de branche, pas besoin d'iteration de Newton).
    Wy = w(:,:,:,1) - r*B0y;
    Wx = w(:,:,:,2) - r*B0x;

    % branche q=0 (zone non saturee)
    a0 = rho/r - A0;
    b0y = Wy/r; b0x = Wx/r;
    s0 = a0 + 0.5*(b0y.^2+b0x.^2);

    % branche q=1 (zone saturee, rho=1)
    a1 = (rho-1)/r - A0;
    b1y = Wy/(1+r); b1x = Wx/(1+r);
    s1 = a1 + 0.5*(b1y.^2+b1x.^2); %#ok<NASGU>

    % par defaut : branche saturee ; on bascule vers la branche libre
    % partout ou celle-ci est auto-consistante (s0<=0)
    a = a1; by = b1y; bx = b1x;
    mask0 = (s0 <= 0);
    a(mask0)  = a0(mask0);
    by(mask0) = b0y(mask0);
    bx(mask0) = b0x(mask0);

    b = cat(4, by, bx);
end

% -------------------------------------------------------------------------
function [a,b] = prox_soft(A0,B0y,B0x,rho,w,r,m,newton_iter)
% Etape proximale ponctuelle (a,b), regime de congestion "doux" (Section
% 2.3.5) : G(rho)=rho^m/m => G*(p)=(p^+)^{m'}/m', m'=m/(m-1),
% q(s):=(G*)'(s)=(s^+)^{1/(m-1)}. Resolution par Newton scalaire vectorise
% (la fonction residu est monotone, la derivee est toujours <= -1, donc
% le pas de Newton ne peut pas diverger - cf. commentaires du chat).
    p_exp = 1/(m-1);
    Wy = w(:,:,:,1) - r*B0y;
    Wx = w(:,:,:,2) - r*B0x;
    normW2 = Wy.^2 + Wx.^2;

    s = A0;   % initialisation
    for it = 1:newton_iter
        spos = max(s,0);
        q = spos.^p_exp;
        qprime = p_exp * max(spos,eps).^(p_exp-1);
        qprime(s<=0) = 0;
        denom = q + r;
        a_s = (rho - q)/r - A0;
        Rval = a_s + 0.5*normW2./denom.^2 - s;
        dRds = -qprime/r - qprime.*normW2./denom.^3 - 1;   % toujours <= -1
        s = s - Rval./dRds;
    end

    q = max(s,0).^p_exp;
    denom = q + r;
    a = (rho - q)/r - A0;
    by = Wy./denom;
    bx = Wx./denom;
    b = cat(4, by, bx);
end
