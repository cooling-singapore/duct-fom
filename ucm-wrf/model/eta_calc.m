function [eta, heights] = eta_calc(p_top_requested, z_levs)
    % WRF Base State Atmosphere Variables
    p00 = 100000;       % Surface pressure, Pa
    t00 = 301;          % Sea level temperature, K
    t0  = 300;          % Base state potential temperature, K
    a   = 50;           % Base state lapse rate 1000 - 300 hPa, K
    mub = p00 - p_top_requested;

    % Constants
    r_d  = 287;         % Dry gas constant
    cp   = 1004;        % Specific heat of dry air at const P.
    cvpm = -717/1004;   % -cv/cp
    g    = 9.81;        % gravity

    % WRF Standard Eta Levels
    znw = [1.000000;0.993000;0.983000;0.970000;0.954000;0.934000;0.909000;...
        0.880000;0.844022;0.808045;0.772067;0.736089;0.671281;0.610883;...
        0.554646;0.502330;0.453708;0.408567;0.366699;0.327910;0.292015;...
        0.258837;0.228210;0.199974;0.173979;0.150082;0.128148;0.108049;...
        0.089663;0.072875;0.057576;0.043663;0.031039;0.019611;0.009292;...
        0.000000];

    %% Section 1.  Find the model top

    % znu       is the eta value on the mass grid points
    % dnw       is the change in eta between eta levels
    % p         is the base state pressure on the mass grid points
    % t         is the base state temperature on the mass grid points
    % t_init    is the base state potential temperature perturbation on the mass grid points
    % alb       is the inverse base state density on the mass grid points

    for k = 1:(size(znw) - 1)
        znu(k) = (znw(k) + znw(k+1)).*0.5;
        dnw(k) = znw(k+1) - znw(k);
        p(k) = znu(k).*mub + p_top_requested;
        t(k) = t00 + a.*log(p(k)./p00);
        t_init(k) = t(k).*((p00./p(k)).^(r_d./cp)) - t0;
        alb(k) = (r_d./p00).*(t_init(k) + t0).*((p(k)./p00).^cvpm);
    end

%    fprintf('znu: ');fprintf('%d ', znu);fprintf('\n');
%    fprintf('dnw: ');fprintf('%d ', dnw);fprintf('\n');
%    fprintf('p: ');fprintf('%d ', p);fprintf('\n');
%    fprintf('t: ');fprintf('%d ', t);fprintf('\n');
%    fprintf('t_init: ');fprintf('%d ', t_init);fprintf('\n');
%    fprintf('alb: ');fprintf('%d ', alb);fprintf('\n');


    % This loops solves the hydrostatic equation using base state
    % geopotential to find the model top geopotential.

    % phb is the base state geopotential on the w grid points
    phb(1) = 0;

    for k = 2:size(znw)
        phb(k) = phb(k-1) - dnw(k-1).*mub.*alb(k-1);
    end
    fprintf('phb: ');fprintf('%d ', phb);fprintf('\n');

    % Solve for model top on the w grid points in z coords, m.
    % not used?? --> ztop = phb(length(phb))./g

    %%

    eta(1) = 1.000;

    % dz         is thickness between z levels
    % pw         is the base state pressure on the w grid points
    % tw         is the base state temperature on the w grid points
    % t_initw    is the base state potential temperature perturbation on the w grid points
    % albw       is the inverse base state density on the w grid points

    for k = 1:(length(z_levs) - 1)
        dz(k) = z_levs(k+1) - z_levs(k);
        pw(k) = eta(k).*mub + p_top_requested;
        tw(k) = t00 + a.*log(pw(k)./p00);
        t_initw(k) = tw(k).*((p00./pw(k)).^(r_d./cp)) - t0;
        albw(k) = (r_d./p00).*(t_initw(k) + t0).*((pw(k)./p00).^cvpm);
        eta(k+1) = eta(k) - ((dz(k).*g)./(mub.*albw(k)));
    end

    eta(length(z_levs)) = 0.000;

    fprintf('dz: ');fprintf('%d ', dz);fprintf('\n');
    fprintf('pw: ');fprintf('%d ', pw);fprintf('\n');
    fprintf('tw: ');fprintf('%d ', tw);fprintf('\n');
    fprintf('t_initw: ');fprintf('%d ', t_initw);fprintf('\n');
    fprintf('albw: ');fprintf('%d ', albw);fprintf('\n');
    fprintf('eta: ');fprintf('%d ', eta);fprintf('\n');

    %%
    % Section 3.  Calculate model levels on z coords from our new eta levs as a
    % visual test.  If the distribution looks good use the eta values in the
    % WRF namelist.

    % This loop calculates the base state atmosphere at the calculated
    % eta levels for use in the hydrostatic equation below.

    % znue       is the eta value on the mass grid points
    % dnwe       is the change in eta between eta levels
    % pe         is the base state pressure on the mass grid points
    % te         is the base state temperature on the mass grid points
    % t_inite    is the base state potential temperature perturbation on the mass grid points
    % albe       is the inverse base state density on the mass grid points

    for k = 1:(length(eta) - 1)
        znue(k) = (eta(k) + eta(k+1)).*0.5;
        dnwe(k) = eta(k+1) - eta(k);
        pe(k) = znue(k).*mub + p_top_requested;
        te(k) = t00 + a.*log(pe(k)./p00);
        t_inite(k) = te(k).*((p00./pe(k)).^(r_d./cp)) - t0;
        albe(k) = (r_d./p00).*(t_inite(k) + t0).*((pe(k)./p00).^cvpm);
    end

    fprintf('znue: ');fprintf('%d ', znue);fprintf('\n');
    fprintf('dnwe: ');fprintf('%d ', dnwe);fprintf('\n');
    fprintf('pe: ');fprintf('%d ', pe);fprintf('\n');
    fprintf('te: ');fprintf('%d ', te);fprintf('\n');
    fprintf('t_inite: ');fprintf('%d ', t_inite);fprintf('\n');
    fprintf('albe: ');fprintf('%d ', albe);fprintf('\n');

    % This loops solves the hydrostatic equation using base state
    % geopotential to find the model to in z coords.

    % phbe is the base state geopotential on the w grid points from the
    % calculated eta values

    phbe(1) = 0;

    for k = 2:length(eta)
        phbe(k) = phbe(k-1) - dnwe(k-1).*mub.*albe(k-1);
    end
    fprintf('phbe: ');fprintf('%d ', phbe);fprintf('\n');

    % Convert geopotential to heights in z, m.
    heights = phbe/g;
    fprintf('heights: ');fprintf('%d ', heights);fprintf('\n');

endfunction