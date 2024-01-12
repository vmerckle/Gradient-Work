%%
% Simulation of crowd motion through JKO flow.
% The function to be minimized is
%       f(p) = <w,p> + h(p)
% w act as an attraction potential.

set(0,'DefaultFigureWindowStyle','docked');

name = 'disk1';

N = 30;

%%
% helpers

normalize = @(x)x/sum(x(:));
% set figure title
setfigname = @(name)set(gcf, 'Name', name, 'NumberTitle','off');
t = linspace(0,1,N); [Y,X] = meshgrid(t,t);
gaussian = @(m,s)exp( -( (X-m(1)).^2+(Y-m(2)).^2 )/(2*s^2) );

%% 
% Load attracting potential and initial density


t = linspace(0,1,N); [Y,X] = meshgrid(t,t);
gaussian = @(m,s)exp( -( (X-m(1)).^2+(Y-m(2)).^2 )/(2*s^2) );
rectange = @(c,w)double( max( abs(X-c(1))/w(1), abs(Y-c(2))/w(2) )<1 );
disk = @(m,s) (X-m(1)).^2+(Y-m(2)).^2 <=s^2;

normalize = @(x)x/sum(x(:));
doclamp = .7;

p0 = gaussian([.5 .9], .14);        
% mask
r = .23;
mask = double( (X-.5).^2+(Y-.45).^2>=r^2 );
% target point
target = round(N*[.5 .03]);
w = .5*Y;

f = @(u)normalize( u ); % useful to shift because of masked values
f = @(u)normalize( min(u,max(u(:))*doclamp) );
p0 = f(p0.*mask+1e-10);

% compute a geodesic metric and geodesic potential
vmin = 0; 
M = zeros(N,N,2,2);
M(:,:,1,1) = mask+vmin; M(:,:,2,2) = mask+vmin;

%%
% Parameters, here for N=100
% good low diffusion params: tau=0.05, gam=0.0001
% kinda lower limit on diffusion: tau = 0.000015;gamma = 0.00000001; CANNOT
% increase tau or it just never solves

tau = 0.000015;
gamma = 0.00000001;
niter = 1000;
nsteps = 100;
model = 'crowd';

%%
% Load the proximal map.
% crowd motion clamping

kappa = max(p0(:)); % box constraint on the density
proxf = @(p,sigma)min(p .* exp(-sigma*w),kappa);  

%%
% Gibbs kernel
% anisotropic metric
using_factorization = 1;
opt.CholFactor = gamma;        
opt.laplacian_type = 'fd'; % finite differences
[blur, Delta, Grad] = blurAnisotropic(M,opt);
% blur: function, Delta:900x900 sparse, Grad: 1800x900 sparse..
filtIter = 5;
K = @(x)blur(x,gamma,filtIter);

%%
% Perform several iterates.

figure(3); setfigname('Gif');

tol = 1e-6;

verb = 1;
DispFunc = @(p)imageplot( format_image(p,mask) ); % 
WriteFunc = [];

% helpers
mynorm = @(x)norm(x(:));

q = p0; p = p0;
p_list = {p};
for it=1:nsteps-1
	%progressbar(it,nsteps-1); 
	q = p;

	%slim try
	if 1
	Constr  = {[] []};
	b = w*0+1;
	for i=1:niter
		p = proxf(K(b), tau/gamma);
        %disp(sum(p(:))); % yes this can be far from 1 at start
		a = p ./ K(b);
		Constr{2}(end+1) = mynorm( b.*K(a)-q )/mynorm(q);
		b = q ./ K(a);
		Constr{1}(end+1) = mynorm( a.*K(b)-p )/mynorm(q);
   		if Constr{1}(end)<tol && Constr{2}(end)>tol
            disp(['only (1) is satisfied: K b a - p, Counter i = ' num2str(i)]);
        end
        if Constr{1}(end)>tol && Constr{2}(end)<tol
            disp(['only (2) is satisfied: b K a - q, Counter i = ' num2str(i)]);
        end
        

		if Constr{1}(end)<tol && Constr{2}(end)<tol
            disp(['both at i = ' num2str(i)]);
			break;
		end
    end
    end
    if Constr{1}(end)>tol && Constr{2}(end)>tol
        disp("ran out of iter");
	end

	% PERFORM JKO STEPPINg
	if 0
	uu = w*0+1; % vector of ones
	%%% projection init %%%
	a = uu; b = uu;
	%%% Dykstra init %%%
	u = uu; u1 = uu; 
	v = uu; v1 = uu; 

	Constr  = {[] []};
	for i=1:niter
		a1 = a;  b1 = b;
		u2 = u1; u1 = u;
		v2 = v1; v1 = v;

		ta1 = a1 .* u2;  
		b = b1 .* v2;
		p = proxf(ta1.*K(b), tau/gamma);
		a = p ./ K(b);            
		Constr{2}(end+1) = mynorm( b.*K(a)-q )/mynorm(q);
		u = u2 .* a1 ./ a;
		v = v2 .* b1 ./ b;

		a1 = a;  b1 = b;
		u2 = u1; u1 = u;
		v2 = v1; v1 = v;

		a = a1 .* u2;
		b = q ./ K(a);
		Constr{1}(end+1) = mynorm( a.*K(b)-p )/mynorm(q);
		u = u2 .* a1 ./ a;
		v = v2 .* b1 ./ b;

		if Constr{1}(end)<tol && Constr{2}(end)<tol
			break;
		end
	end
	end
	% PERFORM JKO STEPPINg
	p_list{it+1} = p;
	DispFunc(p);
	drawnow;
end


% sanity checks
totalSum = sum(cellfun(@(x) sum(x(:)), p_list));
fprintf('Total sum of all matrices: %g\n', totalSum);
lastMatrix = p_list{end};
sumLastMatrix = sum(lastMatrix(:));
fprintf('Sum of the last matrix: %g\n', sumLastMatrix);
lastMatrix = Constr{1};
sumLastMatrix = sum(lastMatrix(:));
fprintf('Sum of the constr1: %g\n', sumLastMatrix);
lastMatrix = Constr{2};
sumLastMatrix = sum(lastMatrix(:));
fprintf('Sum of the constr1: %g\n', sumLastMatrix);
