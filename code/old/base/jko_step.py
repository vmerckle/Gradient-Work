import numpy as np

def perform_jko_stepping(K, q, gamma, tau, proxf, options):
    """
    perform_jko_stepping - perform JKO descent steps
%
%   p,Constr = perform_jko_stepping(K,q, gamma,tau, proxf, options)
%
%   If options.nsteps=1, solve for
%       JKO_{tau*f}(q) = argmin_p  W_gamma(q,p) + tau*f(p)
%   Otherwise, initialize p=q, and perform options.nsteps iterations of
%       q <- JKO_{tau*f}(q)
%
%   proxf(p,sigma) must implement the KL prox
%       argmin_{p'} KL(p'|p) + sigma*f(p)
%   Set proxf=None if no function f is used. 
%
%   Here W_gamma is the entropically smooth wasserstein distance, 
%   for a ground metric c, and 
%   where the Gibbs kernel exp(-c/gamma) is implemented by the callback
%   function K(x) (e.g. a Gaussian blurring)
%

% iterates are
%    pi = diag(a)*K*diag(b)
% should converge to the constraint
%    (C1) pi^T*1 = b.*K(a) = q  
%    (C2) pi*1 = a.*K(b) = p
    """

    niter = options.get('niter', 100)
    nsteps = options.get('nsteps', 200)
    verb = options.get('verb', False)
    DispFunc = options.get('DispFunc', None)
    WriteFunc = options.get('WriteFunc', None)
    tol = options.get('tol', 1e-6)
    prox_postprocess = options.get('prox_postprocess', 0)

    if proxf is None:
        proxf = lambda p, sigma: p

    mynorm = lambda x: np.linalg.norm(x.flatten())

    if nsteps > 1:
        options['nsteps'] = 1
        p = q
        p_list = [p]
        for it in range(nsteps-1):
            if verb == 1:
                print(f"iteration {it+1}", end="")
            p, Constr, _ = perform_jko_stepping(K, p, gamma, tau, proxf, options)
            if prox_postprocess == 1:
                p = proxf(p, 1)
            p_list.append(p)
            if DispFunc is not None:
                DispFunc(p)
                drawnow()
            if WriteFunc is not None:
                WriteFunc(p, it)
        return p, Constr, p_list

    uu = np.ones_like(q)
    a = uu
    b = uu
    u = uu
    u1 = uu
    v = uu
    v1 = uu

    Constr = [[], []]
    p_list = []

    #print(q)
    for i in range(niter):
        if verb == 1:
            print(f".", end="", flush=True)

        for it in range(2):
            a1 = a.copy()
            b1 = b.copy()
            u2 = u1.copy()
            u1 = u.copy()
            v2 = v1.copy()
            v1 = v.copy()

            if it == 1:
                a = a1 * u2
                b = q / K(a)
                Constr[0].append(mynorm(a * K(b) - p) / mynorm(q))
            else:
                ta1 = a1 * u2
                b = b1 * v2
                p = proxf(ta1 * K(b), tau / gamma)
                a = p / K(b)
                Constr[1].append(mynorm(b * K(a) - q) / mynorm(q))

            u = u2 * a1 / a
            v = v2 * b1 / b

        if len(Constr[0]) > 0 and len(Constr[1]) > 0:
            if Constr[0][-1] < tol and Constr[1][-1] < tol:
                if verb == 1:
                    print("     finished")
                return p, Constr, None
    print()
    return p, Constr, None

