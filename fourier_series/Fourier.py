import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sympy as sp
import scipy as sci
import matplotlib.animation as animation
from abc import ABC, abstractmethod



class _FourierTemplate(ABC):
    def __init__(self, f_sym: sp.Expr, x_sym: sp.Symbol, start: float, end: float) -> None:
        self.x_sym = x_sym
        self.f_sym = f_sym
        self.start = start
        self.end = end
        self.x_lim = self.end
        f_sym = sp.sympify(f_sym)
        if not f_sym.free_symbols:
            const = np.float64(f_sym)
            def eval_f(obj):
                if isinstance(obj, np.ndarray):
                    return np.full_like(obj, const)
                else:
                    return const
            self.eval_f = eval_f
        else:
            self.eval_f = sp.lambdify(self.x_sym, self.f_sym, 'numpy')


    def _set_sym_sequences(self, a_0, a_n, b_n) -> None:
        """Sets the symbolic sp.Sequences of the Fourier Series.
        Should be called by each subclass.
        Sequences are subscriptable objects.
        """
        self.a0_sym_sequence = a_0
        self.an_sym_sequence = a_n
        self.bn_sym_sequence = b_n


    def eval_series(self, N):
        series: sp.Expr = self.a0_sym_sequence
        series += sum(self.an_sym_sequence[:N]) + sum(self.bn_sym_sequence[:N])
        if not series.free_symbols:
            def func(obj):
                if isinstance(obj, np.ndarray):
                    return np.full_like(obj, series)
                return series
            
            return func
        
        return sp.lambdify(self.x_sym, series, 'numpy')


    def plot_series(self, ax, N, show_f=True, centre=True, sample=1000, x_lim=1, **kwargs) -> None:
        """Plots the partial sum up to N on a specified axis.
        For formatting and centering, call format_plot first
        Kwargs apply to plot of fourier series."""
        
        if centre:
            self.centre_axis(ax, x_lim, sample=sample)

        series = self.eval_series(N)
        x_sample = np.linspace(-x_lim, x_lim, sample)
        y = series(x_sample)
        style = {"linewidth": 1, "label": f"N = {N}"}
        style.update(kwargs)
        ax.plot(x_sample, y, **style)


        if show_f:
            x_restricted = np.linspace(-self.x_lim, self.x_lim, sample)
            ax.plot(x_restricted, self.eval_f(x_restricted), linewidth=0.5, label=r"$f(x)$")

        ax.legend()


    def centre_axis(self, ax, x_lim, err = 0.2, sample=1000) -> None:
        x_sample = np.linspace(self.start, self.end, sample)
        y_lim = max(int(np.max(np.absolute(self.eval_f(x_sample)))), 1)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        if x_lim:
            ax.set_xlim(-x_lim*(1+err), x_lim*(1+err))
            ax.set_ylim(-y_lim*(1+err), y_lim*(1+err))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2*x_lim, -x_lim))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_ticks_position('bottom')

        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_lim, -y_lim))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which='both',linewidth=0.2)

        ax.spines['bottom'].set_position(('data',0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data',0))
        

    @abstractmethod
    def _init_sym_sequences(self) -> None:
        pass


    @abstractmethod
    def __str__(self) -> str:
        pass



class SymbolicFourier(_FourierTemplate):
    def __init__(self, f_sym: sp.Expr, x_sym: sp.Symbol, start: float=-1, end: float=1, half=None):
        super().__init__(f_sym, x_sym, start, end)
        self.start = sp.sympify(self.start)
        self.end = sp.sympify(self.end)
        self.x_lim = np.float64(self.end)
        self.half = half
        self._init_sym_sequences()
        self._set_sym_sequences(self.a_0, self.a_n, self.b_n)

    def _init_sym_sequences(self) -> None:
        # compute symbolic coefficients on [-1,1] (or half-range [0,1])
        x = self.x_sym
        f = self.f_sym

        if not self.half:
            s = sp.fourier_series(f, (x, self.start, self.end))
            self.a_n = s.an
            self.a_0 = s.a0
            self.b_n = s.bn
        else:
            pi = sp.pi
            n = sp.symbols("n", positive=True, integer=True)
            match self.half:
                case "sin":
                    L = (self.end - self.start)
                    sin_term = sp.sin(n*pi*x / L)
                    limits = (x, self.start, self.end)
                    self.b_n = sp.SeqFormula(
                        2 * sin_term * sp.integrate(f * sin_term, limits)
                      / L, (n, 1, sp.oo))
                    self.a_n = sp.SeqFormula(0, (n, 1, sp.oo))
                    self.a_0 = 0
                
                case "cos":
                    L = (self.end - self.start)
                    cos_term = sp.cos(n*pi*x / L)
                    limits = (x, self.start, self.end)
                    self.a_0 = sp.integrate(f, limits) / L
                    self.a_n = sp.SeqFormula(
                        2 * cos_term * sp.integrate(f * cos_term, limits)
                      / L, (n, 1, sp.oo))
                    self.b_n = sp.SeqFormula(0, (n, 1, sp.oo))

    def __str__(self) -> str:
        const = self.a_0
        n_pos = sp.symbols("n", positive=True, integer=True)
        a_n = self.a_n.formula.subs(self.a_n.variables[0], n_pos)
        b_n = self.b_n.formula.subs(self.b_n.variables[0], n_pos)
        trig = sp.simplify(a_n + b_n)
        return rf"{sp.latex(const)} + \sum_{{n=1}}^\infty {sp.latex(trig)}"



class NumericFourier(_FourierTemplate):
    """Computes fourier series by using numeric integration. 
    Coefficients are accurate when n < 500."""
    def __init__(self, f_sym: sp.Expr, x_sym: sp.Symbol, start: float=-1, end: float=1, half=None, quad_kwargs:dict|None = None):

        self.half = half
        if quad_kwargs is None:
            self.quad_kwargs = {}
        else:
            self.quad_kwargs = quad_kwargs

        super().__init__(f_sym, x_sym, start, end)
        self.x_lim = self.end
        self._init_sym_sequences()
        self._set_sym_sequences(self.a_0, self.a_n, self.b_n)

    @staticmethod
    def num_integrate(integrand: callable, a: float, b: float, **kwargs) -> float:
        return sci.integrate.quad(integrand, a, b, **kwargs)[0]

    def _init_sym_sequences(self) -> None:
        class NumericSeq:
            def __init__(
                    self, integrand_func: sp.Expr, trig_term: sp.Expr, trig_type,
                    x_sym: sp.Symbol, n_sym: sp.Symbol, 
                    a: float, b: float, quad_kwargs: dict, half=False
                    ) -> None:
                """Generates the sequence of sympy terms for the fourier series."""
                self.integrand_func = integrand_func
                self.trig_term = trig_term
                self.x_sym = x_sym
                self.n_sym = n_sym
                self.a = a
                self.b = b
                self.L = (self.b - self.a) if half else (self.b - self.a) / 2
                self.quad_kwargs = quad_kwargs
                self.trig_type = trig_type


            def _get_single(self, n: int):
                n = n+1 # Since first term is always 0
                w = self.trig_type
                wvar = np.pi * n / self.L
                integrand = sp.lambdify(
                    self.x_sym, self.integrand_func, 'numpy')
                
                coef = NumericFourier.num_integrate(integrand, a, b, weight=w, wvar=wvar, **self.quad_kwargs)
                trig = self.trig_term.subs(self.n_sym, n)
                return coef * trig


            
            def __getitem__(self, key: int | slice):
                """Returns the term or list of sympy terms in the series"""
                if isinstance(key, int):
                    return self._get_single(key)

                elif isinstance(key, slice):
                    start = key.start or 0
                    stop = key.stop
                    step = key.step or 1

                    if stop is None:
                        raise ValueError("Slice must have stop for infinite sequence")

                    return [self._get_single(i) for i in range(start, stop, step)]

                else:
                    raise TypeError("Index must be int or slice")
        


        f = self.eval_f
        f_sym = self.f_sym
        a = self.start
        b = self.end
        x_sym = self.x_sym
        n_sym = sp.symbols("n", integer=True, positive=True)
        L = (b - a) / 2
        kwargs = self.quad_kwargs

        if not self.half:
            self.a_0 = sp.sympify(1 / (2*L) * NumericFourier.num_integrate(f, a, b, **self.quad_kwargs))
            cos_term = sp.cos(n_sym * sp.pi * x_sym / L)
            cos_integrand = 1 / L * f_sym
            sin_term = sp.sin(n_sym * sp.pi * x_sym / L)
            sin_integrand = 1 / L * f_sym
            self.a_n = NumericSeq(cos_integrand, cos_term, 'cos', x_sym, n_sym, a, b, kwargs)
            self.b_n = NumericSeq(sin_integrand, sin_term, 'sin', x_sym, n_sym, a, b, kwargs)

        else:
            match self.half:
                case "cos":
                    self.a_0 = sp.sympify(1 / (2*L) * NumericFourier.num_integrate(f, a, b, **self.quad_kwargs))
                    cos_term = sp.cos(n_sym * sp.pi * x_sym / (2*L))
                    cos_integrand = 1 / L * f_sym
                    self.a_n = NumericSeq(cos_integrand, cos_term, 'cos', x_sym, n_sym, a, b, kwargs, half=True)
                    self.b_n = sp.SeqFormula(0, (n_sym, 0, sp.oo))

                case "sin":
                    sin_term = sp.sin(n_sym * sp.pi * x_sym / (2*L))
                    sin_integrand = 1 / L * f_sym
                    self.b_n = NumericSeq(sin_integrand, sin_term, 'sin', x_sym, n_sym, a, b, kwargs, half=True)
                    self.a_0 = sp.Integer(0)
                    self.a_n = sp.SeqFormula(0, (n_sym, 0, sp.oo))



    def __str__(self) -> str:
        series = self.a_0 + sum(self.a_n[:4]) + sum(self.b_n[:4])
        return sp.latex(series)



class Animate:
    def __init__(self, fourier: NumericFourier|SymbolicFourier) -> None:
        self.fourier = fourier
        fourier = self.fourier

        self.x_lim = fourier.x_lim
        self.eval_f = fourier.eval_f
        self.a_0 = fourier.a_0
        self.a_n = fourier.an_sym_sequence
        self.b_n = fourier.bn_sym_sequence
        self.x_sym = fourier.x_sym
        self.start = fourier.start
        self.end = fourier.end


    def convergence(self, fig, ax, N_range=100, step=1, sample=1000, lim=None, interval=50):
        """Animates the Fourier Partial Sums to show convergence to f
        Will not work if series is constant.
        Returns the animation object"""
        x_lim = self.x_lim
        if lim is None:
            lim = x_lim
        self.fourier.centre_axis(ax, lim, sample=sample)
        x_f = np.linspace(-x_lim, x_lim, sample)
        x = np.linspace(-lim, lim, sample)
        ax.plot(x_f, self.eval_f(x_f), label = r"$f(x)$", linewidth=0.5)

        const = np.float64(self.a_0)if self.a_0 else 0
        graph, = ax.plot(x, np.zeros_like(x), label=r"Series")
        ax.legend()
        terms = [lambda arr:np.full_like(arr, const)]

        def init():
            graph.set_data(x, np.zeros_like(x))
            
            return graph

        for k in range(0, N_range, step):
            partial = sum(self.a_n[k:k+step]) + sum(self.b_n[k:k+step])
            terms.append(sp.lambdify(self.x_sym, partial, 'numpy'))  

        def run(i):
            y = graph.get_ydata()
            g = terms[i]
            y += g(x)
            graph.set_ydata(y)
            ax.set_title(f"N = {i * step}")
            return graph
                
        return animation.FuncAnimation(
            fig,
            run,
            interval=interval,
            blit=False,  # blitting can't be used with Figure artists
            frames=range(len(terms)),
            init_func=init
        )

    def heat(self, fig, ax, heat_ax=None, N = 100,
             t0: float= 0, t1: float = 1, sample=1000, t_sample = 100, delay=1, interval=50):
        """Animates the Fourier Partial Sums to show convergence to f
        Will not work if series is constant.
        Returns the animation object
        Only supports sine series."""
        err = 0.2
        x_start = np.floor(float(self.start))
        x_end = np.ceil(float(self.end))
        t = sp.symbols("t", real=True, positive=True)
        L = self.end - self.start
        n = sp.symbols("n", integer=True, positive=True)
        exp = sp.SeqFormula(sp.exp(-n**2*sp.pi**2/L**2 * t), (n, 1, sp.oo))
        series = sp.sympify(0)
        for (sin, exp) in zip(self.b_n[:N], exp[:N]):
            series += sin * exp
        if not series.free_symbols:
            raise ValueError("The series has no sin term.")


        series = sp.lambdify((self.x_sym, t), series, 'numpy')

        x = np.linspace(x_start, x_end, sample)
        t = np.linspace(t0, t1, t_sample)

        y0 = series(x,0)
        ymax = np.max(y0)
        ymin = np.min(y0)
        ax.set_xlim(x_start-err, x_end+err)
        ax.set_ylim(ymin-err, ymax+err)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2*self.x_lim, -self.x_lim))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        

        
        graph, = ax.plot(x, np.zeros_like(x))
        # Frames artificially expanded to create a pause when t < 0
        interval = 50
        duration = int(np.ceil(delay * 1000 / interval))

        # Include heat map
        if heat_ax is not None:
            heat_ax.xaxis.set_major_locator(ticker.LinearLocator(3))
            heat_ax.xaxis.set_tick_params(labelbottom=False)
            heat_ax.xaxis.set_ticks_position("both")
            heat_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 

            gradient = y0
            gradient = np.vstack((y0, y0))
            im = heat_ax.imshow(gradient, aspect='auto', cmap='coolwarm')
            heat_ax.yaxis.set_major_locator(ticker.NullLocator())
            fig.colorbar(im, ax=heat_ax, label='Temperature', location='bottom')
            def init():
                graph.set_data(x, y0)
                im.set_data(gradient)
                
                return graph, im
        
            def run(t):
                if t<0:
                    ax.set_title("t=0")
                    return graph
                elif t > t1:
                    ax.set_title(f"t={t1}")
                    return graph
                T = series(x, t)
                graph.set_ydata(T)
                im.set_data(np.vstack((T,T)))
                ax.set_title(f"t = {t}")

                return graph

        # Version without heat map
        else:
            def init():
                graph.set_data(x, y0)
                
                return graph
        
            def run(t):
                if t<0:
                    ax.set_title("t=0")
                    return graph
                elif t > t1:
                    ax.set_title(f"t={t1}")
                    return graph
                T = series(x, t)
                graph.set_ydata(T)
                ax.set_title(f"t = {t}")

                return graph
                
        return animation.FuncAnimation(
            fig,
            run,
            interval=interval,
            blit=False,  # blitting can't be used with Figure artists
            frames=np.concatenate((np.full(duration, -1),t, np.full(duration, t1+1))),
            init_func=init
        )
