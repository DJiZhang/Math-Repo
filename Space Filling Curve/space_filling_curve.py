
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation



class SpaceFillingCurve:
    """Creates an array of length iterations+1, where each element contains the location 
    of the vertices for drawing the space filling curve.
    If iteration = n, the last array has length 4**n."""
    def __init__(self, iterations=4) -> None:
        self.x1 = np.array([0,0,1,1])
        self.y1 = np.array([0, 0.5, 0.5, 0])
        self.x_arr = [self.x1]
        self.y_arr = [self.y1]
        self.iterations = iterations
        self.create_curve()


    def create_curve(self):
        for _ in range(self.iterations):
            x = self.x_arr[-1]
            y = self.y_arr[-1]
            # New x_n
            self.x_arr.append(
                np.concatenate((0.5*y, 0.5*x, 0.5+0.5*x, 1 - 0.5*y))
                )
            self.y_arr.append(
                np.concatenate((0.5*x, 0.5+0.5*y, 0.5 + 0.5*y, 0.5 - 0.5*x))
            )

    @staticmethod
    def format_plot(ax):
        err = 0.01
        ax.set_xlim(-err, 1+err)
        ax.set_ylim(-err, 1+err)
        ax.set_aspect('equal')
    
    def animate(self, fig, ax, rate="linear", n=None, linewidth: float=0.5):
        if n is None:
            n = self.iterations
        match rate:
            case "linear":
                ax.set_title(f"Linear Rate Animation for n = {n}")
                frames = list(range(len(self.x_arr[n]))) + [-1]*200
                interval = 0
            case "exp":
                ax.set_title(f"Exponential Rate Animation for n = {n} ")
                frames = list(4**i for i in range(0, n+2)) + [-1]*5
                interval = 500
            case _:
                raise ValueError(f"Invlaid rate {rate}. Should be \"linear\" or \"exp\" ")
            
        # Format plot
        self.format_plot(ax)

        return self._animate(fig, ax, n, linewidth, frames, interval)

    def _animate(self, fig, ax, n, linewidth, frames, interval):
        index = n
        x_data = self.x_arr[n]
        y_data = self.y_arr[n]
        Line, = ax.plot([0], [0], label=f"n = {n}", linewidth=linewidth)
        

        def run(i):
            if i < 0:
                return Line,
            Line.set_data(x_data[:i], y_data[:i])
            return Line,

        return animation.FuncAnimation(
            fig,
            run,
            interval=interval,
            blit=True,
            frames = frames
        )

    def plot(self, fig, ax, n=None, linewidth=0.5):
        if n is None:
            n = self.iterations
        else:
            if n > self.iterations or n < 0:
                print(f"Input n = {n} is invalid. Proceeding to plot for n = {self.iterations}")
        self.format_plot(ax)
        ax.plot(self.x_arr[n], self.y_arr[n], linewidth=linewidth, label=f"n={n}")
        ax.set_title("Space Filling Curve")


def main():
    n = 6
    curve = SpaceFillingCurve(n)

    fig, ax = plt.subplots(1, 3)
    ax = ax.flatten()

    # Simple Plotting
    curve.plot(fig, ax[0])
    ax[0].set_title(f"Simple Plot of n = {n}")

    # Linear Animation
    ani1 = curve.animate(fig, ax[1])

    # Exp Animation
    ani2 = curve.animate(fig, ax[2], rate="exp")
    plt.show()

if __name__ == "__main__":
    main()
