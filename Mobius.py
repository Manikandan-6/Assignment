import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self.generate_mesh()

    def generate_mesh(self):
        U = self.U
        V = self.V
        R = self.R

        X = (R + V * np.cos(U / 2)) * np.cos(U)
        Y = (R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        # Partial derivatives
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]

        Xu = np.gradient(self.X, du, axis=1)
        Yu = np.gradient(self.Y, du, axis=1)
        Zu = np.gradient(self.Z, du, axis=1)

        Xv = np.gradient(self.X, dv, axis=0)
        Yv = np.gradient(self.Y, dv, axis=0)
        Zv = np.gradient(self.Z, dv, axis=0)

        # Cross product of partial derivatives
        Nx = Yu * Zv - Zu * Yv
        Ny = Zu * Xv - Xu * Zv
        Nz = Xu * Yv - Yu * Xv

        dS = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        surface_area = np.sum(dS) * du * dv
        return surface_area

    def compute_edge_length(self):
        # Boundary at v = -w/2 and v = w/2 (2 edges)
        edge1 = self.X[0, :], self.Y[0, :], self.Z[0, :]
        edge2 = self.X[-1, :], self.Y[-1, :], self.Z[-1, :]

        def compute_length(edge):
            x, y, z = edge
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

        return compute_length(edge1) + compute_length(edge2)

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Möbius Strip")
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=200)
    area = mobius.compute_surface_area()
    edge_len = mobius.compute_edge_length()

    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {edge_len:.4f}")

    mobius.plot()
