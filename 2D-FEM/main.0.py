import taichi as ti

ti.init()


N: int = 12  # number of squares per line
dt = 5e-5
dx = 1 / N
rho = 4e1  # mass density
num_elements = 2 * N**2  # number of elements
num_vertices = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus, Poisson's ratio
mu = E / 2 / (1 + nu)  # shear modulus
lambda_L = E * nu / (1 + nu) / (1 - 2 * nu)  # Lame's first parameter, lambda
C_1 = mu / 2  # Neo-Hookean solid
D_1 = lambda_L / 2  # Neo-Hookean solid
circle_position, circle_radius = ti.Vector([0.5, 0.0]), 0.31
damping = 14.5


gravity: ti.MatrixField = ti.Vector.field(n=2, dtype=float, shape=())
attractor_position: ti.MatrixField = ti.Vector.field(n=2, dtype=float, shape=())
attractor_strength: ti.ScalarField = ti.field(dtype=float, shape=())


position: ti.MatrixField = ti.Vector.field(
    n=2, dtype=float, shape=num_elements, needs_grad=True
)
velocity: ti.MatrixField = ti.Vector.field(n=2, dtype=float, shape=num_vertices)
element_to_vertices_idx: ti.MatrixField = ti.Vector.field(
    n=3, dtype=int, shape=num_elements
)  # counter-clockwise
U: ti.ScalarField = ti.field(
    dtype=float, shape=(), needs_grad=True
)  # total potential energy
rest_shape_inv: ti.MatrixField = ti.Matrix.field(
    n=2, m=2, dtype=float, shape=num_elements
)
V: ti.ScalarField = ti.field(dtype=float, shape=num_elements)  # volume
W: ti.ScalarField = ti.field(dtype=float, shape=num_elements)  # strain energy density


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        bottom_left_idx = i * (N + 1) + j
        top_left_idx = bottom_left_idx + 1
        bottom_right_idx = bottom_left_idx + N + 1
        top_right_idx = bottom_right_idx + 1
        element_to_vertices_idx[k] = [bottom_left_idx, bottom_right_idx, top_right_idx]
        element_to_vertices_idx[k + 1] = [top_right_idx, top_left_idx, bottom_left_idx]


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        position[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
        velocity[k] = ti.Vector([0, 0])
    for i in range(num_elements):
        v0idx, v1idx, v2idx = element_to_vertices_idx[i]
        v0pos, v1pos, v2pos = position[v0idx], position[v1idx], position[v2idx]
        rest_shape = ti.Matrix.cols([v1pos - v0pos, v2pos - v0pos])  # counter-clockwise
        rest_shape_inv[i] = rest_shape.inverse()


def paint(gui: ti.GUI) -> None:
    # paint pressure
    pos = position.to_numpy()
    w = abs(W.to_numpy())
    e2v = element_to_vertices_idx.to_numpy()
    v0pos, v1pos, v2pos = pos[e2v[:, 0]], pos[e2v[:, 1]], pos[e2v[:, 2]]
    k = w * (10 / E)
    gb = (1 - k) * 0.5
    gui.triangles(v0pos, v1pos, v2pos, color=ti.rgb_to_hex([k + gb, gb, gb]))

    mouse_pos: tuple[float, float] = gui.get_cursor_pos()
    gui.circle(pos=mouse_pos, radius=15, color=0x336699)
    gui.circle(pos=circle_position, radius=int(circle_radius * 512), color=0x666666)
    gui.circles(pos=position.to_numpy(), radius=2, color=0xFFAA33)


@ti.kernel
def calc_energy():
    for i in range(num_elements):
        v0idx, v1idx, v2idx = element_to_vertices_idx[i]
        v0pos, v1pos, v2pos = position[v0idx], position[v1idx], position[v2idx]
        e1, e2 = v1pos - v0pos, v2pos - v0pos
        V[i] = abs(e1.cross(e2))
        shape = ti.Matrix.cols([e1, e2])
        F = shape @ rest_shape_inv[i]  # deformation gradient
        C = F.transpose() @ F  # right Cauchy-Green deformation tensor
        I_1 = C.trace()  # first invariant of C
        J = F.determinant()
        W_i = (
            C_1 * (I_1 - 2 - 2 * ti.log(J)) + D_1 * (J - 1) ** 2
        )  # strain energy density
        W[i] = W_i
        U[None] += V[i] * W_i

        # ! WRONG APPROACH
        # volume = abs(e1.cross(e2))
        # U[None] += volume * W_i
        # `V` should not have grad due to principle of virtual work.

        # ! WRONG APPROACH
        # U[None] += volume * W[i]
        # `W[i]` does not have grad, something is likely to go wrong.


@ti.kernel
def advance():
    # acceleration
    for i in range(num_vertices):
        dm = rho * dx**2
        g = (
            gravity[None] * 0.8
            + attractor_strength[None]
            * (attractor_position[None] - position[i]).normalized()
        )
        acceleration = g * 40
        if not ti.math.isnan(position.grad[i]).any():
            acceleration -= position.grad[i] / dm
        velocity[i] += acceleration * dt

        # circle boundary
        position_to_center = position[i] - circle_position
        distance_to_center = position_to_center.norm()
        if distance_to_center <= circle_radius:
            normal_velocity = velocity[i].dot(position_to_center.normalized())
            if normal_velocity < 0:
                velocity[i] -= 1.5 * normal_velocity * position_to_center.normalized()

        # window boundary
        lower_collision = (position[i] <= 0) & (velocity[i] < 0)
        upper_collision = (position[i] >= 1) & (velocity[i] > 0)
        collision = lower_collision | upper_collision
        for j in ti.static(range(position.n)):
            if collision[j]:
                velocity[i][j] = 0

        velocity[i] *= ti.exp(-damping * dt)
        position[i] += velocity[i] * dt


def main() -> None:
    init_mesh()
    init_pos()
    gravity[None] = [0, -1]

    gui: ti.GUI = ti.GUI()
    pause: bool = False
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            match e.key:
                case "q" | gui.ESCAPE:
                    gui.running = False
                case gui.SPACE:
                    pause = not pause
                case "r":
                    init_pos()
                case "a" | gui.LEFT:
                    gravity[None] = [-1, 0]
                case "s" | gui.DOWN:
                    gravity[None] = [0, -1]
                case "d" | gui.RIGHT:
                    gravity[None] = [+1, 0]
                case "w" | gui.UP:
                    gravity[None] = [0, +1]
        mouse_pos: tuple[float, float] = gui.get_cursor_pos()
        attractor_position[None] = mouse_pos
        attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB)
        if not pause:
            for _ in range(50):
                with ti.ad.Tape(loss=U):
                    calc_energy()
                advance()
        paint(gui=gui)
        gui.show()


if __name__ == "__main__":
    main()
