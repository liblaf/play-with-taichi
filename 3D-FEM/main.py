import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)


# camera
camera_position = ti.Vector([0, -1, 0.5])
camera_lookat = ti.Vector([0, 0, 0.2])


# ground
ground_mesh_vertices: ti.MatrixField = ti.Vector.field(n=3, dtype=float, shape=4)
ground_mesh_vertices[0] = [-1, -1, 0]
ground_mesh_vertices[1] = [+1, -1, 0]
ground_mesh_vertices[2] = [+1, +1, 0]
ground_mesh_vertices[3] = [-1, +1, 0]
ground_mesh_indices: ti.ScalarField = ti.field(dtype=int, shape=2 * 3)
ground_mesh_indices[0] = 0
ground_mesh_indices[1] = 1
ground_mesh_indices[2] = 2
ground_mesh_indices[3] = 2
ground_mesh_indices[4] = 3
ground_mesh_indices[5] = 0


# ball
ball_center = ti.Vector.field(n=3, dtype=float, shape=1)
ball_center[0] = [0, 0, 0]
ball_radius = 0.2


# physical constants
ball_restitution: float = 0.8  # coefficient of restitution
ground_restitution: float = 0.8  # coefficient of restitution
delta_t = 6e-4
E, nu = 1e4, 0.2  # Young's modulus, Poisson's ratio
gravity_direction: ti.MatrixField = ti.Vector.field(n=3, dtype=float, shape=())
gravity: float = 9.8
rho: float = 1e3  # mass density

mu = E / 2 / (1 + nu)  # shear modulus
lambda_L = E * nu / (1 + nu) / (1 - 2 * nu)  # Lame's first parameter, lambda
C_1 = mu / 2  # Neo-Hookean solid
D_1 = lambda_L / 2  # Neo-Hookean solid
damping = 10


# cuboid
cube_edge_length = 0.02
cuboid_origin = ti.Vector([0, 0, 0.5])
cuboid_shape = np.array([8, 8, 8])


num_elements_per_cube: int = 5
num_vertices = int(np.prod(cuboid_shape + 1))
num_elements = int(num_elements_per_cube * np.prod(cuboid_shape))
element_to_vertex_index: ti.MatrixField = ti.Vector.field(
    n=4, dtype=int, shape=num_elements
)


position: ti.MatrixField = ti.Vector.field(
    n=3, dtype=float, shape=num_vertices, needs_grad=True
)
velocity: ti.MatrixField = ti.Vector.field(
    n=3, dtype=float, shape=num_vertices, needs_grad=False
)
volumns: ti.ScalarField = ti.field(dtype=float, shape=num_elements, needs_grad=False)
rest_shape_inv: ti.MatrixField = ti.Matrix.field(
    n=3, m=3, dtype=float, shape=num_elements, needs_grad=False
)
stress_energy_density: ti.ScalarField = ti.field(
    dtype=float, shape=num_elements, needs_grad=True
)
total_energy: ti.ScalarField = ti.field(dtype=float, shape=(), needs_grad=True)


@ti.func
def vertex_ijk_to_idx(i: int, j: int, k: int) -> int:
    return k + (cuboid_shape[2] + 1) * (j + (cuboid_shape[1] + 1) * i)


@ti.func
def cube_ijk_to_idx(i: int, j: int, k: int) -> int:
    return k + cuboid_shape[2] * (j + cuboid_shape[1] * i)


@ti.kernel
def init_vertices():
    for i, j, k in ti.ndrange(*(cuboid_shape + 1)):
        vertex_idx = vertex_ijk_to_idx(i=i, j=j, k=k)
        position[vertex_idx] = (
            cuboid_origin + ti.Vector([i, j, k], dt=float) * cube_edge_length
        )


@ti.kernel
def init_elements():
    for i, j, k in ti.ndrange(*cuboid_shape):
        cube_idx = cube_ijk_to_idx(i=i, j=j, k=k)
        element_to_vertex_index[5 * cube_idx + 0] = [
            vertex_ijk_to_idx(i + 0, j + 0, k + 1),
            vertex_ijk_to_idx(i + 0, j + 1, k + 0),
            vertex_ijk_to_idx(i + 1, j + 0, k + 0),
            vertex_ijk_to_idx(i + 1, j + 1, k + 1),
        ]
        element_to_vertex_index[5 * cube_idx + 1] = [
            vertex_ijk_to_idx(i + 0, j + 0, k + 0),
            vertex_ijk_to_idx(i + 1, j + 0, k + 0),
            vertex_ijk_to_idx(i + 0, j + 1, k + 0),
            vertex_ijk_to_idx(i + 0, j + 0, k + 1),
        ]
        element_to_vertex_index[5 * cube_idx + 2] = [
            vertex_ijk_to_idx(i + 1, j + 1, k + 0),
            vertex_ijk_to_idx(i + 0, j + 1, k + 0),
            vertex_ijk_to_idx(i + 1, j + 0, k + 0),
            vertex_ijk_to_idx(i + 1, j + 1, k + 1),
        ]
        element_to_vertex_index[5 * cube_idx + 3] = [
            vertex_ijk_to_idx(i + 1, j + 0, k + 1),
            vertex_ijk_to_idx(i + 0, j + 0, k + 1),
            vertex_ijk_to_idx(i + 1, j + 0, k + 0),
            vertex_ijk_to_idx(i + 1, j + 1, k + 1),
        ]
        element_to_vertex_index[5 * cube_idx + 4] = [
            vertex_ijk_to_idx(i + 0, j + 1, k + 1),
            vertex_ijk_to_idx(i + 0, j + 0, k + 1),
            vertex_ijk_to_idx(i + 0, j + 1, k + 0),
            vertex_ijk_to_idx(i + 1, j + 1, k + 1),
        ]


@ti.kernel
def init_mesh():
    for i in range(num_elements):
        vertex_idx = element_to_vertex_index[i]
        pos = [position[i] for i in vertex_idx]
        edges = [pos[i] - pos[0] for i in range(1, len(pos))]
        rest_shape = ti.Matrix.cols(edges)
        rest_shape_inv[i] = rest_shape.inverse()


@ti.kernel
def calc_energy():
    for i in range(num_elements):
        vertex_idx = element_to_vertex_index[i]
        pos = [position[i] for i in vertex_idx]
        edges = [pos[i] - pos[0] for i in range(1, len(pos))]
        volumns[i] = abs(edges[0].cross(edges[1]).dot(edges[2]))
        shape = ti.Matrix.cols(edges)

        F = shape @ rest_shape_inv[i]  # deformation gradient
        C = F.transpose() @ F  # right Cauchy-Green deformation tensor
        I_1 = C.trace()  # first invariance (trace) of `C`
        J = F.determinant()
        W = C_1 * (I_1 - 3 - 2 * ti.log(J)) + D_1 * ((J - 1) ** 2)

        stress_energy_density[i] = W
        total_energy[None] += stress_energy_density[i] * volumns[i]


@ti.kernel
def accelerate():
    for i in range(num_vertices):
        delta_mass = rho * volumns[i]
        acceleration = gravity * gravity_direction[None]
        acceleration -= position.grad[i] / delta_mass

        velocity[i] += acceleration * delta_t
        velocity[i] *= ti.exp(-damping * delta_t)

        position[i] += velocity[i] * delta_t


@ti.kernel
def boundary():
    # ball
    for i in range(num_vertices):
        vector = position[i] - ball_center[0]
        distance = vector.norm()
        direction = vector.normalized()
        if distance < ball_radius:
            normal_velocity = velocity[i].dot(direction)
            if normal_velocity < 0:
                velocity[i] -= (1 + ball_restitution) * normal_velocity * direction

    # ground
    for i in range(num_vertices):
        if position[i][2] < 0:
            normal_velocity = velocity[i][2]
            direction = ti.Vector([0, 0, 1])
            if normal_velocity < 0:
                velocity[i] -= (1 + ground_restitution) * normal_velocity * direction


def advance():
    accelerate()
    boundary()


def paint(scene: ti.ui.Scene):
    # ground
    scene.mesh(
        vertices=ground_mesh_vertices,
        indices=ground_mesh_indices,
        color=(100, 0, 0),
    )
    # ball
    scene.particles(centers=ball_center, radius=ball_radius, color=(0, 100, 0))
    # cuboid
    scene.particles(centers=position, radius=0.002, color=(0, 0, 100))


def main():
    gravity_direction[None] = ti.Vector([0, 0, -1])
    init_vertices()
    init_elements()
    init_mesh()

    pause: bool = False

    camera = ti.ui.Camera()
    scene = ti.ui.Scene()
    window = ti.ui.Window(name="3D FEM", res=(1080, 1080))

    while window.running:
        if window.get_event(ti.ui.PRESS):
            match window.event.key:
                case "q" | ti.ui.ESCAPE:
                    window.running = False
                case ti.ui.SPACE:
                    pause = not pause
                case "a" | ti.ui.LEFT:
                    gravity_direction[None] = ti.Vector([-1, 0, 0])
                case "s" | ti.ui.DOWN:
                    gravity_direction[None] = ti.Vector([0, -1, 0])
                case "d" | ti.ui.RIGHT:
                    gravity_direction[None] = ti.Vector([+1, 0, 0])
                case "w" | ti.ui.UP:
                    gravity_direction[None] = ti.Vector([0, +1, 0])
                case "j":
                    gravity_direction[None] = ti.Vector([0, 0, -1])
                case "k":
                    gravity_direction[None] = ti.Vector([0, 0, +1])
        camera.position(*camera_position)
        camera.lookat(*camera_lookat)
        camera.up(x=0, y=0, z=1)
        scene.set_camera(camera=camera)
        scene.point_light(pos=(5, 5, 5), color=(255, 255, 255))

        if not pause:
            with ti.ad.Tape(loss=total_energy):
                calc_energy()
            advance()
        paint(scene=scene)

        canvas = window.get_canvas()
        canvas.scene(scene=scene)

        window.show()


if __name__ == "__main__":
    main()
