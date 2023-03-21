import typing
from pathlib import Path

import numpy as np
import pyvista as pv
import taichi as ti
import tetgen
import typer
from taichi.lang import impl

ti.init(default_fp=ti.f64, fast_math=True)


def faces_to_indices(faces: np.ndarray) -> np.ndarray:
    indices = list()
    i: int = 0
    while i < len(faces):
        for j in range(i + 1, i + 1 + faces[i]):
            indices.append(faces[j])
        i += faces[i] + 1
    return np.array(indices)


# physical constants
sphere_restitution: float = 0.8  # coefficient of restitution
ground_restitution: float = 0.8  # coefficient of restitution
delta_t: ti.ScalarField = ti.field(dtype=float, shape=())
E, nu = 3e4, 0.2  # Young's modulus, Poisson's ratio
gravity_direction: ti.MatrixField = ti.Vector.field(n=3, dtype=float, shape=())
gravity_direction[None] = [0, 0, -1]
gravity: float = 9.8
rho: float = 1e3  # mass density


mu = E / 2 / (1 + nu)  # shear modulus
lambda_L = E * nu / (1 + nu) / (1 - 2 * nu)  # Lame's first parameter, lambda
C_1 = mu / 2  # Neo-Hookean solid
D_1 = lambda_L / 2  # Neo-Hookean solid
damping = 10


# camera
camera_position = ti.Vector([0, -2, 2])
camera_lookat = ti.Vector([0, 0, 0])


# ground
ground = typing.cast(pv.PolyData, typing.cast(pv.PolyData, pv.Plane()).triangulate())
ground_vertices: ti.MatrixField = ti.Vector.field(
    n=3, dtype=ti.f32, shape=ground.n_points
)
ground_faces: ti.ScalarField = ti.field(dtype=int, shape=3 * ground.n_faces)
ground_vertices.from_numpy(ground.points)
ground_faces.from_numpy(faces_to_indices(ground.faces))


# sphere
sphere_center = ti.Vector([0, 0, 0])
sphere_radius = 0.2
sphere = typing.cast(
    pv.PolyData,
    typing.cast(
        pv.PolyData, pv.Sphere(radius=sphere_radius, center=sphere_center)
    ).triangulate(),
)
sphere_vertices: ti.MatrixField = ti.Vector.field(
    n=3, dtype=ti.f32, shape=sphere.n_points
)
sphere_faces: ti.ScalarField = ti.field(dtype=int, shape=3 * sphere.n_faces)
sphere_vertices.from_numpy(sphere.points)
sphere_faces.from_numpy(faces_to_indices(sphere.faces))


# mesh
mesh_origin: ti.Vector = ti.Vector([0, 0, 0.5])
num_vertices: int
num_tetras: int
vertices: ti.MatrixField
tetras: ti.MatrixField

energy_density: ti.ScalarField
mass: ti.ScalarField
position_f32: ti.MatrixField
position: ti.MatrixField
rest_shape_inv: ti.MatrixField
rotation: ti.MatrixField
total_energy: ti.ScalarField
velocity: ti.MatrixField
vertex_mess: ti.ScalarField
volume: ti.ScalarField


@ti.func
def get_shape(tetra_idx: int) -> ti.math.mat3:
    position_idx = tetras[tetra_idx]
    v = [position[j] for j in position_idx]
    edge = [v[j] - v[0] for j in range(1, len(v))]
    return ti.Matrix.cols(edge)


@ti.kernel
def init_fields():
    for i in range(num_vertices):
        position[i] = mesh_origin + vertices[i]

    total_mass = 0.0
    for i in range(num_tetras):
        shape = get_shape(i)
        rest_shape_inv[i] = ti.math.inverse(shape)
        mass[i] = rho * (1.0 / 6.0) * ti.abs(ti.Matrix.determinant(shape))
        total_mass += mass[i]

    for i in range(num_vertices):
        vertex_mess[i] = total_mass / num_vertices


def init(mesh_filepath: str | Path) -> None:
    global num_vertices, num_tetras, vertices, tetras, energy_density, position, position_f32, rest_shape_inv, rotation, mass, total_energy, velocity, vertex_mess, volume

    # mesh = typing.cast(pv.PolyData, pv.read(filename=mesh_filepath))
    mesh = typing.cast(pv.PolyData, pv.Box(bounds=(0, 0.2, -0.2, 0, 0, 0.2), level=8))
    # mesh = typing.cast(pv.PolyData, pv.Icosahedron(radius=0.2))
    mesh = typing.cast(pv.PolyData, mesh.triangulate())
    tet = tetgen.TetGen(mesh)
    tet.make_manifold(verbose=True)
    vertices_numpy, tetras_numpy = tet.tetrahedralize(
        minratio=1.1,
        mindihedral=10,
        verbose=2,
        nobisect=True,
        order=1,
    )
    num_vertices, num_tetras = len(vertices_numpy), len(tetras_numpy)
    vertices = ti.Vector.field(n=3, dtype=float, shape=num_vertices)
    tetras = ti.Vector.field(n=4, dtype=int, shape=num_tetras)
    vertices.from_numpy(np.asarray(vertices_numpy, dtype=np.float32))
    tetras.from_numpy(tetras_numpy)

    energy_density = ti.field(dtype=float, shape=num_tetras, needs_grad=True)
    mass = ti.field(dtype=float, shape=num_tetras)
    position = ti.Vector.field(n=3, dtype=float, shape=num_vertices, needs_grad=True)
    position_f32 = ti.Vector.field(n=3, dtype=ti.f32, shape=num_vertices)
    rest_shape_inv = ti.Matrix.field(n=3, m=3, dtype=float, shape=num_tetras)
    rotation = ti.Matrix.field(
        n=3, m=3, dtype=float, shape=num_tetras, needs_grad=False
    )
    total_energy = ti.field(dtype=float, shape=(), needs_grad=True)
    velocity = ti.Vector.field(n=3, dtype=float, shape=num_vertices)
    vertex_mess = ti.field(dtype=float, shape=num_vertices)
    volume = ti.field(dtype=float, shape=num_tetras, needs_grad=False)

    init_fields()


@ti.func
def svd3d(A, dt, iters=8):
    assert A.n == 3 and A.m == 3
    # assert dt in [ti.f32, ti.f64]
    # if iters is None:
    rets = (
        impl.get_runtime()
        .compiling_callable.ast_builder()
        .sifakis_svd_f64(A.ptr, iters)
    )
    assert len(rets) == 21
    U_entries = rets[:9]
    V_entries = rets[9:18]
    sig_entries = rets[18:]

    U = ti.Matrix.zero(dt, 3, 3)
    V = ti.Matrix.zero(dt, 3, 3)
    sigma = ti.Matrix.zero(dt, 3, 3)
    U[0, 0] = U_entries[0]
    U[0, 1] = U_entries[1]
    U[0, 2] = U_entries[2]
    U[1, 0] = U_entries[3]
    U[1, 1] = U_entries[4]
    U[1, 2] = U_entries[5]
    U[2, 0] = U_entries[6]
    U[2, 1] = U_entries[7]
    U[2, 2] = U_entries[8]
    V[0, 0] = V_entries[0]
    V[0, 1] = V_entries[1]
    V[0, 2] = V_entries[2]
    V[1, 0] = V_entries[3]
    V[1, 1] = V_entries[4]
    V[1, 2] = V_entries[5]
    V[2, 0] = V_entries[6]
    V[2, 1] = V_entries[7]
    V[2, 2] = V_entries[8]
    sigma[0, 0] = sig_entries[0]
    sigma[1, 1] = sig_entries[1]
    sigma[2, 2] = sig_entries[2]
    return U, sigma, V


@ti.func
def polar_decompose3d(A: ti.math.mat3, dt=ti.f64):
    U, sig, V = ti.svd(A, dt)
    return U @ V.transpose(), V @ sig @ V.transpose()


@ti.ad.no_grad
@ti.kernel
def compute_rotation():
    for i in range(num_tetras):
        shape = get_shape(i)
        F = shape @ rest_shape_inv[i]  # deformation gradient
        U, P = ti.polar_decompose(F)
        rotation[i] = P


@ti.kernel
def compute_energy():
    for i in range(num_tetras):
        shape = get_shape(i)
        # print(i, "shape", shape, rest_shape_inv[i].inverse())
        volume[i] = (1.0 / 6.0) * ti.abs(shape.determinant())

        F = shape @ rest_shape_inv[i]  # deformation gradient

        # Hyperelastic material
        # material displacement gradient tensor
        # grad_u = F @ rotation[i].transpose() - ti.math.eye(n=3)
        # E = 0.5 * (
        #     grad_u.transpose() + grad_u + grad_u.transpose() @ grad_u
        # )  # Lagrangian Green strain
        # W = 0.5 * lambda_L * (E.trace() ** 2) + mu * (E @ E).trace()

        # Neo-Hookean solid
        C = F.transpose() @ F  # right Cauchy-Green deformation tensor
        I_1 = C.trace()  # first invariance (trace) of `C`
        J = F.determinant()
        W = C_1 * (I_1 - 3.0 - 2.0 * ti.log(J)) + D_1 * ((J - 1.0) ** 2)

        # Neo-Hookean solid
        # C = F.transpose() @ F  # right Cauchy-Green deformation tensor
        # I_1 = C.trace()  # first invariance (trace) of `C`
        # J = F.determinant()
        # # the first invariant of the isochoric part of the right Cauchy–Green deformation tensor
        # I_1_bar = ti.math.pow(J, -2.0 / 3.0) / I_1
        # W = C_1 * (I_1_bar - 3) + (C_1 / 6 + D_1 / 4) * (J**2 + 1 / (J**2) - 2.0)

        energy_density[i] = W
        # if ti.math.isnan(energy_density[i]):
        #     energy_density[i] = (shape - rest_shape_inv[i].inverse()).norm()
        # print(i, "F", F)

        total_energy[None] += energy_density[i] * volume[i]


@ti.kernel
def compute_acceleration():
    for i in range(num_vertices):
        acceleration = gravity * gravity_direction[None]
        if not ti.math.isnan(position.grad[i]).any():
            acceleration -= position.grad[i] / vertex_mess[i]
        # print(i, "grad", position.grad[i])
        velocity[i] += acceleration * delta_t[None]


@ti.kernel
def compute_collision():
    # sphere
    for i in range(num_vertices):
        vector = position[i] - sphere_center[0]
        distance = vector.norm()
        direction = vector.normalized()
        if distance < sphere_radius:
            position[i] = sphere_center[0] + sphere_radius * direction
            normal_velocity = velocity[i].dot(direction)
            if normal_velocity < 0.0:
                velocity[i] -= (1.0 + sphere_restitution) * normal_velocity * direction

    # ground
    for i in range(num_vertices):
        if position[i][2] < 1e-3:
            position[i][2] = 1e-3
            normal_velocity = velocity[i][2]
            direction = ti.Vector([0, 0, 1])
            if normal_velocity < 0.0:
                velocity[i] -= (1.0 + ground_restitution) * normal_velocity * direction


@ti.kernel
def compute_damping():
    for i in range(num_vertices):
        velocity[i] *= ti.exp(-damping * delta_t[None])


@ti.kernel
def compute_displacement():
    for i in range(num_vertices):
        position[i] += velocity[i] * delta_t[None]
        position_f32[i] = ti.cast(position[i], ti.f32)


def advance():
    compute_acceleration()
    compute_collision()
    compute_damping()
    compute_displacement()


def paint(scene: ti.ui.Scene):
    scene.mesh(vertices=ground_vertices, indices=ground_faces)
    scene.mesh(vertices=sphere_vertices, indices=sphere_faces, color=(1, 0, 0))
    scene.particles(centers=position_f32, radius=1e-3, color=(0, 1, 0))
    # v = ti.Vector.field(n=3, dtype=float, shape=4)
    # f = ti.field(dtype=int, shape=3 * 4)
    # idx = tetras[1]
    # v[0], v[1], v[2], v[3] = (position[i] for i in idx)
    # f.from_numpy(np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]))
    # scene.mesh(vertices=v, indices=f, color=(0, 0, 1))


def main(
    mesh_filepath: str = typer.Option(
        default=Path.cwd() / "data" / "template-closed.obj",
    ),
    _delta_t: float = typer.Option(1e-3, "--delta-t"),
    show_window: bool = typer.Option(default=False),
) -> None:
    delta_t[None] = _delta_t

    init(mesh_filepath=mesh_filepath)

    camera = ti.ui.Camera()
    scene = ti.ui.Scene()
    window = ti.ui.Window(name="FEM", res=(1080, 1080), show_window=show_window)

    camera.lookat(*camera_lookat)
    camera.position(*camera_position)
    camera.up(x=0, y=0, z=1)
    scene.set_camera(camera=camera)

    while window.running:
        with ti.ad.Tape(loss=total_energy):
            compute_rotation()
            compute_energy()
        advance()

        paint(scene=scene)

        scene.ambient_light(color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(5, -5, 5), color=(1, 1, 1))

        canvas = window.get_canvas()
        canvas.scene(scene=scene)

        window.show()


if __name__ == "__main__":
    typer.run(main)
