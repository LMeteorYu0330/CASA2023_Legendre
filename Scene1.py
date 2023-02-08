# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse
import vedo
import numpy as np
from vedo import Volume, show, Plotter
from vedo.applications import RayCastPlotter

import taichi as ti
import exporter

# How to run:
#   `python stable_fluid.py`: use the jacobi iteration to solve the linear system.
#   `python stable_fluid.py -S`: use a sparse matrix to do so.
# parser = argparse.ArgumentParser()
# parser.add_argument('-S',
#                     '--use-sp-mat',
#                     action='store_true',
#                     help='Solve Poisson\'s equation by using a sparse matrix')
# args, unknowns = parser.parse_known_args()

res = 64
dt = 0.03
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 12
gravity = True
debug = False
paused = False
# x_len = 1
# y_len = 1
# z_len = 1
# domain_length = 1
# dx = 1/res

legendre_n = 25
legendre_x_res = 64
legendre_dx = 2. / legendre_x_res
legendre_fit_iter = 1500
legendre_fit_learning_rate = 7e-6
frame = 0
target_frame = 300
ti.init(arch=ti.gpu, device_memory_fraction=0.9)
# ti.init(device_memory_GB=4)
print('Using jacobi iteration')

legendre_table = ti.field(float, shape=(legendre_n, legendre_x_res + 1))
alpha_x = ti.field(float, shape=(legendre_n, legendre_n, legendre_n))

rho_diff = ti.field(float, shape=(res, res, res))

_velocities = ti.Vector.field(3, float, shape=(res, res, res))
_new_velocities = ti.Vector.field(3, float, shape=(res, res, res))
velocity_divs = ti.field(float, shape=(res, res, res))
# velocity_curls = ti.field(3, shape=(res, res, res))
_pressures = ti.field(float, shape=(res, res, res))
_new_pressures = ti.field(float, shape=(res, res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res, res))
_density_color = ti.field(float, shape=(res, res, res))
current_legendre_density_color = ti.field(float, shape=(res, res, res))

src = ti.Vector([res / 2, 5, 5])
dir = ti.Vector([0, 0, 1])


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def sample(qf, u, v, w):
    I = ti.Vector([int(u), int(v), int(w)])
    I = max(0, min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v, w = p
    s, t, q = u - 0.5, v - 0.5, w - 0.5
    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(q)
    # fract
    fu, fv, fw = s - iu, t - iv, q - iw
    a = sample(vf, iu, iv, iw)
    b = sample(vf, iu + 1, iv, iw)
    c = sample(vf, iu, iv + 1, iw)
    d = sample(vf, iu + 1, iv + 1, iw)

    e = sample(vf, iu, iv, iw + 1)
    f = sample(vf, iu + 1, iv, iw + 1)
    g = sample(vf, iu, iv + 1, iw + 1)
    h = sample(vf, iu + 1, iv + 1, iw + 1)

    return lerp(lerp(lerp(a, b, fu), lerp(c, d, fu), fv), lerp(lerp(e, f, fu), lerp(g, h, fu), fv), fw)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j, k in vf:
        p = ti.Vector([i, j, k]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j, k] = bilerp(qf, p) * dye_decay


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template()):
    g_dir = -ti.Vector([0, -9.8, 0]) * 300
    for i, j, k in vf:
        omx, omy, omz = src
        mdir = dir
        dx, dy, dz = (i + 0.5 - omx), (j + 0.5 - omy), (k + 0.5 - omz)
        d2 = dx * dx + dy * dy + dz * dz
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j, k]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt

        v = vf[i, j, k]
        vf[i, j, k] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15) ** 2)) * ti.Vector(
                [0.8, 0.8, 0.8])

        dyef[i, j, k] = dc
        _density_color[i, j, k] = dc[0]


@ti.kernel
def divergence(vf: ti.template()):
    for i, j, k in vf:
        vl = sample(vf, i - 1, j, k)
        vr = sample(vf, i + 1, j, k)
        vb = sample(vf, i, j - 1, k)
        vt = sample(vf, i, j + 1, k)
        vc = sample(vf, i, j, k)
        vzf = sample(vf, i, j, k + 1)
        vzb = sample(vf, i, j, k - 1)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        if k == 0:
            vzb.z = -vc.z
        if k == res - 1:
            vzf.z = -vc.z

        velocity_divs[i, j, k] = (vr.x - vl.x + vt.y - vb.y + vzf.z - vzb.z) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j, k in pf:
        pl = sample(pf, i - 1, j, k)
        pr = sample(pf, i + 1, j, k)
        pb = sample(pf, i, j - 1, k)
        pt = sample(pf, i, j + 1, k)
        pzf = sample(pf, i, j, k + 1)
        pzb = sample(pf, i, j, k - 1)
        div = velocity_divs[i, j, k]
        new_pf[i, j, k] = (pl + pr + pb + pt + pzf + pzb - div) * (1 / 6)
        # print(new_pf[i, j, k], velocity_divs[i, j, k])


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j, k in vf:
        pl = sample(pf, i - 1, j, k)
        pr = sample(pf, i + 1, j, k)
        pb = sample(pf, i, j - 1, k)
        pt = sample(pf, i, j + 1, k)
        pzf = sample(pf, i, j, k + 1)
        pzb = sample(pf, i, j, k - 1)
        vf[i, j, k] -= 0.5 * ti.Vector([pr - pl, pt - pb, pzf - pzb])


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.template()):
    for I in ti.grouped(div_in):
        div_out[I[0] * res + I[1]] = -div_in[I]


@ti.kernel
def apply_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res + I[1]]


@ti.kernel
def fitting_alpha():
    for i, j, k in _density_color:
        rho = _density_color[i, j, k]
        rho_x = 0.0

        for m, n, o in ti.ndrange(legendre_n, legendre_n, legendre_n):
            PxPyPz = legendre_table[m, i] * legendre_table[n, j] * legendre_table[o, k]
            rho_x += alpha_x[m, n, o] * PxPyPz

        rho_diff[i, j, k] = rho - rho_x

    for m, n, o in ti.ndrange(legendre_n, legendre_n, legendre_n):
        m_step = 0.0

        for i, j, k in ti.ndrange(res, res, res):
            PxPyPz = legendre_table[m, i] * legendre_table[n, j] * legendre_table[o, k]
            m_step += PxPyPz * rho_diff[i, j, k]

        alpha_x[m, n, o] += legendre_fit_learning_rate * m_step


# @ti.kernel
# def calculateRMS() -> ti.float32:
#     rms = 0.0
#
#     for i, j, k in velocities_pair.cur:
#         ux, uy, uz = velocities_pair.cur[i, j, k]
#         u_prime_x = 0.0
#         u_prime_y = 0.0
#         u_prime_z = 0.0
#         for m, n, o in ti.ndrange(legendre_n, legendre_n, legendre_n):
#             PxPyPz = legendre_table[m, i] * legendre_table[n, j] * legendre_table[o, k]
#             u_prime_x += alpha_x[m, n, o] * PxPyPz
#             u_prime_y += alpha_y[m, n, o] * PxPyPz
#             u_prime_z += alpha_z[m, n, o] * PxPyPz
#         rms += (ux - u_prime_x) ** 2 + (uy - u_prime_y) ** 2 + (uz - u_prime_z) ** 2
#
#     return ti.sqrt(rms / (res ** 3))
@ti.kernel
def calculateRMSDensity() -> ti.float32:
    rms = 0.0

    for i, j, k in _density_color:
        rho = _density_color[i, j, k]
        rho_x = current_legendre_density_color[i, j, k]
        rms += (rho - rho_x) ** 2

    return ti.sqrt(rms / (res ** 3))


# @ti.kernel
# def printu_u_prime():
#     for i, j, k in velocities_pair.cur:
#         ux, uy, uz = velocities_pair.cur[i, j, k]
#         u_prime_x = 0.0
#         u_prime_y = 0.0
#         u_prime_z = 0.0
#         for m, n, o in ti.ndrange(legendre_n, legendre_n, legendre_n):
#             PxPyPz = legendre_table[m, i] * legendre_table[n, j] * legendre_table[o, k]
#             u_prime_x += alpha_x[m, n, o] * PxPyPz
#             u_prime_y += alpha_y[m, n, o] * PxPyPz
#             u_prime_z += alpha_z[m, n, o] * PxPyPz
#         print("u: ", ux, uy, uz, "u_prime: ", u_prime_x, u_prime_y, u_prime_z)


# @ti.kernel
# def replace_velocity_field():
#     for i, j, k in velocities_pair.cur:
#         u_prime_x = 0.0
#         u_prime_y = 0.0
#         u_prime_z = 0.0
#         for m, n, o in ti.ndrange(legendre_n, legendre_n, legendre_n):
#             PxPyPz = legendre_table[m, i] * legendre_table[n, j] * legendre_table[o, k]
#             u_prime_x += alpha_x[m, n, o] * PxPyPz
#             u_prime_y += alpha_y[m, n, o] * PxPyPz
#             u_prime_z += alpha_z[m, n, o] * PxPyPz
#         velocities_pair.cur[i, j, k] = ti.Vector([u_prime_x, u_prime_y, u_prime_z])


def calculateLegendrePolynomial():
    for n in range(legendre_n):
        for x in range(legendre_x_res + 1):
            x_ = -1 + x * legendre_dx
            if n == 0:
                legendre_table[n, x] = 1
            elif n == 1:
                legendre_table[n, x] = x_
            else:
                legendre_table[n, x] = ((2 * n - 1) * x_ * legendre_table[n - 1, x] - (n - 1) * legendre_table[
                    n - 2, x]) / n


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()


def step():
    # replace_velocity_field()
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)

    velocities_pair.swap()

    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur)

    divergence(velocities_pair.cur)

    solve_pressure_jacobi()
    subtract_gradient(velocities_pair.cur, pressures_pair.cur)


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    alpha_x.fill(0)
    # alpha_y.fill(0)
    # alpha_z.fill(0)
    # grad_alpha_x.fill(0)
    # grad_alpha_y.fill(0)
    # grad_alpha_z.fill(0)
    legendre_table.fill(0)
    dyes_pair.cur.fill(0)
    _density_color.fill(0)
    current_legendre_density_color.fill(0)


@ti.kernel
def make_legendre_density():
    for i, j, k in current_legendre_density_color:
        current_legendre_density_color[i, j, k] = 0.0
        for m, n, o in ti.ndrange(legendre_n, legendre_n, legendre_n):
            PxPyPz = legendre_table[m, i] * legendre_table[n, j] * legendre_table[o, k]
            current_legendre_density_color[i, j, k] += alpha_x[m, n, o] * PxPyPz


# def fillnumpy():
#     arr = vol.tonumpy()
#     arr[:] = _density_color.to_numpy()
#     print("fill")
#     vol.mode(0).c('cool').alpha(0.02)
#     vol.imagedata().GetPointData().GetScalars().Modified()

def step_one_frame(evt):
    global frame
    print("step: ", frame)
    step()
    ti.sync()
    # fillnumpy()
    for iter in range(legendre_fit_iter):
        fitting_alpha()
    print("alpha fitted")
    make_legendre_density()
    print(calculateRMSDensity())
    exporter.makeVol(res, _density_color.to_numpy(),
                     "Z:\TeamFolder\CASA2023_Legendre\shared\Scene1\Lground_truth\Scene1_" + str(
                         frame).zfill(4) + ".vol")
    exporter.makeVol(res, current_legendre_density_color.to_numpy(),
                     "Z:\TeamFolder\CASA2023_Legendre\shared\Scene1\Legendre_volume_sequence\Scene1_" + str(
                         frame).zfill(4) + ".vol")
    frame += 1


reset()
calculateLegendrePolynomial()
# vol = Volume(np.zeros_like(_density_color.to_numpy())).mode(0).c('cool').alpha(0.02)  # change properties
# plt = RayCastPlotter(vol, bg='white', bg2='blackboard', axes=7)  # Plotter instance
# plt.add_callback("KeyPress", step_one_frame)
# fillnumpy()

# plt.show(viewup="z")
# plt.interactive().close()

for i in range(target_frame):
    step_one_frame(0)

# printu_u_prime()
# print(calculateRMS())
