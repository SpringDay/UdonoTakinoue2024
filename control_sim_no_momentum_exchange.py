import numpy as np
import matplotlib.pyplot as plt
# from mpl_axes_aligner import align
import matplotlib.ticker as ticker

# Parameters
a0 = 0.2  # uv-affected ratio
v_out = -0.4
v_in = -0.2
b = 0.8
lam0 = 0.2  # dropout in uv
kappa = 0.05 # dropout in vis
Cg = 0.4 # drag coefficient for ON state
Cgm = 0.1
dt = 0.01 # Time step
T = 400 # dt. 40 or 400 or 800 for none, PM, and JF
tau_uv_D = 800
f = 1 / (2 * T * dt) # switching frequency
t_max = 300  # Maximum time
n_steps = int(t_max / dt) + 1   # Number of steps
xmax = t_max
max_distance = 4.
initN = 100
dead = 30
ymin1, ymax1 = -3, 103
ymin2, ymax2 = -0.2, 0.3
ymin3, ymax3 = -35, 15
ax3_pos = 1.19
condensate_color = 'limegreen'
u_on_color = 'orange'
# m_D_color = 'darkorchid'
v_out_color = 'violet'
m_tot_color = 'crimson'
x_on_color = 'magenta'
# labels
ML_label = r'$m_{\rm L} (t)$'
MD_label = r'$m_{\rm D} (t)$'
# MT_label = r'$m_{\rm ALL} = m_{\rm ON} + m_{\rm OFF}$'
# MT_label = r'$m_{\rm ON} + m_{\rm OFF}$'
xlabel = r'Elapsed time, $t$ [-]'
ylabel = r'Mass for each state, $m_{\rm L} (t) $, $m_{\rm D} (t)$ [-]'
y2_label = r"Velocity for 'Liquid', $u (t) $ [-]"
y3_label = r"Position for 'Liquid', $x (t) $ [-]"
u_label = r'$u (t)$'
# v_label = r'$u_{\rm D}$'
x_label = r'$x (t)$'
fig_title = r'$T$ = ' + f'{T}' + r'$\Delta t$ (' + r'$f =$' + f'{f:.3f})' + r', ${\tau}^{\ast}$ = ' + f'{tau_uv_D}' + r'$\Delta t$ (fixed)'

# plot width, height
width, height = 7.5, 4.1 
fontsize = 15
legend_font = 11
title_font = 16
# out_file = 'rough_sim_JF_control.pdf'
out_file = 'rough_sim_PM_control.pdf'
# out_file = 'rough_sim_NONE.pdf'

# Initial conditions
m_L = np.zeros(n_steps)
m_D = np.zeros(n_steps)
# m_T = np.zeros(n_steps)
u = np.zeros(n_steps)
v = np.zeros(n_steps)
x = np.zeros(n_steps)
m_L[0] = initN  # Initial m_L
m_D[0] = 0  # Initial m_D
# m_T[0] = initN
u[0] = 0 # initial velocity for ON state
v[0] = 0 # that for OFF state. Sign inverted!!
x[0] = 0 # initial posotion

# Time array
t = np.linspace(0, t_max, n_steps)

# Numerical solution using Euler method
for i in range(1, n_steps):
    which = int(i / T)
    laptime = i - which * T
    # position update
    x[i] = x[i-1] + u[i-1] * dt

    # print(irradiation_steps)
    if which % 2 == 0: # UV
        L = lam0 * (laptime / tau_uv_D) ** 1.5
        A = a0 * (laptime / tau_uv_D) ** 1.5 # per unit time. Diffusion in condensate
        # mass exchange
        dmL = A * m_L[i-1] * dt
        m_L[i] = m_L[i-1] - dmL
        m_D[i] = m_D[i-1] + (1 - L) * dmL

        # velocity
        # momentum exchange origin
        duM_dt = - A * v_out
        # dissipation origin
        duG_dt = - Cg / m_L[i-1] * u[i-1]
        # u + du
        du = (duM_dt + duG_dt) * dt
        u[i] = u[i-1] + du
        
        # dispersed state
        v[i] = u[i-1] + v_out
    else: # Vis
        # mass exchange
        dmD = b * m_D[i-1] * dt
        m_L[i] = m_L[i-1] + (1 - kappa) * dmD
        m_D[i] = m_D[i-1] - dmD

        # veolocity
        # momentum exchange origin
        duM_dt = (1 - kappa) * b * m_D[i-1] * (v[i-1] + v_in - u[i-1]) / m_L[i-1]
        # dissipation origin
        duG_dt = - Cg * u[i-1] / m_L[i-1]
        # u + du
        # du = (duM_dt + duG_dt) * dt
        du = duG_dt * dt
        u[i] = u[i-1] + du

        # disperse state
        v[i] = v[i-1] - Cgm * v[i-1] * dt

# Plotting the results
fig, ax1 = plt.subplots(1, 1, figsize=(width, height))
ax2 = ax1.twinx()
ax3 = ax1.twinx()


p1, = ax1.plot(t, m_L, label=ML_label, color = condensate_color)
p1b, = ax1.plot(t, m_D, label=MD_label, color = condensate_color, linestyle='dashed')
# p1c,= ax1.plot(t, m_T, label=MT_label, color = m_tot_color, linestyle='dashdot')
ax1.set_ylabel(ylabel, fontsize=fontsize, labelpad=0, color=condensate_color)
ax1.set_xlabel(xlabel, fontsize=fontsize)
# ax1.legend(loc = 'upper center', fontsize=legend_font)
ax1.set_ylim(ymin1, ymax1)
ax1.set_xlim(0, xmax)
ax1.tick_params(axis='y', labelcolor=condensate_color)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

p2, = ax2.plot(t, u, label=u_label, color = u_on_color, linestyle='solid')
# ax2.plot(t, v, label=v_label, color = v_out_color, linestyle='dashed')
# ax2.set_ylabel('Velocity [-]', fontsize=fontsize)
# ax2.legend(loc='upper right', fontsize=legend_font)
# ax2.set(ylim = (ymin2, ymax2))
# ax2.set_ylabel(color=u_on_color, ylabel=y2_label, fontsize=fontsize)
ax2.set_ylabel(ylabel=y2_label, fontsize=fontsize, color=u_on_color)
ax2.tick_params(axis='y', labelcolor=u_on_color)
ax2.tick_params(axis='y', which='major', width=2)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

p3, = ax3.plot(t, x, label=x_label, color = x_on_color, linestyle='solid')
ax3.spines["right"].set_position(("axes", ax3_pos))
# ax3.legend(loc='center right', fontsize=legend_font)
# ax3.legend(loc='center right', fontsize=legend_font)
# ax3.set(ylim = (- max_distance, 1.))
# ax3.set(ylim = (0., max_distance + 1.))
# ax3.set(ylim = (0, 3.06), ylabel = 'Position')
# ax3.set(ylim = (ymin3, ymax3))
# ax3.set_ylabel(color = x_on_color, ylabel=y3_label, fontsize=fontsize)
ax3.set_ylabel(ylabel=y3_label, fontsize=fontsize, color=x_on_color)
ax3.tick_params(axis='y', labelcolor= x_on_color)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
ax3.tick_params(axis='y', which='major', width=2)

# align the plotting ranges of two yaxes
# org1 = 0
# org2 = 0
# pos = 0.05
# align.yaxes(ax1, org1, ax2, org2)

# ax1.legend(handles = [p1, p1b, p1c, p2, p3], loc='center right', fontsize=legend_font)
# plt.rcParams['legend.facecolor'] = 'white'
legend = plt.legend(handles = [p1, p1b, p2, p3], loc='center right', fontsize=legend_font).get_frame().set_alpha(0.95)
# legend.get_frame().set_facecolor('C0')



# plt.title(r'$F$ = ' + f'{f},' + r'$\tau$ = ' + f'{tau_uv_D}', fontsize=title_font)
plt.title(fig_title, fontsize=title_font)
# plt.legend(loc = 'upper right', fontsize=legend_font)
plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.grid(False)
# plt.show()
# plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(out_file, bbox_inches='tight', transparent=True)
