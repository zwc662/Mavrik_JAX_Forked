import pytest
from jax_mavrik.src.mavrik_aero import MavrikAero

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

expected_actuator_outputs_values = jnp.array([
    [30.000000, 0.069813, 0.000000, 0.000000, 0.000000, 0.000000, 0.069813, 0.000000, 7500.000000, 0.069813, 0.069813, 0.000000, 0.000000, 0.011636, 0.000000, 0.069813, 0.000000, 7500.000000, 0.069813, 0.069813, 0.000000, 0.000000, 0.069813, 0.000000, 551.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.956949, 0.065698, 0.000038, -0.007684, -0.154990, 0.000185, 0.065698, 0.000038, 7500.000000, 0.065698, 0.065698, 0.000038, 0.000038, 0.010950, 0.000006, 0.065698, 0.000038, 7500.000000, 0.065698, 0.065698, 0.000038, 0.000038, 0.065698, 0.000038, 549.669018, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000], 
    [29.914949, 0.060331, 0.000153, -0.015420, -0.297333, 0.000410, 0.060331, 0.000153, 7500.000000, 0.060331, 0.060331, 0.000153, 0.000153, 0.010055, 0.000025, 0.060331, 0.000153, 7500.000000, 0.060331, 0.060331, 0.000153, 0.000153, 0.060331, 0.000153, 548.128813, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.874238, 0.053863, 0.000344, -0.023234, -0.426201, 0.000706, 0.053863, 0.000344, 7500.000000, 0.053863, 0.053863, 0.000344, 0.000344, 0.008977, 0.000057, 0.053863, 0.000344, 7500.000000, 0.053863, 0.053863, 0.000344, 0.000344, 0.053863, 0.000344, 546.637949, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.834992, 0.046450, 0.000612, -0.031147, -0.540992, 0.001099, 0.046450, 0.000612, 7500.000000, 0.046450, 0.046450, 0.000612, 0.000612, 0.007742, 0.000102, 0.046450, 0.000612, 7500.000000, 0.046450, 0.046450, 0.000612, 0.000612, 0.046450, 0.000612, 545.202647, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.797323, 0.038246, 0.000957, -0.039181, -0.641322, 0.001618, 0.038246, 0.000957, 7500.000000, 0.038246, 0.038246, 0.000957, 0.000957, 0.006374, 0.000159, 0.038246, 0.000957, 7500.000000, 0.038246, 0.038246, 0.000957, 0.000957, 0.038246, 0.000957, 543.826784, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.761282, 0.029406, 0.001380, -0.047358, -0.727013, 0.002291, 0.029406, 0.001380, 7500.000000, 0.029406, 0.029406, 0.001380, 0.001380, 0.004901, 0.000230, 0.029406, 0.001380, 7500.000000, 0.029406, 0.029406, 0.001380, 0.001380, 0.029406, 0.001380, 542.512006, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.726863, 0.020080, 0.001883, -0.055700, -0.798073, 0.003143, 0.020080, 0.001883, 7500.000000, 0.020080, 0.020080, 0.001883, 0.001883, 0.003347, 0.000314, 0.020080, 0.001883, 7500.000000, 0.020080, 0.020080, 0.001883, 0.001883, 0.020080, 0.001883, 541.257922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.694014, 0.010414, 0.002466, -0.064231, -0.854687, 0.004201, 0.010414, 0.002466, 7500.000000, 0.010414, 0.010414, 0.002466, 0.002466, 0.001736, 0.000411, 0.010414, 0.002466, 7500.000000, 0.010414, 0.010414, 0.002466, 0.002466, 0.010414, 0.002466, 540.062372, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.662640, 0.000549, 0.003131, -0.072976, -0.897200, 0.005492, 0.000549, 0.003131, 7500.000000, 0.000549, 0.000549, 0.003131, 0.003131, 0.000091, 0.000522, 0.000549, 0.003131, 7500.000000, 0.000549, 0.000549, 0.003131, 0.003131, 0.000549, 0.003131, 538.921725, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.632640, -0.009384, 0.003881, -0.081987, -0.926122, 0.007043, -0.009384, 0.003881, 7500.000000, -0.009384, -0.009384, 0.003881, 0.003881, -0.001564, 0.000647, -0.009384, 0.003881, 7500.000000, -0.009384, -0.009384, 0.003881, 0.003881, -0.009384, 0.003881, 537.832186, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
])

expected_Cx_outputs_values = jnp.array([
    [-114.295761551278, 0, 0],
    [-114.449641091491, 0, 0],
    [-114.163852031687, 0, 0],
    [-113.441654414852, 0, 0],
    [-112.303261884019, 0, 0],
    [-110.78348892127, 0, 0],
    [-108.929021611756, 0, 0],
    [-106.795484478394, 0, 0],
    [-104.444458270136, 0, 0],
    [-101.940582730151, 0, 0],
    [-99.3547737877012, 0, 0]
])

expected_Cy_outputs_values = jnp.array([
        [0, 1.54656473174905e-15, 0],
        [0, -0.0160390391484072, 0],
        [0, -0.0611958036837794, 0],
        [0, -0.134389887735588, 0],
        [0, -0.234701364158825, 0],
        [0, -0.361370119624134, 0],
        [0, -0.513797036084389, 0],
        [0, -0.691546661577133, 0],
        [0, -0.894350936998841, 0],
        [0, -1.12211352186119, 0],
        [0, -1.3749283426252, 0]
    ]
)

expected_Cz_outputs_values = jnp.array([
        [0, 0, -264.893059183493],
        [0, 0, -247.330317583035],
        [0, 0, -227.927496579797],
        [0, 0, -207.063069903171],
        [0, 0, -185.102372839729],
        [0, 0, -162.394413922597],
        [0, 0, -139.269299081639],
        [0, 0, -116.036178507391],
        [0, 0, -92.9816327558558],
        [0, 0, -70.3684267025321],
        [0, 0, -48.4869242664429]
    ]
)

expected_Cl_outputs_values = jnp.array([
    [-3.71891712080154, 0, 0],
    [-3.73892743798086, 0, 0],
    [-3.770934313081, 0, 0],
    [-3.81434583871129, 0, 0],
    [-3.86884630584595, 0, 0],
    [-3.93440170934839, 0, 0],
    [-4.01125413670425, 0, 0],
    [-4.09990704443226, 0, 0],
    [-4.20110335307348, 0, 0],
    [-4.31579817794609, 0, 0],
    [-4.47116516425064, 0, 0]
    ]
)

expected_Cm_outputs_values = jnp.array([
        [0, -97.4271038597041, 0],
        [0, -90.0901400268625, 0],
        [0, -82.1812570744447, 0],
        [0, -73.8415402372756, 0],
        [0, -65.205648840758, 0],
        [0, -56.4007941861578, 0],
        [0, -47.5459473602478, 0],
        [0, -38.7512399003046, 0],
        [0, -30.1175236238111, 0],
        [0, -21.7360612793007, 0],
        [0, -13.7178408886391, 0]
    ]
)

expected_Cn_outputs_values = jnp.array([
        [0, 0, 0.164977625437221],
        [0, 0, 0.191046199997934],
        [0, 0, 0.247575681429308],
        [0, 0, 0.333546638829987],
        [0, 0, 0.44799062413523],
        [0, 0, 0.59000322610442],
        [0, 0, 0.758765968959369],
        [0, 0, 0.953574138754955],
        [0, 0, 1.1738675465473],
        [0, 0, 1.41926139470376],
        [0, 0, 1.69101351913533]
        ])

expected_Ct_outputs_values = jnp.array([
    [23.8478523411019, 0, 0, 0, 0.210084930483265, 1.77635683940025e-15],
    [24.0653722230781, 0, 0, 0, 0.212208064150277, 0],
    [24.267708349065, 0, 0, 0, 0.214335340663162, 4.44089209850063e-16],
    [24.4546794297957, 0, 0, 0, 0.216449502182345, 1.33226762955019e-15],
    [24.6264819613241, 0, 0, 0, 0.218536062825545, -4.44089209850063e-16],
    [24.7836771443068, 0, 0, 0, 0.220583408644189, -8.88178419700125e-16],
    [24.9271621051189, 0, 0, 0, 0.222582756561505, 0],
    [25.058128498218, 0, 0, 0, 0.22452799637347, 4.44089209850063e-16],
    [25.1780116995042, 0, 0, 0, 0.226415441147283, -4.44089209850063e-16],
    [25.2884337985374, 0, 0, 0, 0.228243511444974, -4.44089209850063e-16],
    [25.5345143902457, 0, 0, 0, 0.229192307180639, 0]
])

expected_Kq_outputs_values = jnp.zeros((11, 3))
    
vned_values = jnp.array([
    [30.0000, 0, 0], [29.9568, -0.0000, -0.0997], [29.9143, -0.0000, -0.1922],
    [29.8730, -0.0001, -0.2767], [29.8329, -0.0002, -0.3529], [29.7944, -0.0004, -0.4203],
    [29.7574, -0.0006, -0.4789], [29.7222, -0.0009, -0.5286], [29.6886, -0.0014, -0.5692],
    [29.6565, -0.0019, -0.6011], [29.6261, -0.0024, -0.6243]
])
xned_values = jnp.array([
    [0, 0, 0], [0.2998, -0.0000, -0.0005], [0.5991, -0.0000, -0.0020],
    [0.8981, -0.0000, -0.0043], [1.1966, -0.0000, -0.0075], [1.4947, -0.0000, -0.0114],
    [1.7925, -0.0000, -0.0159], [2.0899, -0.0000, -0.0209], [2.3869, -0.0000, -0.0264],
    [2.6837, -0.0000, -0.0323], [2.9801, -0.0001, -0.0384]
])
euler_values = jnp.array([
    [0, 0.0698, 0], [-0.0000, 0.0690, 0.0000], [-0.0002, 0.0668, 0.0000],
    [-0.0003, 0.0631, 0.0000], [-0.0006, 0.0583, 0.0000], [-0.0010, 0.0524, 0.0000],
    [-0.0014, 0.0455, 0.0001], [-0.0019, 0.0379, 0.0001], [-0.0025, 0.0296, 0.0002],
    [-0.0032, 0.0208, 0.0002], [-0.0040, 0.0117, 0.0003]
])
vb_values = jnp.array([
    [29.9269, 0, 2.0927], [29.8923, -0.0001, 1.9667], [29.8605, -0.0004, 1.8037],
    [29.8309, -0.0010, 1.6083], [29.8028, -0.0017, 1.3853], [29.7755, -0.0027, 1.1393],
    [29.7484, -0.0038, 0.8750], [29.7209, -0.0053, 0.5969], [29.6924, -0.0069, 0.3092],
    [29.6626, -0.0089, 0.0163], [29.6313, -0.0111, -0.2781]
])
ab_values = jnp.array([
    [-3.6179, 0, -10.5957], [-3.3106, -0.0213, -14.5262], [-3.0595, -0.0425, -17.9956],
    [-2.8740, -0.0638, -20.9965], [-2.7576, -0.0853, -23.5272], [-2.7093, -0.1073, -25.5916],
    [-2.7239, -0.1301, -27.1984], [-2.7932, -0.1543, -28.3612], [-2.9064, -0.1804, -29.0974],
    [-3.0515, -0.2090, -29.4287], [-3.2104, -0.2409, -29.3826]
])
expected_forces_values = jnp.array([
    [-90.4479, 0, -264.8931], [-90.3843, -0.0160, -247.3303], [-89.8961, -0.0612, -227.9275],
    [-88.9870, -0.1344, -207.0631], [-87.6768, -0.2347, -185.1024], [-85.9998, -0.3614, -162.3944],
    [-84.0019, -0.5138, -139.2693], [-81.7374, -0.6915, -116.0362], [-79.2664, -0.8944, -92.9816],
    [-76.6521, -1.1221, -70.3684], [-73.8203, -1.3749, -48.4869]
])
expected_moments_values = jnp.array([
    [-3.7189, -97.2170, 0.1650], [-3.7389, -89.8779, 0.1910], [-3.7709, -81.9669, 0.2476],
    [-3.8143, -73.6251, 0.3335], [-3.8688, -64.9871, 0.4480], [-3.9344, -56.1802, 0.5900],
    [-4.0113, -47.3234, 0.7588], [-4.0999, -38.5267, 0.9536], [-4.2011, -29.8911, 1.1739],
    [-4.3158, -21.5078, 1.4193], [-4.4712, -13.4886, 1.6910]
])



assert expected_actuator_outputs_values.shape == (11, 45)
 
@pytest.fixture
def mavrik_aero():
    mavrik_setup = MavrikSetup(file_path="/Users/weichaozhou/Workspace/Mavrik_JAX/jax_mavrik/aero_export.mat")
    return MavrikAero(mavrik_setup=mavrik_setup)

@pytest.fixture
def control_inputs():
    return ControlInputs(
        wing_tilt=0.0, tail_tilt=0.0, aileron=0.0,
        elevator=0.0, flap=0.0, rudder=0.0,
        RPM_tailLeft=7500, RPM_tailRight=7500,
        RPM_leftOut1=7500, RPM_left2=7500,
        RPM_left3=7500, RPM_left4=7500,
        RPM_left5=7500, RPM_left6In=7500,
        RPM_right7In=7500, RPM_right8=7500,
        RPM_right9=7500, RPM_right10=7500,
        RPM_right11=7500, RPM_right12Out=7500
    )

'''
@pytest.mark.parametrize("vned, xned, euler, vb, ab, expected_forces, expected_moments, expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values, expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values", [
    (jnp.array([30.0000, 0, 0]), jnp.array([0, 0, 0]), jnp.array([0, 0.0698, 0]), jnp.array([29.9269, 0, 2.0927]), jnp.array([-3.6179, 0, -10.5957]), jnp.array([-90.4479, 0, -264.8931]), jnp.array([-3.7189, -97.2170, 0.1650]), expected_actuator_outputs[0]),
    (jnp.array([29.9568, -0.0000, -0.0997]), jnp.array([0.2998, -0.0000, -0.0005]), jnp.array([-0.0000, 0.0690, 0.0000]), jnp.array([29.8923, -0.0001, 1.9667]), jnp.array([-3.3106, -0.0213, -14.5262]), jnp.array([-90.3843, -0.0160, -247.3303]), jnp.array([-3.7389, -89.8779, 0.1910]), expected_actuator_outputs[1]),
    (jnp.array([29.9143, -0.0000, -0.1922]), jnp.array([0.5991, -0.0000, -0.0020]), jnp.array([-0.0002, 0.0668, 0.0000]), jnp.array([29.8605, -0.0004, 1.8037]), jnp.array([-3.0595, -0.0425, -17.9956]), jnp.array([-89.8961, -0.0612, -227.9275]), jnp.array([-3.7709, -81.9669, 0.2476]), expected_actuator_outputs[2]),
    (jnp.array([29.8730, -0.0001, -0.2767]), jnp.array([0.8981, -0.0000, -0.0043]), jnp.array([-0.0003, 0.0631, 0.0000]), jnp.array([29.8309, -0.0010, 1.6083]), jnp.array([-2.8740, -0.0638, -20.9965]), jnp.array([-88.9870, -0.1344, -207.0631]), jnp.array([-3.8143, -73.6251, 0.3335]), expected_actuator_outputs[3]),
    (jnp.array([29.8329, -0.0002, -0.3529]), jnp.array([1.1966, -0.0000, -0.0075]), jnp.array([-0.0006, 0.0583, 0.0000]), jnp.array([29.8028, -0.0017, 1.3853]), jnp.array([-2.7576, -0.0853, -23.5272]), jnp.array([-87.6768, -0.2347, -185.1024]), jnp.array([-3.8688, -64.9871, 0.4480]), expected_actuator_outputs[4]),
    (jnp.array([29.7944, -0.0004, -0.4203]), jnp.array([1.4947, -0.0000, -0.0114]), jnp.array([-0.0010, 0.0524, 0.0000]), jnp.array([29.7755, -0.0027, 1.1393]), jnp.array([-2.7093, -0.1073, -25.5916]), jnp.array([-85.9998, -0.3614, -162.3944]), jnp.array([-3.9344, -56.1802, 0.5900]), expected_actuator_outputs[5]),
    (jnp.array([29.7574, -0.0006, -0.4789]), jnp.array([1.7925, -0.0000, -0.0159]), jnp.array([-0.0014, 0.0455, 0.0001]), jnp.array([29.7484, -0.0038, 0.8750]), jnp.array([-2.7239, -0.1301, -27.1984]), jnp.array([-84.0019, -0.5138, -139.2693]), jnp.array([-4.0113, -47.3234, 0.7588]), expected_actuator_outputs[6]),
    (jnp.array([29.7222, -0.0009, -0.5286]), jnp.array([2.0899, -0.0000, -0.0209]), jnp.array([-0.0019, 0.0379, 0.0001]), jnp.array([29.7209, -0.0053, 0.5969]), jnp.array([-2.7932, -0.1543, -28.3612]), jnp.array([-81.7374, -0.6915, -116.0362]), jnp.array([-4.0999, -38.5267, 0.9536]), expected_actuator_outputs[7]),
    (jnp.array([29.6886, -0.0014, -0.5692]), jnp.array([2.3869, -0.0000, -0.0264]), jnp.array([-0.0025, 0.0296, 0.0002]), jnp.array([29.6924, -0.0069, 0.3092]), jnp.array([-2.9064, -0.1804, -29.0974]), jnp.array([-79.2664, -0.8944, -92.9816]), jnp.array([-4.2011, -29.8911, 1.1739]), expected_actuator_outputs[8]),
    (jnp.array([29.6565, -0.0019, -0.6011]), jnp.array([2.6837, -0.0000, -0.0323]), jnp.array([-0.0032, 0.0208, 0.0002]), jnp.array([29.6626, -0.0089, 0.0163]), jnp.array([-3.0515, -0.2090, -29.4287]), jnp.array([-76.6521, -1.1221, -70.3684]), jnp.array([-4.3158, -21.5078, 1.4193]), expected_actuator_outputs[9]),
    (jnp.array([29.6261, -0.0024, -0.6243]), jnp.array([2.9801, -0.0001, -0.0384]), jnp.array([-0.0040, 0.0117, 0.0003]), jnp.array([29.6313, -0.0111, -0.2781]), jnp.array([-3.2104, -0.2409, -29.3826]), jnp.array([-73.8203, -1.3749, -48.4869]), jnp.array([-4.4712, -13.4886, 1.6910]), expected_actuator_outputs[10])
])
'''
@pytest.mark.parametrize(
    "vned, xned, euler, vb, ab, expected_forces, expected_moments, expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values, expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values",
    zip(
        vned_values, xned_values, euler_values, vb_values, ab_values, expected_forces_values, expected_moments_values,
        expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values,
        expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values
    )
)


def test_mavrik_aero(mavrik_aero, control_inputs, vned, xned, euler, vb, ab, expected_forces, expected_moments, expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values, expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values):
    state = StateVariables(
        u=vb[0], v=vb[1], w=vb[2],
        Xe=xned[0], Ye=xned[1], Ze=xned[2],
        roll=euler[0], pitch=euler[1], yaw=euler[2],
        VXe=vned[0], VYe=vned[1], VZe=vned[2],
        p=0.0, q=0.0, r=0.0,
        Fx=0.0, Fy=0.0, Fz=0.0,
        L=0.0, M=0.0, N=0.0
    )

    forces, moments, actuator_outputs = mavrik_aero(state, control_inputs)
    actuator_outputs_array = jnp.array([
        actuator_outputs.U, actuator_outputs.alpha, actuator_outputs.beta,
        actuator_outputs.p, actuator_outputs.q, actuator_outputs.r,
        actuator_outputs.wing_alpha, actuator_outputs.wing_beta, actuator_outputs.wing_RPM,
        actuator_outputs.left_alpha, actuator_outputs.right_alpha,
        actuator_outputs.left_beta, actuator_outputs.right_beta,
        actuator_outputs.wing_prop_alpha, actuator_outputs.wing_prop_beta,
        actuator_outputs.tail_alpha, actuator_outputs.tail_beta, actuator_outputs.tail_RPM,
        actuator_outputs.tailLeft_alpha, actuator_outputs.tailRight_alpha,
        actuator_outputs.tailLeft_beta, actuator_outputs.tailRight_beta,
        actuator_outputs.tail_prop_alpha, actuator_outputs.tail_prop_beta,
        actuator_outputs.Q, actuator_outputs.aileron, actuator_outputs.elevator,
        actuator_outputs.flap, actuator_outputs.rudder, actuator_outputs.wing_tilt,
        actuator_outputs.tail_tilt, actuator_outputs.RPM_tailLeft, actuator_outputs.RPM_tailRight,
        actuator_outputs.RPM_leftOut1, actuator_outputs.RPM_left2, actuator_outputs.RPM_left3,
        actuator_outputs.RPM_left4, actuator_outputs.RPM_left5, actuator_outputs.RPM_left6In,
        actuator_outputs.RPM_right7In, actuator_outputs.RPM_right8, actuator_outputs.RPM_right9,
        actuator_outputs.RPM_right10, actuator_outputs.RPM_right11, actuator_outputs.RPM_right12Out
    ])

    F0, M0 = mavrik_aero.Ct(actuator_outputs)
    Ct_array = jnp.array([F0.Fx, F0.Fy, F0.Fz, M0.L, M0.M, M0.N])
    F1 = mavrik_aero.Cx(actuator_outputs)
    Cx_array = jnp.array([F1.Fx, F1.Fy, F1.Fz])
    F2 = mavrik_aero.Cy(actuator_outputs)
    Cy_array = jnp.array([F2.Fx, F2.Fy, F2.Fz])
    F3 = mavrik_aero.Cz(actuator_outputs)
    Cz_array = jnp.array([F3.Fx, F3.Fy, F3.Fz])
    M1 = mavrik_aero.L(actuator_outputs)
    Cl_array = jnp.array([M1.L, M1.M, M1.N])
    M2 = mavrik_aero.M(actuator_outputs)
    Cm_array = jnp.array([M2.L, M2.M, M2.N])
    M3 = mavrik_aero.Kq(actuator_outputs)
    Kq_array = jnp.array([M3.L, M3.M, M3.N])
    

    
    print("Testing Results:")

    Cx_close = jnp.allclose(Cx_array, expected_Cx_outputs_values, atol=1)
    print("Cx Outputs close???", Cx_close)
    if not Cx_close:
        print(f"\n  Expected: {expected_Cx_outputs_values}\n  Got: {Cx_array}")
        max_diff_index_Cx = jnp.argmax(jnp.abs(Cx_array - expected_Cx_outputs_values))
        print(f"\n  Max difference in Cx at index {max_diff_index_Cx}: Expected {expected_Cx_outputs_values[max_diff_index_Cx]}, Got {Cx_array[max_diff_index_Cx]}")

    Cy_close = jnp.allclose(Cy_array, expected_Cy_outputs_values, atol=1)
    print("Cy Outputs close???", Cy_close)
    if not Cy_close:
        print(f"\n  Expected: {expected_Cy_outputs_values}\n  Got: {Cy_array}")
        max_diff_index_Cy = jnp.argmax(jnp.abs(Cy_array - expected_Cy_outputs_values))
        print(f"\n  Max difference in Cy at index {max_diff_index_Cy}: Expected {expected_Cy_outputs_values[max_diff_index_Cy]}, Got {Cy_array[max_diff_index_Cy]}")

    Cz_close = jnp.allclose(Cz_array, expected_Cz_outputs_values, atol=1)
    print("Cz Outputs close???", Cz_close)
    if not Cz_close:
        print(f"\n  Expected: {expected_Cz_outputs_values}\n  Got: {Cz_array}")
        max_diff_index_Cz = jnp.argmax(jnp.abs(Cz_array - expected_Cz_outputs_values))
        print(f"\n  Max difference in Cz at index {max_diff_index_Cz}: Expected {expected_Cz_outputs_values[max_diff_index_Cz]}, Got {Cz_array[max_diff_index_Cz]}")

    Cl_close = jnp.allclose(Cl_array, expected_Cl_outputs_values, atol=1)
    print("Cl Outputs close???", Cl_close)
    if not Cl_close:
        print(f"\n  Expected: {expected_Cl_outputs_values}\n  Got: {Cl_array}")
        max_diff_index_Cl = jnp.argmax(jnp.abs(Cl_array - expected_Cl_outputs_values))
        print(f"\n  Max difference in Cl at index {max_diff_index_Cl}: Expected {expected_Cl_outputs_values[max_diff_index_Cl]}, Got {Cl_array[max_diff_index_Cl]}")

    Cm_close = jnp.allclose(Cm_array, expected_Cm_outputs_values, atol=1)
    print("Cm Outputs close???", Cm_close)
    if not Cm_close:
        print(f"\n  Expected: {expected_Cm_outputs_values}\n  Got: {Cm_array}")
        max_diff_index_Cm = jnp.argmax(jnp.abs(Cm_array - expected_Cm_outputs_values))
        print(f"\n  Max difference in Cm at index {max_diff_index_Cm}: Expected {expected_Cm_outputs_values[max_diff_index_Cm]}, Got {Cm_array[max_diff_index_Cm]}")

    Kq_close = jnp.allclose(Kq_array, expected_Kq_outputs_values, atol=1)
    print("Kq Outputs close???", Kq_close)
    if not Kq_close:
        print(f"\n  Expected: {expected_Kq_outputs_values}\n  Got: {Kq_array}")
        max_diff_index_Kq = jnp.argmax(jnp.abs(Kq_array - expected_Kq_outputs_values))
        print(f"\n  Max difference in Kq at index {max_diff_index_Kq}: Expected {expected_Kq_outputs_values[max_diff_index_Kq]}, Got {Kq_array[max_diff_index_Kq]}")

    actuator_close = jnp.allclose(actuator_outputs_array, expected_actuator_outputs_values, atol=1)
    print('Actuator Outputs close???', actuator_close)
    if not actuator_close:
        print(f"\n  Expected: {expected_actuator_outputs_values}\n  Got: {actuator_outputs_array}")
        max_diff_index = jnp.argmax(jnp.abs(actuator_outputs_array - expected_actuator_outputs_values))
        print(f"\n  Max difference at index {max_diff_index}: Expected {expected_actuator_outputs_values[max_diff_index]}, Got {actuator_outputs_array[max_diff_index]}\n\n")
     
    forces_array = jnp.array([forces.Fx, forces.Fy, forces.Fz])
    moments_array = jnp.array([moments.L, moments.M, moments.N])

    forces_close = jnp.allclose(forces_array, expected_forces, atol=1)
    print("Forces close???", forces_close)
    if not forces_close:
        print( f"\n  Expected: {expected_forces}\n  Got: {forces_array}")
        max_diff_index_forces = jnp.argmax(jnp.abs(forces_array - expected_forces))
        print(f"\n  Max difference in forces at index {max_diff_index_forces}: Expected {expected_forces[max_diff_index_forces]}, Got {forces_array[max_diff_index_forces]}")

    moments_close =  jnp.allclose(moments_array, expected_moments, atol=1)
    print("Moments close???", moments_close)
    if not moments_close:
        print(f"\n  Expected: {expected_moments}\n  Got: {moments_array}")
        max_diff_index_moments = jnp.argmax(jnp.abs(moments_array - expected_moments))
        print(f"\n  Max difference in moments at index {max_diff_index_moments}: Expected {expected_moments[max_diff_index_moments]}, Got {moments_array[max_diff_index_moments]}")