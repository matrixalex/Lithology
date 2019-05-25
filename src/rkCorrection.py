import numpy as np
from scipy.linalg import lstsq
from scipy import interpolate
import math

GK_TC_DS_DS = np.array((0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
                        0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
                        0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36))

GK_TC_DS_MUDDENS = np.array((1.07, 1.19, 1.31, 1.43, 1.55, 1.66, 1.78, 1.9))

GK_TC_DS = np.array(
    ((0.817, 0.817, 0.817, 0.817, 0.817, 0.817, 0.817, 0.817),
     (0.840681363, 0.845405096, 0.849126825, 0.852705411, 0.85642714, 0.859719439, 0.863298025, 0.86687661),
     (0.851408029, 0.860994608, 0.868484122, 0.875674056, 0.88286399, 0.889454763, 0.896644697, 0.903984422),
     (0.862280702, 0.876754386, 0.887865497, 0.898684211, 0.909502924, 0.919298246, 0.930116959, 0.940935673),
     (0.873144777, 0.892467096, 0.90716886, 0.921590591, 0.936012321, 0.949173901, 0.963595631, 0.977877345),
     (0.883938547, 0.907960894, 0.926396648, 0.944413408, 0.962430168, 0.978910615, 0.996927374, 1.014944134),
     (0.897376332, 0.923749658, 0.943563815, 0.962694725, 0.981962285, 0.999590052, 1.018720962, 1.037988521),
     (0.910709775, 0.939504925, 0.96059611, 0.981055822, 1.00138924, 1.020080828, 1.040540541, 1.061000253),
     (0.924097538, 0.955295243, 0.977767153, 0.999402343, 1.020918001, 1.04076022, 1.06239541, 1.083911069),
     (0.937569265, 0.971185815, 0.994828223, 1.017731806, 1.040512252, 1.061322497, 1.084226081, 1.106883389),
     (0.950957854, 0.98697318, 1.012005109, 1.036015326, 1.060025543, 1.081992337, 1.106002554, 1.129885057),
     (0.965688712, 1.002859274, 1.029090005, 1.05420189, 1.079438091, 1.102560915, 1.127797116, 1.152909),
     (0.980327869, 1.018822101, 1.046144505, 1.072616879, 1.09896782, 1.123132969, 1.149605343, 1.175956284),
     (0.995126112, 1.034726453, 1.063238699, 1.090898014, 1.118557329, 1.1437797, 1.171439015, 1.198976483),
     (1.009691935, 1.050536518, 1.080304604, 1.109034268, 1.137879312, 1.164301373, 1.193031037, 1.221991462),
     (1.024428113, 1.066452748, 1.097401925, 1.127419522, 1.157437118, 1.1849705, 1.214988096, 1.244902184),
     (1.038491073, 1.080437704, 1.112017662, 1.142637742, 1.173161835, 1.201286235, 1.231810328, 1.262430409),
     (1.052383493, 1.094361437, 1.126467449, 1.157684098, 1.188900747, 1.217449306, 1.248665955, 1.27997154),
     (1.066509088, 1.108484799, 1.141168799, 1.172956231, 1.204743663, 1.233922895, 1.265710327, 1.297497759),
     (1.080487992, 1.122458375, 1.155681731, 1.188061076, 1.220517149, 1.250211003, 1.282590348, 1.314969692),
     (1.094395941, 1.136407731, 1.170136557, 1.203193792, 1.236176405, 1.26639803, 1.299380643, 1.332437878),
     (1.108390023, 1.150415722, 1.18473167, 1.218291761, 1.251927438, 1.282690854, 1.316326531, 1.349962207),
     (1.12248, 1.16448, 1.19936, 1.23352, 1.26776, 1.29904, 1.33328, 1.36744),
     (1.136379251, 1.178375129, 1.213758159, 1.248625902, 1.283407764, 1.315269667, 1.350051529, 1.384919272),
     (1.1504329, 1.192460317, 1.228445166, 1.263798701, 1.299242424, 1.331709957, 1.367063492, 1.402417027),
     (1.164459996, 1.206449782, 1.242954416, 1.278891621, 1.314923397, 1.347928882, 1.383866087, 1.419992434),
	 (1.174459996, 1.306449782, 1.342954416, 1.378891621, 1.414923397, 1.447928882, 1.483866087, 1.519992434))).T

def _create2dPallete(x, y, z):
    xTmp = np.hstack(([-0.1], x, [100]))
    yTmp = np.hstack(([-0.1], y, [100]))
    zTmp = np.hstack((z[:, [0]], z, z[:,[-1]]))
    zTmp = np.vstack((zTmp[[0], :], zTmp, zTmp[[-1], :]))
    xx, yy = np.meshgrid(x, y)
    return interpolate.interp2d(xTmp, yTmp, zTmp)

gkDsPallete = _create2dPallete(GK_TC_DS_DS, GK_TC_DS_MUDDENS, GK_TC_DS)




#палетка для ввода поправок за каверны, приборы ДРСТ-1,1М,2,3, НГГК-62,СП-62, ТРКУ-100, Р3
NK_TC_WASHOUT = np.array(
    ((0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
     (0.00, 0.70, 0.92, 0.92, 0.92, 0.74, 0.54, 0.28, 0.00)))
#поправка за толщину глинистой корки}
NK_TC_MUD_DS = np.array((0.16, 0.20, 0.24, 0.28))

NK_TC_MUD_MUD = np.array((0.01, 0.02))

NK_TC_MUD_KP = np.array((0.00, 0.15, 0.30))

#ДРСТ-1,1М,2,НГГК-62,СП-62

NK_TC_MUD_DRST1 = np.array((
    ((0.005, 0.008, 0.010), (0.005, 0.008, 0.010)),
    ((0.008, 0.011, 0.010), (0.012, 0.015, 0.013)),
    ((0.015, 0.018, 0.016), (0.019, 0.022, 0.021)),
    ((0.020, 0.023, 0.022), (0.025, 0.030, 0.026))))

 # ДРСТ-3

NK_TC_MUD_DRST3 = np.array(
    (((0.011, 0.015, 0.011), (0.022, 0.027, 0.023)),
    ((0.014, 0.020, 0.018), (0.028, 0.035, 0.030)),
    ((0.020, 0.028, 0.021), (0.036, 0.049, 0.041)),
    ((0.027, 0.035, 0.020), (0.043, 0.058, 0.048)))
 )
# ТРКУ-100
NK_TC_MUD_TRKU = (
    (((0.002, 0.005, 0.004), (0.004, 0.007, 0.005)),
     ((0.008, 0.014, 0.010), (0.009, 0.015, 0.008)),
     ((0.009, 0.015, 0.010), (0.018, 0.022, 0.020)),
     ((0.022, 0.025, 0.023), (0.040, 0.045, 0.042)))
)
#Р3
NK_TC_MUD_R3 = np.array(
    (((0.004, 0.006, 0.005), (0.007, 0.010, 0.008)),
     ((0.006, 0.008, 0.007), (0.018, 0.030, 0.025)),
     ((0.008, 0.012, 0.010), (0.023, 0.042, 0.040)),
     ((0.010, 0.018, 0.015), (0.025, 0.050, 0.045)))
)

def _createNkMudTcDict():
    nkMudTcDict = {}
    drst1ToolNames =["ДРСТ-1", "ДРСТ1", "ДРСТ-1М", "ДРСТ-2", "ДРСТ2", "НГГК-62", "СП-62", "НГГК", "НГГК5", "НГГК53", "НГГК55", "НГГК 55", "НГГК 56", "НГГК 57" "НГГК57", "ДРСТ"]
    for toolName in drst1ToolNames:
        nkMudTcDict[toolName] = NK_TC_MUD_DRST1

    drst3ToolNames =["ДРСТ-3", "ДРСТ-4", "ДРСТ4", "ДРСТ3", "ДРСТ360", "ДРСТ390", "СРК", "РК", "РКС"]
    for toolName in drst3ToolNames:
        nkMudTcDict[toolName] = NK_TC_MUD_DRST3

    RK3ToolNames =["МАРК", "РК-3", "РКС-3", "РК4", "РК-4", "РКС-3М", "РКМ3", "РКМ", "РКС5", "РК-5", "РК5", "Рк5", "Рк-5", "РК8", "Р3", "МАРК-7", "КСАТ РК4", "КСАТ-4", "КСАТ РК-5", "КСАТ-РК4"]
    for toolName in RK3ToolNames:
        nkMudTcDict[toolName] = NK_TC_MUD_R3

    nkMudTcDict["ТРКУ-100"] = NK_TC_MUD_TRKU 

    return nkMudTcDict

nkMudTcDict = _createNkMudTcDict()

def lagCorrection(depth, y, speed, tau):
    derivativeSpan = 5
    dGk = lsDerivative(depth, y, derivativeSpan)
    return y - speed / 3600 * tau * dGk

def lsDerivative(x, y, span):
    dy = np.ones(y.shape) * np.nan
    halfSpan = math.floor(span / 2)
    for i in range(halfSpan, y.size - halfSpan):
        x_tmp = np.hstack((x[(i - halfSpan) : (i + halfSpan+1)].reshape(-1,1) - x[(i - halfSpan)], np.ones((span, 1))))
        y_tmp = y[i - halfSpan : i + halfSpan + 1]
        if np.where(np.isnan(y_tmp))[0].size == 0:
            dy[i] = lstsq(x_tmp, y_tmp)[0][0]
        else:
            dy[i] = np.nan
    return dy

def wellDiameterCorrectionGk(gk, ds, dsNom, mudDens):
    if ds is None:
        ds = np.ones(gk.shape) * dsNom
    else:
        ds[np.isnan(ds)] = dsNom
    gkCorrected = np.ones(gk.shape) * np.nan
    for i, gkValue in enumerate(gk):
        gkCorrected[i] = gkValue * gkDsPallete(ds[i], mudDens)
    return gkCorrected

def gkLevelNgkCorrection(ngk, gk, gkUe, ngkUe, alpha):
    numericTypes = [int, float]
    if (type(ngkUe) in numericTypes and type(gkUe) in numericTypes and
        type(alpha) in numericTypes):
        return (ngk * ngkUe - alpha * gk *gkUe) / ngkUe
    else:
        return ngk

def getWashoutValue(akp):
    n = math.floor(akp / 0.05)
    if n < 0 or n >= 8 :
        return 0
    else:
        return (NK_TC_WASHOUT[1, n] + ((akp - NK_TC_WASHOUT[0, n]) * (NK_TC_WASHOUT[1, n+1] - NK_TC_WASHOUT[1,n])) / 0.05) / 100

def getMudPalleteValue(ads, amud, akp, tool):
    ds = ads
    if ds <= NK_TC_MUD_DS[0]:
        dsn1 = 0
        ds = NK_TC_MUD_DS[0]
        dsn2 = 0
    elif ds >= NK_TC_MUD_DS[3]:
        dsn1 = 3
        dsn2 = 3
        ds = NK_TC_MUD_DS[3]
    else:
        for i in range(3):
            if (NK_TC_MUD_DS[i] <= ds) and (ds <= NK_TC_MUD_DS[i+1]):
                dsn1 = i
                dsn2 = i + 1

    pallete1 = _create2dPallete(NK_TC_MUD_MUD, NK_TC_MUD_KP,
                                nkMudTcDict[tool][dsn1].T)
    pallete2 = _create2dPallete(NK_TC_MUD_MUD, NK_TC_MUD_KP,
                                nkMudTcDict[tool][dsn2].T)
    v1 = pallete1(amud, akp)
    v2 = pallete2(amud, akp)

    return v1 + (ds - NK_TC_MUD_DS[dsn1]) * (v2 - v1) / 0.04



def getNkPalleteValue(ads, adn, akp, tool):
    dDs = adn - ads
    if dDs == 0:
        return 0
    elif dDs <0:
        return getWashoutValue(akp)
    else:
        return getMudPalleteValue(ads, dDs, akp, tool)

def wellNgkCorrection(ngk, ds, dnom, tool):
    KP_GL = 0.4
    KP_PL = 0.01
    ngkMin = np.nanmin(ngk)
    ngkMax = np.nanmax(ngk)
    t1 = math.log(KP_PL)
    t2 = math.log(KP_PL / KP_GL) /(ngkMax - ngkMin)
    t3 = (ngkMax - ngkMin) / math.log(KP_PL / KP_GL)
    ngkCorrected = np.ones(ngk.shape) * np .nan
    for i in range(ngk.size):
        if np.isnan(ngk[i]) or np.isnan(ds[i]):
            ngkCorrected[i] = ngk[i]
            continue

        akp = math.exp(t1 - (ngkMax - ngk[i]) *t2 )
        ypal = getNkPalleteValue(ds[i], dnom, akp, tool)
        akp = akp - ypal * akp
        ngkCorrected[i] = ngkMax - (math.log(KP_PL) - math.log(akp)) * t3
    return ngkCorrected

def Smoothing(y, x, h):
    '''y = Method_curve, x = Z_curve, h = tvdss window'''
                                #central defference with 'h'-tvdss window
    #get non-NaN
    y = y[np.isfinite(x)]
    x = x[(np.isfinite(x))&(np.isfinite(y))]
    y = y[np.isfinite(y)]
    #get deriv y'_x
    der = np.zeros(len(y))   
    der[0] = (y[1] - y[0])/(x[0] - x[1])
    if h > 0:
        for i in range(1, len(y[x >= x[0] - h])):
            der[i] = (y[i+1] - y[i-1])/(x[i-1] - x[i+1])
        for i in range(len(y[x >= x[0] - h]) + 1, len(y[x >= x[-1] + h])):
            der[i] = (y[x >= x[i+1] - h][-1] - y[x <= x[i-1] + h][0])/(x[x <= x[i-1] + h][0] - x[x >= x[i+1] - h][-1])
        for i in range(len(y[x >= x[-1] + h]) + 1, len(y)-1):
            der[i] = (y[i+1] - y[i-1])/(x[i-1] - x[i+1])
    elif h == 0:
        for i in range(1, len(y) - 1):
            der[i] = (y[i+1] - y[i-1])/(x[i-1] - x[i+1])
    der[-1] = (y[-1] - y[-2])/(x[-2] - x[-1])
                                 #integrating within trapeze    
    crv = np.zeros(len(y))
    crv[0] = y[0]
    for i in range(1, len(y)):
        crv[i] = crv[i-1] + (der[i] + der[i-1])*(x[i-1] - x[i])/2
    return crv

def gkCorrection(depth, gk, ds, dNom, speed, tau, mudDens):
    gkCorrectedAp = lagCorrection(depth, gk, speed, tau)
    gkCorrectedSt = wellDiameterCorrectionGk(gkCorrectedAp, ds, dNom, mudDens)
    return gkCorrectedSt, gkCorrectedAp

def ngkCorrection(depth, ngk, gk,  ds, speed, tau, gkUe, ngkUe, alpha, dnom, tool, isNgk):
    ngkCorrectedAp = lagCorrection(depth, ngk, speed, tau)
    if not isNgk:
        return ngkCorrectedAp, ngkCorrectedAp

    ngkCorrected = gkLevelNgkCorrection(ngkCorrectedAp, gk, gkUe, ngkUe, alpha)
    ngkCorrected = wellNgkCorrection(ngkCorrectedAp, ds, dnom, tool)

    return ngkCorrected, ngkCorrectedAp
