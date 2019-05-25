import numpy as np
import lasio
import xlsxwriter

class Well:
    def __init__(self, filename):

        #готовой функцией библиотеки lasio считываем лас-файл, далее будем из него вытягивать данные
        self.lasObject = lasio.read(filename)#, autodetect_encoding = True)

        #считываем с полученного лас-объекта все кривые(геофизические методы ГК НГК....)
        self.curves = {curve.mnemonic : curve.data for curve in self.lasObject.curves}

        #считываем с лас-объекта имя скважины
        self.name = str(self.lasObject.well["WELL"].value) 

        PZ_names = ["KS", "КС", "ПЗ", "pz"]
        for name in PZ_names:
            if name in self.curves.keys():
                self.curves["PZ"] = self.curves[name]
                
        PS_names = ["SP", "ПС", "ps"]
        for name in PS_names:
            if name in self.curves.keys():
                self.curves["PS"] = self.curves[name]
                
        DS_names = ["CALI", "ДС", "KV", "ds"]
        for name in DS_names:
            if name in self.curves.keys():
                self.curves["DS"] = self.curves[name]
                
        GK_names = ["GR", "ГК", "gk"]
        for name in GK_names:
            if name in self.curves.keys():
                self.curves["GK"] = self.curves[name]
                
        NGK_names = ["NGL", "НГК", "NKDT", "JB", "JM", "ngk"]
        for name in NGK_names:
            if name in self.curves.keys():
                self.curves["NGK"] = self.curves[name]

        if "MD" in self.curves.keys():
            self.curves["DEPT"] = self.curves["MD"]
        if "GK" in self.curves:
            gkLim = (np.nanmin(self.curves["GK"]), np.nanmax(self.curves["GK"]))
            self.curves["GKNORM"] = (self.curves["GK"] - gkLim[0]) / (gkLim[1] - gkLim[0])
        if "NGK" in self.curves:
            ngkLim = (np.nanmin(self.curves["NGK"]), np.nanmax(self.curves["NGK"] * 1))
            self.curves["NGKNORM"] = (self.curves["NGK"] - ngkLim[0]) / (ngkLim[1]- ngkLim[0])
        if "PZ" in self.curves:
            pzLim = (np.nanmin(self.curves["PZ"]), np.nanmax(self.curves["PZ"] * 1))
            self.curves["PZNORM"] = (self.curves["PZ"] - pzLim[0]) / (pzLim[1] - pzLim[0])
        if "PS" in self.curves:
            psLim = (np.nanmin(self.curves["PS"]), np.nanmax(self.curves["PS"] * 1))
            self.curves["PSNORM"] = (self.curves["PS"] - psLim[0]) / (psLim[1] - psLim[0])
        if "DS" in self.curves:
            dsLim = (np.nanmin(self.curves["DS"]), np.nanmax(self.curves["DS"] * 1))
            self.curves["DSNORM"] = (self.curves["DS"] - dsLim[0]) / (dsLim[1] - dsLim[0])
        self.name = str(self.lasObject.well["WELL"].value)
        if "GK" in self.curves:
            self.depthLimits = (np.min(self.curves["DEPT"][np.logical_not(np.isnan(self.curves["GK"]))]),
                np.max(self.curves["DEPT"][np.logical_not(np.isnan(self.curves["GK"]))]))
